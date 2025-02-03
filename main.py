import torch
import argparse
import logging  
from logger_and_utils import setup_logger, save_model, get_output_folder, get_embedding_dim
from model_and_data_setup import load_model_and_tokenizer, load_and_preprocess_dataset, collate_fn
from training_and_evaluation import train_fne, evaluate_fne, train_regular, evaluate_regular, train_xval, evaluate_xval, train_vanilla, evaluate_vanilla
from torch.utils.data import DataLoader
from number_encoders.FNE import FNE
from number_encoders.XVAL import XVAL
from number_encoders.vanilla import VanillaEmbedding
from fractions import Fraction
import wandb
import pdb
import os
from huggingface_hub import HfApi
from transformers import get_scheduler

from datasets import load_dataset
import re
def extract_max_num_from_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    # Initialize a variable to hold the maximum number found
    max_number = float('-inf')  # Use negative infinity as the initial value

    # Function to extract numbers from a single example
    def find_max_number(example):
        # Combine all values into a single string and extract numbers using regex
        all_values = ' '.join(map(str, example.values()))
        numbers = [float(num) for num in re.findall(r'\d+\.?\d*', all_values)]
        # Return a dictionary with the maximum number found, or None if no numbers
        return {"max_number": max(numbers, default=None)}

    # Iterate through the dataset and find the maximum number
    for split in dataset.keys():  # Loop through dataset splits like 'train', 'test', etc.
        max_in_split = dataset[split].map(find_max_number)
        # Extract valid numbers from the mapped results
        max_numbers = [num["max_number"] for num in max_in_split if num["max_number"] is not None]
        if max_numbers:
            max_number = max(max_number, max(max_numbers))

    if max_number == float('-inf'):
        raise ValueError(f"Could not extract max number from dataset: {dataset_name}")
    logging.info(f"max number : {max_number}")
    return max_number

# Set the Hugging Face token as an environment variable
os.environ["HF_TOKEN"] = "{yourhftoken}"


# os.environ["WANDB_API_KEY"] = '{yourwandbkey}'
# wandb.login()

# Helper function to convert a list of strings to floats, including fractions
def parse_period_base_list(period_base_list):
    return [float(Fraction(base)) for base in period_base_list]

def load_and_prepare_data(args, tokenizer):
    train_data, test_data = load_and_preprocess_dataset(
        args.dataset, tokenizer, args.num_train_samples, args.num_test_samples, method=args.method
    )
    logging.info(f"2 data example: {train_data[:2]}")
    num_token = tokenizer.convert_tokens_to_ids("[NUM]") if (args.method == 'fne' or args.method == 'xval' or args.method == 'vanilla' )else None
    return train_data, test_data, num_token

def create_data_loaders(train_data, test_data, tokenizer, num_token, args):
    if args.num_train_samples == 0:
        logging.info("Skipping training as num_train_samples is set to 0.")
        train_loader = None  # Not needed, so we do not create it
    else:
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, num_token, method=args.method)
        )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, num_token, method=args.method)
    )
    return train_loader, test_loader

def initialize_optimizer_and_scheduler(model, train_loader, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.2 * total_steps)  # 20% warmup
    scheduler = get_scheduler(
        name=args.scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler

def run_epoch(model, train_loader, test_loader, optimizer, scheduler, number_encoder, args, epoch, device):
    if args.method == 'regular':
        train_loss = train_regular(model, train_loader, optimizer, scheduler, device, args)
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_regular(
            model, test_loader, tokenizer, device, print_labels=True, max_print_examples=5
        )
    elif args.method == 'fne':
        train_loss = train_fne(model, train_loader, number_encoder, optimizer, scheduler, args, args.int_digit_len, args.frac_digit_len, args.len_gen_size, device)
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_fne(
            model, test_loader, number_encoder, args.int_digit_len, args.frac_digit_len, device, print_labels=True, max_print=5
        )
    elif args.method == 'xval':
        train_loss = train_xval(model, train_loader, number_encoder, optimizer, scheduler, args, device)
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_xval(
            model, test_loader, number_encoder, device, print_labels=True, max_print=5
        )
    elif args.method == 'vanilla':
        train_loss = train_vanilla(model, train_loader, number_encoder, optimizer, scheduler, args, device)
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_vanilla(
            model, test_loader, number_encoder, device, print_labels=True, max_print=5
        )
    else:
        raise ValueError(f"Unsupported method '{args.method}'. Choose from ['regular', 'fne', 'xval'].")

    # Log results to the console
    logging.info(f"{'Epoch':<40}{epoch + 1:<10}")
    logging.info(f"{'Train Loss':<40}{train_loss:.4f}")
    logging.info(f"{'Test Loss':<40}{test_loss:.4f}")
    logging.info(f"{'Whole Num Acc':<40}{whole_number_accuracy * 100:.6f}%")
    logging.info(f"{'Digit-wise Acc':<40}{digit_wise_accuracy * 100:.6f}%")
    logging.info(f"{'MSE':<40}{mse:.6f}")
    logging.info(f"{'R^2':<40}{r2:.15f}")
    logging.info(f"{'Learning Rate':<40}{scheduler.get_last_lr()[0]:.6f}")
    # Log results to WandB
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "whole_number_accuracy": whole_number_accuracy * 100,
        "digit_wise_accuracy": digit_wise_accuracy * 100,
        "mse": mse,
        "r2": r2,
        "learning_rate": scheduler.get_last_lr()[0]
    })

    return whole_number_accuracy
def evaluate_model(model, test_loader, tokenizer, number_encoder, args, device, stage="Initial"):
    logging.info(f"Starting {stage} evaluation.")
    
    if args.method == 'regular':
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_regular(
            model, test_loader, tokenizer, device, print_labels=True, max_print_examples=10
        )
    elif args.method == 'fne':
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_fne(
            model, test_loader, number_encoder, args.int_digit_len, args.frac_digit_len, device, print_labels=True, max_print=5
        )
    elif args.method == 'xval':
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_xval(
            model, test_loader, number_encoder, device, print_labels=True, max_print=5
        )
    else:
        raise ValueError(f"Unsupported method '{args.method}'. Choose from ['regular', 'fne', 'xval'].")

    logging.info(f"{stage} Test Results:")
    logging.info(f"{'Test Loss':<40}{test_loss:.4f}")
    logging.info(f"{'Whole Num Accuracy':<40}{whole_number_accuracy * 100:.6f}%")
    logging.info(f"{'Digit-wise Accuracy':<40}{digit_wise_accuracy * 100:.6f}%")
    logging.info(f"{'MSE':<40}{mse:.6f}")
    logging.info(f"{'R^2':<40}{r2:.15f}")
    # Log results to WandB
    wandb.log({
        f"{stage.lower()}_test_loss": test_loss,
        f"{stage.lower()}_whole_number_accuracy": whole_number_accuracy * 100,
        f"{stage.lower()}_digit_wise_accuracy": digit_wise_accuracy * 100,
        f"{stage.lower()}_mse": mse,
        f"{stage.lower()}_r2": r2
    })

    return test_loss, whole_number_accuracy, digit_wise_accuracy, mse, r2

def create_dataloader_and_train(args, model, tokenizer):
    # Data preparation
    train_data, test_data, num_token = load_and_prepare_data(args, tokenizer)
    train_loader, test_loader = create_data_loaders(train_data, test_data, tokenizer, num_token, args)

    # FNE initialization
    number_encoder = None
    embedding_dim = get_embedding_dim(model)
    if args.method == 'fne':
        number_encoder = FNE(
            embedding_dim, int_digit_len=args.int_digit_len,
            frac_digit_len=args.frac_digit_len, period_base_list=args.period_base_list,add_linear=args.add_linear, device=device
        ).to(device)
    elif args.method == 'vanilla':
        number_encoder = VanillaEmbedding(
            embedding_dim, int_digit_len=args.int_digit_len,
            frac_digit_len=args.frac_digit_len, device=device
        ).to(device)
    elif args.method == 'xval':
        max_num = extract_max_num_from_dataset(args.dataset)
        number_encoder = XVAL(embedding_dim=embedding_dim, max_num=max_num, device=device).to(device)

    # Perform initial evaluation
    if args.num_train_samples == 0:
        evaluate_model(model, test_loader, tokenizer, number_encoder, args, device, stage="Single Evaluation")
        return
    # Optimizer and scheduler
    optimizer, scheduler = initialize_optimizer_and_scheduler(model, train_loader, args)

    # Training loop
    for epoch in range(args.epochs):
        logging.info('-' * 100)
        logging.info(f"Starting Epoch {epoch + 1}/{args.epochs}")
        whole_number_accuracy = run_epoch(
            model, train_loader, test_loader, optimizer, scheduler, number_encoder, args, epoch, device
        )
        if whole_number_accuracy == 1.0:
            logging.info("Stopping early as whole number accuracy has reached 100%.")
            break

    # Perform final evaluation
    evaluate_model(model, test_loader, tokenizer, number_encoder, args, device, stage="Final")
    
    # Setup argument parser
parser = argparse.ArgumentParser(description="Training LLMs with custom embeddings and Fourier loss")
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
parser.add_argument('--int_digit_len', type=int, default=5, help='Number of digits')
parser.add_argument('--frac_digit_len', type=int, default=5, help='Number of digits')
parser.add_argument('--len_gen_size', type=int, default=0, help='fne add k 0s after numbers to len gen')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--name', type=str, default='', help='log name')
parser.add_argument('--model', type=str, default='gpt2', help='Model name')
parser.add_argument('--dataset', type=str, default='{yourid}/language_math_10base', help='Dataset name')
parser.add_argument('--train_from_scratch', action='store_true', help='Train the model from scratch without loading pre-trained weights')
parser.add_argument('--use_digit_wise_tokenizer', action='store_true', help='whether use digitwise tokenizer')
parser.add_argument('--num_train_samples', type=int, default=None, help='Number of training samples to use')
parser.add_argument('--num_test_samples', type=int, default=None, help='Number of test samples to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--model_size_level', type=int, default=-1, help='from 1 to 8, choose the size of the model train from scratch')
parser.add_argument('--method', type=str, choices=['regular', 'fne', 'xval','vanilla'], default='fne', help='Training method: regular or fne')
parser.add_argument('--scheduler_name', type=str, default='cosine',help='Name of the learning rate scheduler (e.g., linear, constant, cosine, etc.)')
parser.add_argument('--period_base_list', type=str, nargs='+', default=['2', '5'], help='List of period bases to use in Fourier embedding (e.g., 2, 5, 1/3)')
parser.add_argument('--clip', action='store_true', help='Enable clipping')
parser.add_argument('--not_add_linear', action='store_true', help='Add linear after FNE')
args = parser.parse_args()

# Convert the period_base_list from strings to floats (handle fractions like 1/3)
args.add_linear = not args.not_add_linear
args.period_base_list = parse_period_base_list(args.period_base_list)
run_name = f"{args.name}{args.method}_{args.model}_{args.dataset}_seed{args.seed}"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)
# Initialize W&B
wandb.init(
    project="FNE",
    config=vars(args),  # Convert argparse Namespace to a dictionary
    name=run_name
)
# Setup output folder and logger
output_folder = get_output_folder(args)
setup_logger(output_folder)

logging.info(args)
if ',' in args.dataset:
    args.dataset = tuple(args.dataset.split(','))
    logging.info(f"Dataset specified as tuple: {args.dataset}")
else:
    logging.info(f"Dataset specified as single name: {args.dataset}")

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    
# Load model, tokenizer, and dataset
model, tokenizer = load_model_and_tokenizer(args.model, "/home/{yourname}/hg_cache", device, args.train_from_scratch, args.model_size_level, args.use_digit_wise_tokenizer)


create_dataloader_and_train(args, model, tokenizer)

wandb.finish()
# Save model
# save_model(model, tokenizer, output_folder)
