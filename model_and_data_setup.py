import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import re
import logging
import pdb
from torch.nn.utils.rnn import pad_sequence


def split_dataset(dataset, test_size=0.1, seed=42):
    """
    Splits a dataset into train and test sets.
    """
    return dataset.train_test_split(test_size=test_size, seed=seed)
MODEL_CONFIG_TABLE = {
    1: {"hidden_size": 64, "intermediate_size": 256, "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2},
    2: {"hidden_size": 128, "intermediate_size": 512, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2},
    3: {"hidden_size": 192, "intermediate_size": 768, "num_hidden_layers": 3, "num_attention_heads": 6, "num_key_value_heads": 3},
    4: {"hidden_size": 256, "intermediate_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 4},
    5: {"hidden_size": 320, "intermediate_size": 1280, "num_hidden_layers": 5, "num_attention_heads": 8, "num_key_value_heads": 4},
    6: {"hidden_size": 384, "intermediate_size": 1536, "num_hidden_layers": 6, "num_attention_heads": 8, "num_key_value_heads": 4},
    7: {"hidden_size": 512, "intermediate_size": 2048, "num_hidden_layers": 7, "num_attention_heads": 10, "num_key_value_heads": 5},
    8: {"hidden_size": 640, "intermediate_size": 2560, "num_hidden_layers": 8, "num_attention_heads": 12, "num_key_value_heads": 6},
    9: {"hidden_size": 704, "intermediate_size": 2816, "num_hidden_layers": 8, "num_attention_heads": 12, "num_key_value_heads": 6},
    10: {"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 8, "num_attention_heads": 12, "num_key_value_heads": 6},
}



def load_model_and_tokenizer(
    model_name,
    cache_dir,
    device,
    train_from_scratch=False,
    size_level=-1,
    use_digit_wise_tokenizer=False
):
    """
    Loads a model and tokenizer, initializing from scratch if specified.
    Adjusts config to match a size_level (1 to 8) if training from scratch.
    Adds [PAD] and [NUM] tokens if not present and prints model size.
    If use_digit_wise_tokenizer, loads a digit-wise tokenizer (example: Llama-2).
    Ensures the vocab_size is consistent between model and tokenizer by
    expanding the smaller one with dummy tokens.
    """

    # ------------------------------------------------------------------
    # 1) Load/Initialize the Model
    # ------------------------------------------------------------------
    if train_from_scratch:
        if size_level == -1:
            # Load default configuration without modification
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_config(config).to(device)
            logging.info("Model initialized from scratch with default configuration.")
        else:
            if size_level not in MODEL_CONFIG_TABLE:
                raise ValueError(f"Invalid size level '{size_level}'. Available: 1 to 8")
            config_params = MODEL_CONFIG_TABLE[size_level]

            # Load base config + update with our custom params
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            config.update(config_params)

            model = AutoModelForCausalLM.from_config(config).to(device)
            logging.info(f"Model initialized from scratch with size level: {size_level}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            cache_dir=cache_dir
        ).to(device)
        logging.info("Pre-trained model loaded!")

    # ------------------------------------------------------------------
    # 2) Load the Tokenizer
    # ------------------------------------------------------------------
    if use_digit_wise_tokenizer:
        # Example: load a digit-wise tokenizer from Llama-2
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            cache_dir=cache_dir
        )
        logging.info("Digit-wise tokenizer loaded from 'meta-llama/Llama-2-7b-hf'.")
    else:
        # Load standard tokenizer from model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        logging.info("Standard tokenizer loaded.")

    # ------------------------------------------------------------------
    # 3) Add Any Extra Special Tokens You Need
    # ------------------------------------------------------------------
    #  - Example: [PAD], [NUM]
    special_tokens_added = False
    if '[PAD]' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        special_tokens_added = True
    if '[NUM]' not in tokenizer.get_vocab():
        tokenizer.add_tokens(["[NUM]"])
        special_tokens_added = True

    # ------------------------------------------------------------------
    # 4) Expand the Smaller One (Tokenzier vs. Model Config)
    # ------------------------------------------------------------------
    # Check if the model config has an explicit vocab_size
    # (Some models might not have it set precisely, so handle carefully.)
    config_vocab_size = getattr(model.config, "vocab_size", None)
    tokenizer_vocab_size = len(tokenizer)

    # If either is None, we can't do a direct comparison; skip expansion
    if config_vocab_size is not None:
        if config_vocab_size > tokenizer_vocab_size:
            # The model expects a bigger vocab => Expand tokenizer
            missing = config_vocab_size - tokenizer_vocab_size
            dummy_tokens = [f"[DUMMY_{i}]" for i in range(missing)]
            tokenizer.add_tokens(dummy_tokens)
            logging.info(f"Expanded tokenizer by {missing} dummy tokens "
                         f"so it matches model vocab_size={config_vocab_size}.")
        elif config_vocab_size < tokenizer_vocab_size:
            # The tokenizer is bigger => Expand model.config.vocab_size
            logging.info(f"Expanding model config from {config_vocab_size} to {tokenizer_vocab_size} "
                         "to match tokenizer.")
            model.config.vocab_size = tokenizer_vocab_size

    # Now ensure the final count is consistent
    final_vocab_size = len(tokenizer)
    model.resize_token_embeddings(final_vocab_size)

    # Optionally update model.config.vocab_size to final_vocab_size
    model.config.vocab_size = final_vocab_size

    # ------------------------------------------------------------------
    # 5) Log the Actual Model Size
    # ------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params / 1_000_000  # Convert to millions
    logging.info(f"Actual model size (total parameters): {total_params_in_millions:.2f}M")

    logging.info(f"Padding token ID: {tokenizer.pad_token_id}")
    # If you're not using digit-wise, also log the [NUM] ID
    if not use_digit_wise_tokenizer:
        num_id = tokenizer.convert_tokens_to_ids('[NUM]')
        logging.info(f"[NUM] token ID: {num_id}")

    # ------------------------------------------------------------------
    # Return Model and Final Tokenizer
    # ------------------------------------------------------------------
    return model, tokenizer


def create_scatter_tensor(batch_input_ids, batch_numbers, num_token_id):
    scatter_tensor_batch = []
    for input_ids, numbers in zip(batch_input_ids, batch_numbers):
        scatter_tensor = torch.zeros(len(input_ids), dtype=torch.float64)
        num_index = 0
        for i, token_id in enumerate(input_ids):
            if token_id == num_token_id:
                if num_index < len(numbers):
                    scatter_tensor[i] = numbers[num_index]
                    num_index += 1
        scatter_tensor_batch.append(scatter_tensor)
    return torch.stack(scatter_tensor_batch)


def preprocess(entry, is_tabular, method='regular'):
    """
    Preprocesses an entry based on its format and the selected method ('fne' or 'regular').
    """
    def preprocess_entry(entry):
        keys = list(entry.keys())
        features, label_key = keys[:-1], keys[-1]
        question = ' '.join(f'{entry[key]},' for key in features).strip(',')
        label = entry[label_key]
        return question, label

    question, label = preprocess_entry(entry) if is_tabular else (str(entry["question"]), float(entry["label"]))

    if method == 'fne' or method == 'xval' or method =='vanilla':
        label = float(label)
        numbers = [float(num) for num in re.findall(r'\d+\.?\d*', question)]
        # if ((round(numbers[0]*1000)+round(numbers[1]*1000)) !=  round(label*1000)):
        question = re.sub(r'\d+\.?\d*', ' [NUM] ', question)
        return {'question_with_num': question, 'numbers': numbers, 'label': label}
    elif method == 'regular':
        return {'question': question, 'label': label}


def load_and_preprocess_dataset(dataset_name, tokenizer, num_train_samples=None, num_test_samples=None, method='regular'):
    """
    Loads and preprocesses a dataset for 'fne' or 'regular' methods.
    """
    is_tabular = 'tabular' in dataset_name if isinstance(dataset_name, str) else 'tabular' in dataset_name[0]
    dataset = load_dataset(dataset_name) if isinstance(dataset_name, str) else load_dataset(*dataset_name)

    if 'test' not in dataset:
        logging.info("Splitting train set into train and test splits.")
        dataset = split_dataset(dataset['train'])

    logging.info(f"Train dataset length: {len(dataset['train'])}, Test dataset length: {len(dataset['test'])}")

    def preprocess_entry(entry):
        """
        Preprocesses a single entry based on the method ('fne' or 'regular').
        For 'regular', computes 'question_len' for generating loss mask and attention mask.
        """
        result = preprocess(entry, is_tabular, method)
        
        if method == 'fne' or method =='xval' or method =='vanilla':
            input_text = result['question_with_num']
            input_ids = tokenizer.encode(input_text, return_tensors="pt").squeeze(0)
            return {
                'input_ids': input_ids,
                'numbers': result.get('numbers', []),
                'label': result['label']
            }
        
        elif method == 'regular':
            question_text = result['question']
            input_text = question_text + ' ' + str(result['label'])
            input_ids = tokenizer.encode(input_text, return_tensors="pt").squeeze(0)
            question_len = len(tokenizer.encode(question_text, return_tensors="pt").squeeze())
            return {
                'input_ids': input_ids,
                'question_len': question_len
            }

    train_data = [preprocess_entry(entry) for entry in dataset['train'].select(range(num_train_samples))] if num_train_samples!=None else [
        preprocess_entry(entry) for entry in dataset['train']]
    test_data = [preprocess_entry(entry) for entry in dataset['test'].select(range(num_test_samples))] if num_test_samples!=None else [
        preprocess_entry(entry) for entry in dataset['test']]

    return train_data, test_data


def collate_fn(batch, tokenizer, num_token_id=None, max_length=128, method='regular'):
    """
    Collates a batch of data for 'fne' or 'regular' methods.
    """
    input_ids = [item['input_ids'] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    input_ids_padded = input_ids_padded[:, :max_length]
    attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()

    if method == 'fne' or method == 'xval' or method == 'vanilla':
        scatter_tensor = create_scatter_tensor(input_ids_padded, [item['numbers'] for item in batch], num_token_id)
        last_token_mask = torch.zeros_like(input_ids_padded, dtype=torch.float32)
        for i, seq in enumerate(input_ids_padded):
            last_non_pad_idx = (seq != tokenizer.pad_token_id).nonzero()[-1].item()
            last_token_mask[i, last_non_pad_idx] = 1
        return {
            'input_ids': input_ids_padded,
            'scatter_tensor': scatter_tensor,
            'attention_mask': attention_mask,
            'labels': torch.tensor([item['label'] for item in batch], dtype=torch.float64),
            'last_token_mask': last_token_mask
        }

    elif method == 'regular':
        question_lens = [item['question_len'] for item in batch]
        loss_mask = torch.zeros_like(input_ids_padded, dtype=torch.float32)
        question_lens_tensor = torch.tensor(question_lens, dtype=torch.long).unsqueeze(1)
        batch_size, seq_len = input_ids_padded.shape
        indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        loss_mask = (indices >= question_lens_tensor).float() * attention_mask.float()
            # Prepare labels to calculate loss only for specified positions
        labels = input_ids_padded.clone()
        labels[loss_mask == 0] = -100  # Use -100 to ignore tokens during loss calculation

        return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask,
        'labels': labels
        }
import torch
import logging
from torch.utils.data import DataLoader

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Parameters
    model_name = "gpt2"  # You can replace this with any compatible model name
    cache_dir = "/mnt/nfs1/{yourname}/cache"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = "{yourid}/sub_max1000_sample100000_split0.8"  # Example dataset, replace with actual dataset
    num_train_samples = 10
    num_test_samples = 5
    method = "regular"  # Change to 'regular' if not using FNE
    max_length = 128
    train_from_scratch = False

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, cache_dir, device, train_from_scratch)

    # Load and preprocess dataset
    train_data, test_data = load_and_preprocess_dataset(
        dataset_name, tokenizer, num_train_samples, num_test_samples, method
    )

    # Display first few examples of preprocessed data
    print("First few preprocessed training examples:")
    for example in train_data[:3]:
        print(example)

    # Create DataLoader
    collate_function = lambda batch: collate_fn(batch, tokenizer, tokenizer.convert_tokens_to_ids("[NUM]"), max_length, method)
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_function)

    # Display a batch from DataLoader
    print("\nBatch from DataLoader:")
    for batch in train_dataloader:
        for key, value in batch.items():
            print(f"{key}: {value}")
        break  # Display only the first batch

if __name__ == "__main__":
    main()