import torch
import logging
import pdb
import numpy as np
import torch.nn.functional as F
from logger_and_utils import print_gpu_memory_usage
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
def is_numeric(s):
    # Use a regular expression to match valid float strings
    return bool(re.match(r"^-?\d+(\.\d+)?$", s))
def handle_nan_loss(batch, before_decoder, data_idx, model, save_path="debug_nan_loss.pt"):
    """
    Handles NaN loss by saving the problematic batch and model state before the decoder.

    Args:
        batch (dict): The current input batch.
        before_decoder (torch.Tensor): The model's hidden state before the decoder step.
        data_idx (int): The index of the current data step.
        model (torch.nn.Module): The model instance.
        save_path (str): Path to save the debug data.

    Returns:
        None
    """
    debug_data = {
        "batch": batch,
        "before_decoder": before_decoder.cpu() if before_decoder is not None else None,
        "data_idx": data_idx,
        "model_state": model.state_dict(),
    }
    save_path = os.path.join("fail_case_log", save_path)
    torch.save(debug_data, save_path)
    logging.info(f"Saved debug data to {save_path}. Stopping training due to NaN loss.")
def get_regular_embeddings(model, input_ids):
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        # GPT-2 models
        return model.transformer.wte(input_ids)
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # LLaMA models
        return model.model.embed_tokens(input_ids)
    else:
        raise AttributeError(f"Cannot find token embeddings in the model: {type(model)}")
    
# ---------------------------------------------------------------------
def train_fne(model, train_loader, fne, optimizer, scheduler, args, int_digit_len, frac_digit_len, len_gen_size,  device):
    model.train()  # Set the model to training mode
    fne.train()  # Set FNE to training mode, making all parameters trainable
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        scatter_tensor = batch['scatter_tensor'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        last_token_mask = batch['last_token_mask'].to(device)
        len_gen = torch.randint(0, len_gen_size+1, (1,), device=device).item()
        
        regular_embeddings = get_regular_embeddings(model, input_ids)
        fourier_embeddings = fne(scatter_tensor, len_gen=len_gen)
        input_embeddings = regular_embeddings + fourier_embeddings

        outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        before_decoder = outputs.hidden_states[-1]
        last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)

        loss = fne.fourier_compute_loss(
            last_token_hidden_state, 
            labels, 
            int_digit_len, 
            frac_digit_len, 
            len_gen=len_gen
        )
        # if not loss.detach().isfinite():  # Check for NaN loss
        #     handle_nan_loss(
        #         batch=batch,
        #         before_decoder=before_decoder,
        #         data_idx=batch_idx,
        #         model=model,
        #         save_path="fne_debug_nan_loss.pt"
        #     )
        #     break  # Stop training due to NaN loss
        loss.backward()
        if args.clip == True:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Choose max_norm as per your task
            torch.nn.utils.clip_grad_norm_(fne.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # Update the learning rate
        optimizer.zero_grad()

        total_loss += loss.item()

    logging.info(f"avg Loss: {total_loss / len(train_loader)}")
    return total_loss / len(train_loader)
def evaluate_fne(
    model,
    test_loader,
    fne,
    int_digit_len,
    frac_digit_len,
    device,
    print_labels=False,
    max_print=10
):
    """
    Evaluates the model and logs up to `max_print` misprediction examples if `print_labels` is True.
    """
    logging.info('eval start')
    model.eval()  # Set the model to evaluation mode
    fne.eval()    # Set FNE to evaluation mode
    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0  # For MSE
    total_variance = 0       # For R^2
    total_digits = 0
    correct_digits = 0
    all_labels = []
    all_predictions = []

    # We'll store misprediction examples here.
    mispredictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)

            # Get embeddings
            regular_embeddings = get_regular_embeddings(model, input_ids)
            fourier_embeddings = fne(scatter_tensor)
            input_embeddings = regular_embeddings + fourier_embeddings

            # Forward pass
            outputs = model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            before_decoder = outputs.hidden_states[-1]

            last_token_hidden_state = (
                before_decoder * last_token_mask.unsqueeze(-1)
            ).sum(dim=1)

            # Compute predicted numbers using the Fourier embeddings
            predicted_numbers = fne.fourier_compute_prediction(
                last_token_hidden_state, int_digit_len, frac_digit_len
            )

            # Store for variance computation
            all_labels.append(labels.cpu())
            all_predictions.append(predicted_numbers.cpu())

            # Whole-number accuracy
            tolerance = 10 ** (-frac_digit_len)
            correct_predictions = torch.abs(predicted_numbers - labels) < tolerance
            total_correct += correct_predictions.sum().item()
            total_samples += labels.size(0)

            # Digit-wise accuracy
            for i in range(labels.size(0)):
                actual_value = str(labels[i].item())
                predicted_value = str(predicted_numbers[i].item())
                min_len = len(actual_value)
                correct_digits += sum(
                    1 for a, p in zip(actual_value[:min_len], predicted_value[:min_len]) if a == p
                )
                total_digits += len(actual_value)

            # Identify mispredictions
            for i in range(labels.size(0)):
                if not correct_predictions[i]:  # Outside of tolerance
                    mispredictions.append(
                        (predicted_numbers[i].item(), labels[i].item())
                    )

            # Compute squared error for MSE
            squared_error = torch.sum((predicted_numbers - labels) ** 2).item()
            total_squared_error += squared_error

            # Compute loss
            loss = fne.fourier_compute_loss(
                last_token_hidden_state, labels, int_digit_len, frac_digit_len
            )
            total_loss += loss.item()

    # Concatenate all labels and predictions
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    # Compute total variance for R^2
    mean_label = all_labels.mean().item()
    total_variance = torch.sum((all_labels - mean_label) ** 2).item()

    # Compute metrics
    avg_loss = total_loss / len(test_loader)
    whole_number_accuracy = total_correct / total_samples
    digit_wise_accuracy = correct_digits / total_digits
    mse = total_squared_error / total_samples
    r2 = (
        1 - (total_squared_error / total_variance)
        if total_variance > 0
        else float('nan')
    )

    # Finally, log misprediction examples if desired
    if print_labels:
        if len(mispredictions) == 0:
            logging.info("No mispredictions found!")
        else:
            log_count = min(len(mispredictions), max_print)
            logging.info(f"Mispredictions (showing up to {log_count} examples):")
            for i in range(log_count):
                predicted_val, actual_val = mispredictions[i]
                logging.info(f"Predicted: {predicted_val}, Actual: {actual_val}")

    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2


# ---------------------------------------------------------------------
# Regular training loop
def train_regular(model, dataloader, optimizer, scheduler, device, args):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Apply loss mask to compute loss only on the label part
        # if not masked_loss.detach().isfinite():  # Check for NaN loss
        #     logging.info('nan loss!!')
        #     pdb.set_trace()
        #     break  # Stop training due to NaN loss
        optimizer.zero_grad()
        loss.backward()

        if args.clip == True:
        # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed

        optimizer.step()
        scheduler.step()  # Update the learning rate

        total_loss += loss.item()

    logging.info(f"avg Loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)
def evaluate_regular(model, dataloader, tokenizer, device, print_labels=False, max_print_examples=10):
    logging.info('eval start')
    model.eval()
    total_loss = 0
    total_examples = 0
    total_correct_examples = 0  # For whole-number accuracy
    total_characters = 0
    correct_characters = 0
    total_squared_error = 0  # Initialize total squared error for MSE
    total_variance = 0       # Initialize total variance for R^2
    numeric_examples = 0  # Counter for numeric examples
    printed_examples = 0
    all_labels = []  # For variance computation

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            
           
            total_loss += loss.item()
            
            # Get the predicted token IDs (taking argmax over the logits)
            predictions = torch.argmax(logits, dim=-1)
            examplelist = []
            for i in range(len(input_ids)):
                # Get label portion indices where there is a valid label
                label_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
                actual_tokens = input_ids[i, label_indices].cpu().numpy()
                predicted_tokens = predictions[i, label_indices-1].cpu().numpy()
                # Decode to strings for character-wise and whole-number comparison
                actual_label = tokenizer.decode(actual_tokens, skip_special_tokens=True).strip()
                predicted_label = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()

                # Whole-number accuracy
                if actual_label == predicted_label:
                    total_correct_examples += 1
                total_examples += 1

                # Compute squared error for MSE (convert strings to floats)
                if is_numeric(predicted_label):
                    actual_value = float(actual_label)
                    predicted_value = float(predicted_label)
                    squared_error = (actual_value - predicted_value) ** 2
                    total_squared_error += squared_error
                    numeric_examples += 1

                    # Collect all ground truth values for variance calculation
                    all_labels.append(actual_value)

                # Character-wise accuracy
                max_len = max(len(actual_label), len(predicted_label))
                padded_actual = actual_label.ljust(max_len)
                padded_predicted = predicted_label.ljust(max_len)
                
                correct_characters += sum(1 for actual_char, pred_char in zip(padded_actual, padded_predicted) if actual_char == pred_char)
                total_characters += max_len

                # Collect mispredictions for logging (defer printing to outside the loop)
                if print_labels and printed_examples < max_print_examples:
                    examplelist.append(f"({predicted_label}, {actual_label})")
                    printed_examples += 1

            # Log mispredictions outside the loop
            if print_labels and examplelist:
                logging.info(" ".join(examplelist))


    avg_loss = total_loss / len(dataloader)
    whole_number_accuracy = total_correct_examples / total_examples
    digit_wise_accuracy = correct_characters / total_characters

    # Compute variance for R^2
    if numeric_examples > 0:
        mean_label = sum(all_labels) / len(all_labels)
        total_variance = sum((label - mean_label) ** 2 for label in all_labels)
        mse = total_squared_error / numeric_examples  # Mean Squared Error
        r2 = 1 - (total_squared_error / total_variance) if total_variance > 0 else float('nan')
    else:
        mse = -1
        r2 = float('nan')


    
    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2

#-------------------------------
# ---------------------------------------------------------------------
def train_xval(model, train_loader, xval, optimizer, scheduler, args, device):
    model.train()  # Set the model to training mode
    xval.train()  # Set FNE to training mode, making all parameters trainable
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        scatter_tensor = batch['scatter_tensor'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        last_token_mask = batch['last_token_mask'].to(device)

        regular_embeddings = get_regular_embeddings(model, input_ids)
        input_embeddings = xval(scatter_tensor,regular_embeddings)
        outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        before_decoder = outputs.hidden_states[-1]
        last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)

        loss = xval.compute_loss(last_token_hidden_state, labels,)
        # if not loss.detach().isfinite():  # Check for NaN loss
        #     logging.info('nan loss!')
        #     break  # Stop training due to NaN loss

        loss.backward()
        
        if args.clip == True:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Choose max_norm as per your task
            torch.nn.utils.clip_grad_norm_(xval.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # Update the learning rate
        optimizer.zero_grad()

        total_loss += loss.item()

    logging.info(f"avg Loss: {total_loss / len(train_loader)}")
    return total_loss / len(train_loader)

def evaluate_xval(model, test_loader, xval, device, print_labels=False, max_print=10):
    logging.info('eval start')
    model.eval()  # Set the model to evaluation mode
    xval.eval()  # Set xval to evaluation mode
    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0  # Initialize total squared error for MSE
    total_variance = 0       # Initialize total variance for R^2
    printed_examples = 0  # Counter for printed examples
    total_digits = 0
    correct_digits = 0
    all_labels = []  # Collect labels for variance computation

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)

            # Get embeddings
            regular_embeddings = get_regular_embeddings(model, input_ids)
            input_embeddings = xval(scatter_tensor, regular_embeddings)

            # Forward pass
            outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
            before_decoder = outputs.hidden_states[-1]

            last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)

            # Compute predicted numbers using the xval embeddings
            predicted_numbers = xval.compute_prediction(last_token_hidden_state)
            
            # Compute whole-number accuracy
            # tolerance = 0.5  # NOTE:only for xval on int task!!!
            tolerance = 0.5  # NOTE:only for xval on decimal task!!!
            correct_predictions = torch.abs(predicted_numbers - labels) < tolerance
            
            total_correct += correct_predictions.sum().item()
            total_samples += labels.size(0)

            # Collect labels for variance computation
            all_labels.extend(labels.cpu().numpy())

            # Compute digit-wise accuracy
            for i in range(labels.size(0)):
                actual_value = str(labels[i].item())
                predicted_value = str(predicted_numbers[i].item())
                min_len = len(actual_value)
                correct_digits += sum(1 for a, p in zip(actual_value[:min_len], predicted_value[:min_len]) if a == p)
                total_digits += len(actual_value)

            # Compute loss using xval.compute_loss
            loss = xval.compute_loss(last_token_hidden_state, labels)
            total_loss += loss.item()

            # Compute squared error for MSE
            squared_error = torch.sum((predicted_numbers - labels) ** 2).item()
            total_squared_error += squared_error

            # Optionally log predictions and actual labels
            if print_labels and printed_examples < max_print:
                output_pairs = []
                for i in range(len(labels)):
                    if printed_examples >= max_print:
                        break
                    actual_label = labels[i].cpu().numpy()
                    predicted_label = predicted_numbers[i].cpu().numpy()
                    output_pairs.append((predicted_label, actual_label))
                    printed_examples += 1
                logging.info("Predictions and Labels: " + " ".join(f"({pred},{lbl})" for pred, lbl in output_pairs))

    # Compute average loss, whole-number accuracy, digit-wise accuracy, and MSE
    avg_loss = total_loss / len(test_loader)
    whole_number_accuracy = total_correct / total_samples
    digit_wise_accuracy = correct_digits / total_digits
    mse = total_squared_error / total_samples

    # Compute variance for R^2
    if total_samples > 1:  # Ensure there are enough samples
        mean_label = sum(all_labels) / len(all_labels)
        total_variance = sum((label - mean_label) ** 2 for label in all_labels)
        r2 = 1 - (total_squared_error / total_variance) if total_variance > 0 else float('nan')
    else:
        r2 = float('nan')


    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2

#-------------------------------
# ---------------------------------------------------------------------
def train_vanilla(model, train_loader, vanilla_model, optimizer, scheduler, args, device):
    model.train()
    vanilla_model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        scatter_tensor = batch['scatter_tensor'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        last_token_mask = batch['last_token_mask'].to(device)
        
        # Get embeddings
        regular_embeddings = get_regular_embeddings(model, input_ids)
        vanilla_embeddings = vanilla_model(scatter_tensor)
        input_embeddings = regular_embeddings + vanilla_embeddings

        # Forward pass
        outputs = model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        
        # Get last token hidden state
        last_token_hidden_state = (last_hidden_state * last_token_mask.unsqueeze(-1)).sum(dim=1)

        # Compute loss
        loss = vanilla_model.compute_loss(last_token_hidden_state, labels)
        
        # Backpropagation
        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(vanilla_model.parameters(), max_norm=1.0)
            
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Training Loss: {avg_loss}")
    return avg_loss

def evaluate_vanilla(model, test_loader, vanilla_model, device, print_labels=False, max_print=10):
    model.eval()
    vanilla_model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0
    total_variance = 0
    total_digits = 0
    correct_digits = 0
    mispredictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)

            # Get embeddings
            regular_embeddings = get_regular_embeddings(model, input_ids)
            vanilla_embeddings = vanilla_model(scatter_tensor)
            input_embeddings = regular_embeddings + vanilla_embeddings

            # Forward pass
            outputs = model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1]
            
            # Get last token hidden state
            last_token_hidden_state = (last_hidden_state * last_token_mask.unsqueeze(-1)).sum(dim=1)

            # Compute predictions
            predicted_numbers = vanilla_model.compute_prediction(last_token_hidden_state)
            
            # Calculate metrics
            tolerance = 10 ** (-vanilla_model.frac_digit_len)
            correct = torch.abs(predicted_numbers - labels) < tolerance
            total_correct += correct.sum().item()
            total_samples += labels.size(0)
            
            # Digit-wise accuracy
            scaled_labels = (labels * (10 ** vanilla_model.frac_digit_len)).long()
            scaled_preds = (predicted_numbers * (10 ** vanilla_model.frac_digit_len)).long()
            
            for i in range(labels.size(0)):
                label_digits = []
                pred_digits = []
                
                # Extract digits from label
                num = scaled_labels[i]
                for p in vanilla_model.powers_of_ten:
                    label_digits.append((num // p) % 10)
                
                # Extract digits from prediction
                num = scaled_preds[i]
                for p in vanilla_model.powers_of_ten:
                    pred_digits.append((num // p) % 10)
                
                # Compare digits
                for l, p in zip(label_digits, pred_digits):
                    if l == p:
                        correct_digits += 1
                    total_digits += 1

            # Store mispredictions
            for i in range(labels.size(0)):
                if not correct[i]:
                    mispredictions.append((
                        predicted_numbers[i].item(),
                        labels[i].item()
                    ))

            # Calculate loss and errors
            loss = vanilla_model.compute_loss(last_token_hidden_state, labels)
            total_loss += loss.item()
            total_squared_error += torch.sum((predicted_numbers - labels) ** 2).item()

    # Calculate final metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples
    digit_accuracy = correct_digits / total_digits
    mse = total_squared_error / total_samples
    mean_label = torch.cat([batch['labels'].to(device) for batch in test_loader]).mean()
    total_variance = torch.sum((torch.cat([batch['labels'].to(device) for batch in test_loader]) - mean_label) ** 2).item()
    r2 = 1 - (total_squared_error / total_variance) if total_variance != 0 else 0

    # Log mispredictions
    if print_labels and mispredictions:
        logging.info(f"Mispredictions (first {max_print}):")
        for pred, true in mispredictions[:max_print]:
            logging.info(f"Predicted: {pred:.5f}, True: {true:.5f}")

    return  avg_loss, (accuracy, digit_accuracy), mse, r2

    # return {
    #     'loss': avg_loss,
    #     'accuracy': accuracy,
    #     'digit_accuracy': digit_accuracy,
    #     'mse': mse,
    #     'r2': r2,
    #     'mispredictions': mispredictions[:max_print]
    # }