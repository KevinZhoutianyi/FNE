import os
import numpy as np
import pandas as pd

# Function to generate unique multiplication pairs
def generate_unique_multiplication_pairs(max_value, num_samples):
    question_label_pairs = set()
    
    # Generate more samples than needed to ensure the final size
    target_samples = int(num_samples)  # 10% more to ensure enough unique samples
    
    while len(question_label_pairs) < target_samples:
        num1 = np.random.randint(0, max_value + 1)
        num2 = np.random.randint(0, max_value + 1)
        
        # Order the pair to ensure uniqueness
        num1, num2 = min(num1, num2), max(num1, num2)
        
        product = num1 * num2
        question_label_pairs.add((f"{num1}*{num2}=", product))
    
    # Convert to a list and trim down to exactly num_samples
    question_label_pairs = list(question_label_pairs)[:num_samples]
    
    return question_label_pairs

# Parameters for the dataset
max_value = 1000  # Adjust as needed
num_samples = 500000  # Number of unique samples to generate
split_ratio = 0.8  # Train-test split ratio

# Generate unique multiplication pairs
multiplication_data = generate_unique_multiplication_pairs(max_value, num_samples)

# Convert to DataFrame
df = pd.DataFrame(multiplication_data, columns=['question', 'label'])

# Split the dataset into train, validation, and test sets
train_size = int(len(df) * split_ratio * 0.9)
validation_size = int(len(df) * split_ratio * 0.1)
train_df_mul = df[:train_size]
validation_df_mul = df[train_size:train_size + validation_size]
test_df_mul = df[train_size + validation_size:]

# Define paths for saving datasets
dir = f"dataset/operandmultiplication"
train_csv_path = f"{dir}/train_dataset.csv"
validation_csv_path = f"{dir}/validation_dataset.csv"
test_csv_path = f"{dir}/test_dataset.csv"

# Create the folder if it doesn't exist
if not os.path.exists(dir):
    os.makedirs(dir)

# Save dataframes to CSV
train_df_mul.to_csv(train_csv_path, index=False)
validation_df_mul.to_csv(validation_csv_path, index=False)
test_df_mul.to_csv(test_csv_path, index=False)

print(f"Multiplication datasets saved to {dir}")
