import os
import numpy as np
import pandas as pd

# Function to generate unique addition pairs
def generate_unique_addition_pairs(max_value, num_samples):
    question_label_pairs = set()
    target_samples = int(num_samples * 1.1)  # 10% extra to ensure enough unique samples
    
    while len(question_label_pairs) < num_samples:
        num1 = np.random.randint(0, max_value + 1)
        num2 = np.random.randint(0, max_value + 1)
        
        # Sort to ensure (num1, num2) is unique regardless of order
        num1, num2 = sorted((num1, num2))
        result = num1 + num2
        result = result / 1000
        
        question_label_pairs.add((f"{num1/1000}+{num2/1000}=", result))
    
    # Convert to list and trim down to exactly num_samples
    question_label_pairs = list(question_label_pairs)[:num_samples]
    return question_label_pairs

# Parameters
max_value = 1000000
num_samples = 1000000
split_ratio = 0.8

# Generate dataset
addition_data = generate_unique_addition_pairs(max_value, num_samples)
df = pd.DataFrame(addition_data, columns=['question', 'label'])

# Split the dataset
train_size = int(len(df) * split_ratio * 0.9)
validation_size = int(len(df) * split_ratio * 0.1)
train_df = df[:train_size]
validation_df = df[train_size:train_size + validation_size]
test_df = df[train_size + validation_size:]

# Save to CSV
dir_path = "dataset/decimal_addition"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

train_df.to_csv(f"{dir_path}/train_dataset.csv", index=False)
validation_df.to_csv(f"{dir_path}/validation_dataset.csv", index=False)
test_df.to_csv(f"{dir_path}/test_dataset.csv", index=False)

print(f"Addition datasets saved to {dir_path}")
