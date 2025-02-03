import os
import numpy as np
import pandas as pd

# Function to generate unique classification pairs
def generate_unique_classification_pairs(max_value, num_samples, a, b, c, d):
    question_label_pairs = set()
    target_samples = int(num_samples * 1.1)  # 10% extra to ensure enough unique samples
    
    while len(question_label_pairs) < num_samples:
        num1 = np.random.randint(0, max_value + 1)
        num2 = np.random.randint(0, max_value + 1)
        num3 = np.random.randint(0, max_value + 1)
        
        # Sort to ensure (num1, num2, num3) is unique regardless of order
        nums = sorted((num1, num2, num3))
        result = a * nums[0] + b * nums[1] + c * nums[2] - d
        label = 1 if result > 0 else 0  # Binary classification
        
        question_label_pairs.add((f"{nums[0]},{nums[1]},{nums[2]}", label))
    
    # Convert to list and trim down to exactly num_samples
    question_label_pairs = list(question_label_pairs)[:num_samples]
    return question_label_pairs

# Parameters
max_value = 1000
num_samples = 100000
split_ratio = 0.8
a, b, c, d = 1.5, -2, 0.5, 10  # Coefficients for the classification function

# Generate dataset
classification_data = generate_unique_classification_pairs(max_value, num_samples, a, b, c, d)
df = pd.DataFrame(classification_data, columns=['question', 'label'])

# Split the dataset
train_size = int(len(df) * split_ratio * 0.9)
validation_size = int(len(df) * split_ratio * 0.1)
train_df = df[:train_size]
validation_df = df[train_size:train_size + validation_size]
test_df = df[train_size + validation_size:]

# Save to CSV
dir_path = "dataset/classification"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

train_df.to_csv(f"{dir_path}/train_dataset.csv", index=False)
validation_df.to_csv(f"{dir_path}/validation_dataset.csv", index=False)
test_df.to_csv(f"{dir_path}/test_dataset.csv", index=False)

print(f"Classification datasets saved to {dir_path}")
