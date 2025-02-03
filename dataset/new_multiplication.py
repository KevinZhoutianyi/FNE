import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_unique_multiplication_pairs_by_product_buckets(
    max_value: int,
    num_samples: int,
    num_splits: int = 10
):
    """
    Generate unique multiplication pairs by splitting the product range [0, max_value^2]
    into `num_splits` buckets. We then gather (num_samples // num_splits) samples per
    bucket such that the product of the two operands falls in that bucket.
    """
    # Number of pairs per bucket
    sub_samples = num_samples // num_splits

    # We'll store (question_str, label) in a set to avoid duplicates
    question_label_pairs = set()

    # The maximum product is max_value * max_value
    max_product = max_value * max_value

    # Precompute bucket edges
    # e.g. if max_product = 100^2 = 10000, dividing into 10 splits means each is size=1000
    bucket_size = (max_product + 1) // num_splits  # +1 so we include max_product in range

    with tqdm(total=num_samples, desc="Generating multiplication pairs") as pbar:
        # Loop over each bucket
        for i in range(num_splits):
            # Calculate product range for this bucket
            bucket_min = i * bucket_size
            # For the last bucket, include everything up to max_product
            if i < num_splits - 1:
                bucket_max = (i+1) * bucket_size - 1
            else:
                bucket_max = max_product

            samples_collected_this_bucket = 0

            # Keep generating random pairs until we fill up this bucket or exceed total
            while samples_collected_this_bucket < sub_samples:
                # If total samples already reached, break
                if len(question_label_pairs) >= num_samples:
                    break

                # Generate two random numbers in [0, max_value]
                num1 = np.random.randint(0, max_value + 1)
                num2 = np.random.randint(0, max_value + 1)
                product = num1 * num2

                # Check if product is in this bucket range
                if bucket_min <= product <= bucket_max:
                    # Order the pair for uniqueness
                    x, y = min(num1, num2), max(num1, num2)
                    # Create a question-label pair
                    pair = (f"{x}*{y}=", product)

                    # Only add if unique
                    if pair not in question_label_pairs:
                        question_label_pairs.add(pair)
                        samples_collected_this_bucket += 1
                        pbar.update(1)

                # Also stop if total samples has reached or exceeded
                if len(question_label_pairs) >= num_samples:
                    break

            # Stop outer loop if we've already generated everything
            if len(question_label_pairs) >= num_samples:
                break

    # Convert set to list, truncate if there's any overshoot (edge cases)
    question_label_pairs = list(question_label_pairs)[:num_samples]
    return question_label_pairs


# ------------------
# Example usage below
# ------------------

if __name__ == "__main__":
    # Parameters
    max_value = 10000        # Max operand
    num_samples = 1000000    # Total # of samples
    split_ratio = 0.8        # Train/test split ratio
    num_splits = 10          # Split product range [0..max_value^2] into 10 segments

    # 1) Generate unique multiplication pairs
    multiplication_data = generate_unique_multiplication_pairs_by_product_buckets(
        max_value=max_value,
        num_samples=num_samples,
        num_splits=num_splits
    )

    # 2) Convert to DataFrame
    df = pd.DataFrame(multiplication_data, columns=["question", "label"])

    # 3) Split the dataset into train, validation, and test
    train_size = int(len(df) * split_ratio * 0.9)
    validation_size = int(len(df) * split_ratio * 0.1)

    train_df_mul = df[:train_size]
    validation_df_mul = df[train_size : train_size + validation_size]
    test_df_mul = df[train_size + validation_size :]

    # 4) Define save paths
    dir_name = "dataset/newmultiplication"
    train_csv_path = os.path.join(dir_name, "train_dataset.csv")
    validation_csv_path = os.path.join(dir_name, "validation_dataset.csv")
    test_csv_path = os.path.join(dir_name, "test_dataset.csv")

    # 5) Create directory if not exists
    os.makedirs(dir_name, exist_ok=True)

    # 6) Save to CSV
    train_df_mul.to_csv(train_csv_path, index=False)
    validation_df_mul.to_csv(validation_csv_path, index=False)
    test_df_mul.to_csv(test_csv_path, index=False)

    print(f"Multiplication datasets saved to {dir_name}")
