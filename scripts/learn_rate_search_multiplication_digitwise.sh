#!/bin/bash

source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv

# Dataset and model details
dataset="{yourid}/int_multiplication"
model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=512
max_train_samples=360000
learning_rates=(5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6)

# # Methods to iterate
methods=("regular" "fne" "xval")

# Loop for learning rate search with --train_from_scratch
for method in "${methods[@]}"
do
    echo "Searching best learning rate for method $method with --train_from_scratch and full dataset size $max_train_samples"

    # Iterate over learning rates
    for lr in "${learning_rates[@]}"
    do
        echo "Running model $model on dataset $dataset with method $method, learning_rate=$lr (with train_from_scratch)"

        sbatch --gres=gpu:a6000:1 --nodelist {yournode} --time 3-0 <<EOT
#!/bin/bash
source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv
cd ..
python main.py --lr $lr --name learningrate_search --model_size_level 4 --epochs 100 --clip --batch_size $batch_size --model $model --int_digit_len 10 --frac_digit_len 0 --dataset $dataset --train_from_scratch --method $method --num_train_samples $max_train_samples  --len_gen_size 2
EOT
    done
done

