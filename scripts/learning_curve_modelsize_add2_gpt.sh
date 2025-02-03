#!/bin/bash

source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv

# Dataset and model details
dataset="{yourid}/int_addition2"
model="openai-community/gpt2-large"
batch_size=512
size_levels=(1 2 3 4 5 6)  # Define the size levels to iterate overnum
# Methods to iterate
num_train_samples=200000
methods=('regular' 'xval' 'fne') # 

# Outer loop for methods
for method in "${methods[@]}"
do
    # Assign learning rate based on method
    if [[ $method == "regular" ]]; then
        lr=0.005
    elif [[ $method == "fne" ]]; then
        lr=0.005
    elif [[ $method == "xval" ]]; then
        lr=0.0001
    fi

    # Inner loop for size levels
    for size_level in "${size_levels[@]}"
    do
        if [[ $method == "regular" ]]; then
            # Run with use_digit_wise_tokenizer
            echo "Running model $model on dataset $dataset with method $method, size_level=$size_level, lr=$lr (with tokenizer)"
            sbatch --gres=gpu:a6000:1 --nodelist {yournode} --time 3-0 <<EOT
#!/bin/bash
source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv
cd ..
python main.py --lr $lr --epochs 100 --use_digit_wise_tokenizer --period_base_list 10 --name learningcurve_modelsize --clip --batch_size $batch_size --model $model --int_digit_len 10 --frac_digit_len 0 --dataset $dataset --train_from_scratch --method $method --model_size_level $size_level --num_train_samples $num_train_samples
EOT

            # Run without use_digit_wise_tokenizer
            echo "Running model $model on dataset $dataset with method $method, size_level=$size_level, lr=$lr (without tokenizer)"
            sbatch --gres=gpu:a6000:1 --nodelist {yournode} --time 3-0 <<EOT
#!/bin/bash
source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv
cd ..
python main.py --lr $lr --epochs 100 --period_base_list 10 --name learningcurve_modelsize --clip --batch_size $batch_size --model $model --int_digit_len 10 --frac_digit_len 0 --dataset $dataset --train_from_scratch --method $method --model_size_level $size_level --num_train_samples $num_train_samples
EOT
        else
            # Run for other methods
            echo "Running model $model on dataset $dataset with method $method, size_level=$size_level, lr=$lr"
            sbatch --gres=gpu:a6000:1 --nodelist {yournode} --time 3-0 <<EOT
#!/bin/bash
source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv
cd ..
python main.py --lr $lr --epochs 100  --period_base_list 10 --name learningcurve_modelsize --clip --batch_size $batch_size --model $model --int_digit_len 10 --frac_digit_len 0 --dataset $dataset --train_from_scratch --method $method --model_size_level $size_level --num_train_samples $num_train_samples
EOT
        fi
    done
done


method="regular"
lr=0.00005
echo "Running model $model on dataset $dataset with method $method (no train_from_scratch)"
sbatch --gres=gpu:a6000:1 --nodelist {yournode} --time 3-0 <<EOT
#!/bin/bash
source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv
cd ..
python main.py --lr $lr --name learningcurve_modelsize --epochs 30 --clip --batch_size 200 --model $model --dataset $dataset --method $method --num_train_samples $num_train_samples
EOT