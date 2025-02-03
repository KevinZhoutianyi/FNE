#!/bin/bash

source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv

# Dataset and model details
dataset="{yourid}/int_addition_three"
model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=512
max_train_samples=720000

# Methods to iterate
methods=('regular' 'fne' 'xval') #  

num_train_samples_list=(0 100)
while [[ ${num_train_samples_list[-1]} -lt $max_train_samples ]]; do
    num_train_samples_list+=($((${num_train_samples_list[-1]} * 2)))
done

# Ensure the last run is set to max_train_samples
if [[ ${num_train_samples_list[-1]} -ne $max_train_samples ]]; then
    num_train_samples_list+=($max_train_samples)
fi

for method in "${methods[@]}"
do
    # Set learning rate based on method
    if [[ $method == "regular" ]]; then
        lr=0.005
    elif [[ $method == "fne" ]]; then
        lr=0.005
    elif [[ $method == "xval" ]]; then
        lr=0.0001
    fi

    # Inner loop for training samples
    for num_train_samples in "${num_train_samples_list[@]}"
    do
        if [[ $num_train_samples -gt $max_train_samples ]]; then
            continue
        fi

        if [[ $method == "regular" ]]; then
            # Run with use_digit_wise_tokenizer
            echo "Running model $model on dataset $dataset with method $method, num_train_samples=$num_train_samples (with tokenizer)"
            sbatch --gres=gpu:a6000:1 --nodelist {yournode} --time 3-0 <<EOT
#!/bin/bash
source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv
cd ..
python main.py --lr $lr --name learningcurve_datasize --use_digit_wise_tokenizer --period_base_list 10 --model_size_level 4 --epochs 100 --clip --batch_size $batch_size --model $model --int_digit_len 10 --frac_digit_len 0 --dataset $dataset --train_from_scratch --method $method --num_train_samples $num_train_samples
EOT

            # Run without use_digit_wise_tokenizer
            echo "Running model $model on dataset $dataset with method $method, num_train_samples=$num_train_samples (without tokenizer)"
            sbatch --gres=gpu:a6000:1 --nodelist {yournode} --time 3-0 <<EOT
#!/bin/bash
source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv
cd ..
python main.py --lr $lr --name learningcurve_datasize --period_base_list 10 --model_size_level 4 --epochs 100 --clip --batch_size $batch_size --model $model --int_digit_len 10 --frac_digit_len 0 --dataset $dataset --train_from_scratch --method $method --num_train_samples $num_train_samples
EOT
        else
            # For other methods
            echo "Running model $model on dataset $dataset with method $method, num_train_samples=$num_train_samples"
            sbatch --gres=gpu:a6000:1 --nodelist {yournode} --time 3-0 <<EOT
#!/bin/bash
source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv
cd ..
python main.py --lr $lr --name learningcurve_datasize --period_base_list 10 --model_size_level 4 --epochs 100 --clip --batch_size $batch_size --model $model --int_digit_len 10 --frac_digit_len 0 --dataset $dataset --train_from_scratch --method $method --num_train_samples $num_train_samples
EOT
        fi
    done
done

# Additional loop for regular method without --train_from_scratch
method="regular"
lr=0.00005
for num_train_samples in "${num_train_samples_list[@]}"
do
    if [[ $num_train_samples -gt $max_train_samples ]]; then
        continue
    fi

    echo "Running model $model on dataset $dataset with method $method (no train_from_scratch), num_train_samples=$num_train_samples"

    sbatch --gres=gpu:a6000:1 --nodelist {yournode} --time 3-0 <<EOT
#!/bin/bash
source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv
cd ..
python main.py --lr $lr --name learningcurve_datasize --epochs 30 --clip --batch_size 200 --model $model --dataset $dataset --method $method --num_train_samples $num_train_samples
EOT
done