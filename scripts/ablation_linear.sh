#!/bin/bash

source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv

# Dataset and model details
datasets=(
    "{yourid}/int_subtract"
    "{yourid}/decimal_addition"
    "{yourid}/int_addition2"
    "{yourid}/int_multiplication"
)
model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=512
lr=0.005  # Learning rate for FNE method

# Iterate over datasets
for dataset in "${datasets[@]}"
do
    # Set digit lengths based on dataset
    if [[ $dataset == "{yourid}/decimal_addition" ]]; then
        int_digit_len=7
        frac_digit_len=3
        num_train_samples=720000
    elif [[ $dataset == "{yourid}/int_multiplication" ]]; then
        int_digit_len=10
        frac_digit_len=0
        num_train_samples=360000
    else
        int_digit_len=10
        frac_digit_len=0
        num_train_samples=720000
    fi

    # Outer loop for add_linear and not_add_linear
    for use_not_add_linear in True False
    do
        if [[ $use_not_add_linear == "True" ]]; then
            linear_flag="--not_add_linear"
        else
            linear_flag=""
        fi

        echo "Running model $model on dataset $dataset with ${linear_flag}"
        sbatch --gres=gpu:a6000:1 --nodelist {yournode} --time 3-0 <<EOT
#!/bin/bash
source ~/.bashrc
conda activate /home/{yourname}/anaconda3/envs/math/myenv
cd ..
python main.py --lr $lr --name ablationlinear_datasize --period_base_list 10 --model_size_level 4 --epochs 100 --clip --batch_size $batch_size --model $model --int_digit_len $int_digit_len --frac_digit_len $frac_digit_len --dataset $dataset --train_from_scratch --method fne $linear_flag --num_train_samples $num_train_samples
EOT
    done
done
