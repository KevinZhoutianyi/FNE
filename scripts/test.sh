#!/bin/bash

source ~/.bashrc
conda activate fne
cd ..

python main.py --lr 0.005 --epochs 5 --period_base_list 10  --name test \
--clip --batch_size 512 --model "meta-llama/Llama-3.2-1B-Instruct" --int_digit_len 7 --frac_digit_len 3 --dataset "Onlydrinkwater/decimal_addition" \
--train_from_scratch --method fne --model_size_level 4 --num_train_samples 10000  