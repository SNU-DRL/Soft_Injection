#!/bin/bash

echo "Running ours method with model: meta-llama/Llama-3.1-8b"

echo "Running code block 0"
CUDA_VISIBLE_DEVICES=0 python eval.py --exact_match --model_name meta-llama/Llama-3.1-8b --data_name icl --num_example 50 --num_shot 10 --max_token 20 --eval_num_shot 0 --activation_path ./site/Llama-3.1-8b/mean_activations --alphas_path ./site/Llama-3.1-8b/alphas/alphas_interv_ag_news_epoch_400_lr_0.2.pt --result_folder ./site/Llama-3.1-8b/result --cur_mode interv --experiment_name interv_ag_news_epoch_400_lr_0.2 --batch_size 1 --test_batch_size 1 --data_path ./dataset --dataset_names ag_news --lr 0.2 --epoch 400&

wait
echo 'Successfully ran the code'
