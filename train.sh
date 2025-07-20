#!/bin/bash

# 创建日志目录
mkdir -p logs

# 使用nohup运行训练脚本，并将输出重定向到日志文件
nohup NGPU=2 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10846 train.py \
	--train_data processed_data/train.pkl \
	--eval_data processed_data/test.pkl \
	--use_checkpoint \
	--lr 3e-5 \
	--model_size large \
	--num_workers 8 \
	--optim adamw \
	--box_number 36 \
	--scheduler linear \
	--weight_decay 0.01 \
	--save_freq 5000 \
	--eval_freq 5000 \
	--print_freq 100 \
	--text_maxlength 400 \
	--seed 833 \
	--name exp_high_quality \
	--checkpoint_dir ./checkpoints_high_quality \
	--per_gpu_batch_size 1 \
	--n_block 9 \
	--n_tags 30 \
	--n_im_context 5 \
	--n_ex_context 40 \
	--total_step 30000 \
	--warmup_step 2000 > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 输出进程ID
echo "Training started with PID: $!"

