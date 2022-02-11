#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 12:00:00
#SBATCH -o BS/long_tail/work/duka/act_rec-%j-1.out
#SBATCH -e BS/long_tail/work/duka/act_rec-err-%j-1.out
#SBATCH --gres gpu:4
#SBATCH -c 12

echo "using GPU ${CUDA_VISIBLE_DEVICES}"
# Make conda available
eval "$(conda shell.bash hook)"
# Activate my conda environment
conda activate /BS/feat_augm/work/amaconda3/envs/ua
# Create the logging dir if it does not exist
mkdir -p /BS/long_tail/work/duka/logging

# execute the python script
cd /BS/unintentional_actions/work/unintentional_actions/action_classification
python main.py /BS/unintentional_actions/work/unintentional_actions/configs/ablations/transform_type_ablations/stage3/transform_type_ablation_random_speedup_stage3_f2c.yml