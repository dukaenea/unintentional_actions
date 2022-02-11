#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 12:00:00
#SBATCH -o /BS/unintentional_actions/nobackup/slurm_logs/dsv-%j-1.out
#SBATCH -e /BS/unintentional_actions/nobackup/slurm_logs/dsv-err-%j-1.out
#SBATCH --gres gpu:4
#SBATCH -c 12

echo "using GPU ${CUDA_VISIBLE_DEVICES}"
# Make conda available
eval "$(conda shell.bash hook)"
# Activate my conda environment
conda activate /BS/feat_augm/work/amaconda3/envs/ua
# Create the logging dir if it does not exist
#mkdir -p /BS/long_tail/work/duka/logging
#mkdir -p /BS/long_tail/work/duka/oops/train
#mkdir -p /BS/long_tail/work/duka/oops/val

# execute the python script
cd /BS/unintentional_actions/work/unintentional_actions/utils
python downsize_video.py