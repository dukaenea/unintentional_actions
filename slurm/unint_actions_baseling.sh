#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 1-23:00:00
#SBATCH -o /BS/unintentional_actions/nobackup/slurm_logs/unint_act-%j-1.out
#SBATCH -e /BS/unintentional_actions/nobackup/slurm_logs/unint_act-err-%j-1.out
#SBATCH --gres gpu:4
#SBATCH -c 12

echo "using GPU ${CUDA_VISIBLE_DEVICES}"
# Make conda available
eval "$(conda shell.bash hook)"
# Activate my conda environment
conda activate feat_augm_env

# execute the python script
cd /BS/unintentional_actions/work/unintentional_actions/baselines
python unint_actions.py