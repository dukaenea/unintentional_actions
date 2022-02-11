#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 08:00:00
#SBATCH -o /BS/unintentional_actions/nobackup/slurm_logs/new_loc-%j-1.out
#SBATCH -e /BS/unintentional_actions/nobackup/slurm_logs/new_loc-err-%j-1.out
#SBATCH --gres gpu:4
#SBATCH -c 12

echo "using GPU ${CUDA_VISIBLE_DEVICES}"
# Make conda available
eval "$(conda shell.bash hook)"
# Activate my conda environment
conda activate ua

# execute the python script
cd /BS/unintentional_actions/work/unintentional_actions/new_action_localisation
python main.py /BS/unintentional_actions/work/unintentional_actions/configs/best_action_classification.yml