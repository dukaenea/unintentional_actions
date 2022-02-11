#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 08:00:00
#SBATCH -o /BS/unintentional_actions/nobackup/slurm_logs/act_loc-%j-1.out
#SBATCH -e /BS/unintentional_actions/nobackup/slurm_logs/act_loc-err-%j-1.out
#SBATCH --gres gpu:4
#SBATCH -c 12

echo "using GPU ${CUDA_VISIBLE_DEVICES}"
# Make conda available
eval "$(conda shell.bash hook)"
# Activate my conda environment
conda activate ua

# execute the python script
cd /BS/unintentional_actions/work/unintentional_actions/action_localization
python main_al.py