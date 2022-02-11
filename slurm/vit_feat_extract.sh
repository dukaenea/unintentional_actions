#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 1-20:00:00
#SBATCH -o /BS/unintentional_actions/nobackup/slurm_logs/vit_ext-%j-1.out
#SBATCH -e /BS/unintentional_actions/nobackup/slurm_logs/vit_ext-err-%j-1.out
#SBATCH --gres gpu:4
#SBATCH -c 8
#SBATCH --mem-per-cpu=100G

echo "using GPU ${CUDA_VISIBLE_DEVICES}"
# Make conda available
eval "$(conda shell.bash hook)"
# Activate my conda environment
conda activate ua
nvidia-smi
# execute the python script
cd /BS/unintentional_actions/work/unintentional_actions/feat_exteact
python vit_feat_extract.py