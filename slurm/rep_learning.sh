#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 1-00:00:00
#SBATCH -o /BS/unintentional_actions/nobackup/slurm_logs/rep_lrn-%j-1.out
#SBATCH -e /BS/unintentional_actions/nobackup/slurm_logs/rep_lrn-err-%j-1.out
#SBATCH --gres gpu:4
#SBATCH -c 16
#SBATCH --mem-per-cpu=16G

echo "using GPU ${CUDA_VISIBLE_DEVICES}"
# Make conda available
eval "$(conda shell.bash hook)"
# Activate my conda environment
conda activate ua

# execute the python script
cd /BS/unintentional_actions/work/unintentional_actions/rep_learning
python main.py /BS/unintentional_actions/work/unintentional_actions/configs/ablations/transform_type_ablations/transform_type_ablation_shuffle_stage2.yml