#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 2-00:00:00
#SBATCH -o /BS/unintentional_actions/nobackup/slurm_logs/end2end_rep_lrn-%j-1.out
#SBATCH -e /BS/unintentional_actions/nobackup/slurm_logs/end2end_rep_lrn-err-%j-1.out
#SBATCH --gres gpu:4

echo "using GPU ${CUDA_VISIBLE_DEVICES}"
# Make conda available
eval "$(conda shell.bash hook)"
# Activate my conda environment
conda activate ua

# execute the python script
cd /BS/unintentional_actions/work/unintentional_actions/rep_learning
python end2end.py