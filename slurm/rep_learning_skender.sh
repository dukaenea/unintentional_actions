#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 08:00:00
#SBATCH -o /BS/feature_bank/work/enea/rep_lrn-%j-1.out
#SBATCH -e /BS/feature_bank/work/enea/rep_lrn-err-%j-1.out
#SBATCH --gres gpu:4
#SBATCH -c 16


echo "using GPU ${CUDA_VISIBLE_DEVICES}"
# Make conda available
eval "$(conda shell.bash hook)"
# Activate my conda environment
conda activate /BS/feat_augm/work/amaconda3/envs/ua
# Create the logging dir if it does not exist
mkdir -p /BS/feature_bank/work/enea/logging

# execute the python script
cd /BS/unintentional_actions/work/unintentional_actions/rep_learning
python main.py /BS/unintentional_actions/work/unintentional_actions/configs/ablations/transform_type_ablations/transform_type_ablation_double_flip.yml