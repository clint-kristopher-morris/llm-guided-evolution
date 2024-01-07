#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 8-00:00              		# Runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mem 8G
#SBATCH -c 1                          # number of CPU cores
# echo "launching AIsurBL"
# hostname
# # module load anaconda3/2020.07 2021.11
# module load cuda/11.0
# export CUDA_VISIBLE_DEVICES=0

# source /opt/apps/Module/anaconda3/2021.11/bin/activate
# conda activate AIsurBL
# conda info

python run.py