#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 8-00:00
#SBATCH --gres=gpu:1
#SBATCH -C "NVIDIAA100-SXM4-80GB|NVIDIAA10080GBPCIe|GeForceGTX1080Ti|TeslaV100-PCIE-32GB|TeslaV100S-PCIE-32GB"
#SBATCH --mem 8G
#SBATCH -c 32
echo "Launching AIsurBL"
hostname

# Load GCC version 9.2.0
module load gcc/13.2.0
# module load cuda/11.8

# Activate Conda environment
source /opt/apps/Module/anaconda3/2021.11/bin/activate mix
conda info

# Set the TOKENIZERS_PARALLELISM environment variable if needed
export TOKENIZERS_PARALLELISM=false

# Run Python script
python ./SotaTest/train.py -bs 216 -network "network_xXxT4ivwfr3gBdc9W11Q1ZYnFSq" -data "./../../ExquisiteNetV2/cifar10" -end_lr 0.001 -seed 21 -val_r 0.2 -amp
