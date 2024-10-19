#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8-00:00              		# Runtime in D-HH:MM
#SBATCH --mem 16G
#SBATCH -c 1                          # number of CPU cores
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH -C "NVIDIAA100-SXM4-80GB|NVIDIAA10080GBPCIe|TeslaV100-PCIE-32GB|TeslaV100S-PCIE-32GB|NVIDIARTX6000AdaGeneration|NVIDIARTXA6000|NVIDIARTXA5000|NVIDIARTXA4000|GeForceGTX1080Ti|QuadroRTX4000|QuadroP4000|GeForceGTX1080|TeslaP4"
echo "launching AIsurBL"
hostname
# module load anaconda3/2020.07 2021.11
module load cuda/12
export CUDA_VISIBLE_DEVICES=0

source /opt/apps/Module/anaconda3/2021.11/bin/activate
conda activate llm_guided_evolution
conda info

python run_improved.py first_test
