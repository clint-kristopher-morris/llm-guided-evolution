#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 8-00:00
#SBATCH --gres=gpu:3
#SBATCH -C "NVIDIAA100-SXM4-80GB|NVIDIAA10080GBPCIe|TeslaV100-PCIE-32GB|GeForceGTX1080Ti|GeForceGTX1080"
#SBATCH --mem 10G
#SBATCH -c 48
echo "Launching AIsurBL"
hostname

# Load GCC version 9.2.0
module load gcc/13.2.0
# module load cuda/11.8

# Activate Conda environment
source /opt/apps/Module/anaconda3/2021.11/bin/activate mix
conda info
# NVIDIAA100-SXM4-80GB|NVIDIAA10080GBPCIe|TeslaV100-PCIE-32GB
# TeslaV100S-PCIE-32GB|TeslaV100S-PCIE-32GB|GeForceGTX1080Ti|GeForceGTX1080
# Set the TOKENIZERS_PARALLELISM environment variable if needed
export TOKENIZERS_PARALLELISM=false

# Run Python script for ExquisiteNetV2
# python llm_crossover.py '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/ExquisiteNetV2/network.py' '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/ExquisiteNetV2/models/network_x.py' '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/ExquisiteNetV2/models/network_z.py' --top_p 0.15 --temperature 0.1 --apply_quality_control 'True' --bit 8

# Run Python script for PointNet++
python llm_crossover.py '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/Pointnet_Pointnet2_pytorch/models/pointnet2_cls_ssg.py' '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/Pointnet_Pointnet2_pytorch/models/llmge-models/pointnet2_cls_ssg_x.py' '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/Pointnet_Pointnet2_pytroch/models/llmge-models/pointnet2_cls_ssg_z.py' python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg