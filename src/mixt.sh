#________________________________________________________________________
# Changed indivudal configrations like path for main model in Point Transformers #1
#________________________________________________________________________

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

#________________________________________________________________________
# Added Run Scripts Commands to run PointNet++ using PyTorch it reflects my Path update yours as well
# Added Run Scripts COmmands to run Point Transformers using 
#________________________________________________________________________

# Run Python script - ExquisiteNetV2
python llm_crossover.py '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/ExquisiteNetV2/network.py' '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/ExquisiteNetV2/models/network_x.py' '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/ExquisiteNetV2/models/network_z.py' --top_p 0.15 --temperature 0.1 --apply_quality_control 'True' --bit 8

# Run Python Script - PointNet++ PyTorch #1


# Run python Script - Point Transformers #1
#python llm_crossover.py '/gv1/projects/AI_Surrogate/dev/dev/clint/CodeLLama/codellama/sota/Point-Transformers/models/Menghao/model.py' '/home/hice1/madewolu9/scratch/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/Point-Transformers/models/llmge_models/model_x.py' '/home/hice1/madewolu9/scratch/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/Point-Transformers/models/llmge_models/model_z.py'  --top_p 0.15 --temperature 0.1 --apply_quality_control 'True' --bit 8
