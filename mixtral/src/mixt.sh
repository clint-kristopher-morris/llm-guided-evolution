#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 8-00:00
#SBATCH --gres=gpu:2
#SBATCH -C "TeslaV100-PCIE-32GB"
#SBATCH --mem 32G
#SBATCH -c 1
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
python llm_crossover.py '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/ExquisiteNetV2/network.py' '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/ExquisiteNetV2/models/network_x.py' '/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/sota/ExquisiteNetV2/models/network_z.py' --top_p 0.15 --temperature 0.1
