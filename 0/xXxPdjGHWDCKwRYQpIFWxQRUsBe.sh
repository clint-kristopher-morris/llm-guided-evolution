#!/bin/bash
#SBATCH --job-name=llm_oper
#SBATCH -t 8-00:00
#SBATCH --gres=gpu:1
#SBATCH -G 1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem 16G
#SBATCH -c 12
echo "Launching AIsurBL"
hostname
# Load GCC version 9.2.0
# module load gcc/13.2.0
# module load cuda/11.8
module load cuda/12
module load cuda
module load anaconda3
# Activate Conda environment
conda activate llmge-env
# conda info
# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Run Python script
python src/llm_mutation.py /storage/ice1/2/6/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/ExquisiteNetV2/network.py /storage/ice1/2/6/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/ExquisiteNetV2/models/network_xXxPdjGHWDCKwRYQpIFWxQRUsBe.py 0/xXxPdjGHWDCKwRYQpIFWxQRUsBe_model.txt --top_p 0.1 --temperature 0.28 --apply_quality_control 'False' --hugging_face True
