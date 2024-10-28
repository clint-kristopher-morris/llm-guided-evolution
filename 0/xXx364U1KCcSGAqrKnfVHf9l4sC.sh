#!/bin/bash
#SBATCH --job-name=llm_oper
#SBATCH -t 8-00:00
#SBATCH --gres=gpu:1
#SBATCH -G 1
#SBATCH -C "NVIDIAA100-SXM4-80GB|NVIDIAA10080GBPCIe|TeslaV100-PCIE-32GB|TeslaV100S-PCIE-32GB|NVIDIARTX6000AdaGeneration|NVIDIARTXA6000|NVIDIARTXA5000|NVIDIARTXA4000|GeForceGTX1080Ti|QuadroRTX4000|QuadroP4000|GeForceGTX1080|TeslaP4"
#SBATCH --mem 16G
#SBATCH -c 12
echo "Launching AIsurBL"
hostname

# Load GCC version 9.2.0
# module load gcc/13.2.0
# module load cuda/11.8
module load cuda/12
# Activate Conda environment
source /opt/apps/Module/anaconda3/2021.11/bin/activate llm_guided_evolution
# conda info

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Run Python script
python src/llm_mutation.py /storage/ice1/2/6/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/ExquisiteNetV2/network.py /storage/ice1/2/6/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/ExquisiteNetV2/models/network_xXx364U1KCcSGAqrKnfVHf9l4sC.py 0/xXx364U1KCcSGAqrKnfVHf9l4sC_model.txt --top_p 0.1 --temperature 0.22 --apply_quality_control 'False' --hugging_face True
