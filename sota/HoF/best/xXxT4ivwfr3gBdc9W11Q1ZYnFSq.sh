#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 8-00:00
#SBATCH --gres=gpu:1
#SBATCH -C "NVIDIAA100-SXM4-80GB|NVIDIAA10080GBPCIe|TeslaV100-PCIE-32GB|QuadroRTX4000|GeForceGTX1080Ti|GeForceGTX1080|TeslaV100-PCIE-32GB|TeslaV100S-PCIE-32GB"
#SBATCH --mem 4G
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
python src/llm_mutation.py /gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/GuidedEvolution/sota/ExquisiteNetV2/models/network_xXxwXpmEcxlQJwniUEUHRt0kKls.py /gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/GuidedEvolution/sota/ExquisiteNetV2/models/network_xXxT4ivwfr3gBdc9W11Q1ZYnFSq.py 0/xXxT4ivwfr3gBdc9W11Q1ZYnFSq_model.txt --top_p 0.1 --temperature 0.33 --apply_quality_control 'False' --hugging_face True
