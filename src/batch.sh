#!/bin/bash

#SBATCH --time=01:00:00  # Set the maximum runtime
#SBATCH --account=csnlp  # Replace with your actual course tag
#SBATCH --output=train_output.out  # Specify the output file
#SBATCH --gpus=2  # Request two GPUs

. /etc/profile.d/modules.sh  # Load the modules environment
module load cuda/12.1  # Load CUDA module

# Activate the virtual environment
source /home/aakkus/MechInt/Transformers_Mechanistic_Interpretability-main/MechInt/bin/activate

# Set WandB API Key
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"

# Navigate to the project directory
cd /home/aakkus/MechInt/Transformers_Mechanistic_Interpretability-main

# Run the training script
python src/train_gpu.py
