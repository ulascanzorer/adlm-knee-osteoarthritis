#!/bin/bash

#SBATCH --job-name=inference_clustering_and_visualization
#SBATCH --output=slurm_job.out
#SBATCH --error=slurm_job.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=phd
#SBATCH --time=0-0:05:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Load python uv module
ml python/uv

# Run the program
nvidia-smi
sleep 10

uv run main.py \
  --model_name autoencoder \
  --weights_path 'weights_ae/50_ae.pth' \
  --side left

# Final message
python -c "print('The job is finished!')"
