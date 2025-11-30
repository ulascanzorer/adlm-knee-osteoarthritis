#!/bin/bash

#SBATCH --job-name=knee_ae_train
#SBATCH --output=train_ae%j.out
#SBATCH --error=train_ae%j.err
#SBATCH --time=1-00:00:00              # 1 day, adjust as needed
#SBATCH --gres=gpu:1                   # 1 GPU
#SBATCH --cpus-per-task=8              # dataloader workers, etc.
#SBATCH --mem=32G                      # increase for 224x224xD volumes
#SBATCH --nice=100                     # optional priority smoothing

# Load python uv module
ml python/uv

uv sync

# Activate the project environment (.venv created by uv)
source .venv/bin/activate

echo "Python interpreter:"
which python

echo "Python version:"
python --version

echo "GPU status:"
nvidia-smi
echo


# Make sure WANDB won’t ask for login on the cluster
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb_runs"


echo "W&B Mode: $WANDB_MODE"

# ----------------------------
# RUN TRAINING
# ----------------------------

echo "STARTING TRAINING JOB"
echo "Running train.py..."
python train.py

echo "Training DONE"