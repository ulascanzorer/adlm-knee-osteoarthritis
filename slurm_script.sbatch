#!/bin/bash

#SBATCH --job-name=inference_clustering_and_visualization
#SBATCH --output=slurm_job_%j.out
#SBATCH --error=slurm_job_%j.err
#SBATCH --time=0-05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G              


source /etc/profile.d/modules.sh
module load python/uv
cd /vol/miltank/users/$USER/git/adlm-knee-osteoarthritis
source .venv/bin/activate

nvidia-smi
which python
python --version

python main.py \
  --model_name autoencoder_pain \
  --weights_path 'weights_pain_ae/best_pain_ae_phase2_finetune.pth' \
  --side left

# Final message
python -c "print('The job is finished!')"