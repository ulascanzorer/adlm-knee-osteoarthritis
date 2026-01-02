#!/bin/bash

#SBATCH --job-name=knee_pain_input_ae_train
#SBATCH --output=train_pain_input_ae_%j.out
#SBATCH --error=train_pain_input_ae_%j.err
#SBATCH --qos=master-queuesave
#SBATCH --time=0-14:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G               

source /etc/profile.d/modules.sh
module load python/uv

uv sync

source .venv/bin/activate

echo "Python interpreter:"
which python

echo "Python version:"
python --version

echo "GPU status:"
nvidia-smi
echo

cd /vol/miltank/users/$USER/Documents/adlm-knee-osteoarthritis

echo "STARTING PAIN-INPUT AE TRAINING"
echo "Running ae_pain_input/train.py..."

python ae_pain_input/train.py

echo "Training DONE"