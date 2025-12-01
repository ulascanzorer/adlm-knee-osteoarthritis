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
 
# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you always run this script from a clean terminal
# conda deactivate
 
# If the following does not work, try 'source activate <env-name>'
#conda activate deanon




# Run the program
nvidia-smi

sleep 10

uv run main.py \
  --model_name autoencoder \
  --weights_path trained_knee_3d_autoencoder_n980.pth \
  --side left \
  --max_patients 1000

python -c "The job is finished!"
