#!/bin/bash
#SBATCH --job-name=autobench
#SBATCH -A bii_nssac 
#SBATCH --time=5:00:00
# #SBATCH --nodes=1 --ntasks-per-node=30
# #SBATCH -p bii
# #SBATCH -p standard
#SBATCH --ntasks=10
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-%j.out  # Standard output log path
#SBATCH --error=logs/slurm-%j.err   # Standard error log path

#export CONDA_ROOT="/home/$USER/miniconda3"
#source "$CONDA_ROOT/etc/profile.d/conda.sh"
source activate base
conda activate nix

python AutoVanillaTransformer.py

