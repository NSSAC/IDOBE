#!/bin/bash

#SBATCH --job-name=CES
#SBATCH -n 1
#SBATCH --partition=bii
#SBATCH --account=nssac_students
#SBATCH --time=40:00:00
#SBATCH --mem=50000

module load miniforge
conda activate baseline

# Fix for outdated system libstdc++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Limit number of threads to avoid RLIMIT_NPROC issue
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run the script
python CES_PIPELINE.py 56
