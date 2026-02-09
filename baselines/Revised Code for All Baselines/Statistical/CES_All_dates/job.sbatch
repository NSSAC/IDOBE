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
python CES_PIPELINE.py 56
