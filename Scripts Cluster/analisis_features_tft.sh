#!/bin/bash
#SBATCH --job-name=tft_features
#SBATCH --output=slurm_out_features_%j.txt
#SBATCH --error=slurm_err_features_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210

export PYTHONNOUSERSITE=1
source /home/modules/spack/opt/spack/linux-rocky9-zen4/gcc-14.2.0/miniconda3-24.7.1-jwxiannln4jlkqta37mvoxkrrf4tumwh/etc/profile.d/conda.sh
conda activate tft_amd_gpu_env
export HSA_OVERRIDE_GFX_VERSION=9.0.10

python analisis_features_tft.py
