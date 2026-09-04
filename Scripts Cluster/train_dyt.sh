#!/bin/bash
#SBATCH --job-name=montt_15_dyt
#SBATCH --output=slurm_out_%j.txt
#SBATCH --error=slurm_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210

export PYTHONNOUSERSITE=1

source /home/modules/spack/opt/spack/linux-rocky9-zen4/gcc-14.2.0/miniconda3-24.7.1-jwxiannln4jlkqta37mvoxkrrf4tumwh/etc/profile.d/conda.sh

# 1. Activar el entorno virtual que creamos y limpiamos
conda activate tft_amd_gpu_env

# 2. Asegurar la compatibilidad de arquitectura para la MI210
export HSA_OVERRIDE_GFX_VERSION=9.0.10

# 3. Lanzar el entrenamiento
python train_dyt.py \
    --carpeta-modelos /home/minas01/BMitchell/Multi-Modelos_TFT/h/DyT/pred_sol_clima_6 \
    --carpeta-logs /home/minas01/BMitchell/Multi-Modelos_TFT/h/DyT/pred_sol_clima_6 \
    --barras  \
    --solo-precios \
    --verbose