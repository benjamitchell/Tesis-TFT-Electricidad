#!/bin/bash
#SBATCH --job-name=piloto_out_h24
#SBATCH --array=0-3                           # 4 jobs: 4 barras, arquitectura LN, horizonte 24
#SBATCH --output=slurm_%a_%j_out.txt
#SBATCH --error=slurm_%a_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210

# ── Piloto combinado: train sin recorte automatico de outliers (mismo criterio ──
# que train_piloto_outliers.sh) + horizonte day-ahead h=24 (MIMO), en vez de h=1.
# Dataloaders generados localmente reusando el datasets_norm del piloto de
# outliers (Scripts/(scratchpad)/preparar_piloto_outliers_h24.py), sin
# recomputar Prophet ni features solares/climaticas de nuevo.

BARRAS=(ATACAMA CHARRUA P.AZUCAR TARAPACA)
BARRA="${BARRAS[$SLURM_ARRAY_TASK_ID]}"

BASE_DIR="/home/minas01/BMitchell"
EXPERIMENTO="pred_sol_clima_piloto_sin_outliers_h24"
CARPETA_MODELOS="$BASE_DIR/Multi-Modelos_TFT/h/LN/$EXPERIMENTO"
CARPETA_LOGS="$BASE_DIR/Logs_TFT/h/LN/$EXPERIMENTO"

echo "Barra:       $BARRA"
echo "Experimento: $EXPERIMENTO (sin outliers en train + h=24)"
echo "Modelos:     $CARPETA_MODELOS"

# ── Entorno ────────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1
source /home/modules/spack/opt/spack/linux-rocky9-zen4/gcc-14.2.0/miniconda3-24.7.1-jwxiannln4jlkqta37mvoxkrrf4tumwh/etc/profile.d/conda.sh
conda activate tft_amd_gpu_env
export HSA_OVERRIDE_GFX_VERSION=9.0.10

python train_tft.py \
    --carpeta-modelos "$CARPETA_MODELOS" \
    --carpeta-logs    "$CARPETA_LOGS" \
    --barras          "$BARRA" \
    --solo-precios \
    --verbose
