#!/bin/bash
#SBATCH --job-name=global_sin_out
#SBATCH --output=slurm_%j_out.txt
#SBATCH --error=slurm_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210
#SBATCH --exclude=gn004

# ── Modelo TFT GLOBAL sin recorte de outliers en train: las ocho barras del
# piloto (pred_sol_clima_piloto_sin_outliers, train sin Z-score) pooled en un
# unico TimeSeriesDataSet, igual que train_global.sh pero partiendo de datos
# de train sin limpiar. Dataloaders generados localmente con
# Scripts/preparar_modelo_global.py --sin-outliers. Un solo job, no un array.

BASE_DIR="/home/minas01/BMitchell"
EXPERIMENTO="pred_sol_clima_global_sin_outliers"
CARPETA_MODELOS="$BASE_DIR/Multi-Modelos_TFT/h/LN/$EXPERIMENTO"
CARPETA_LOGS="$BASE_DIR/Logs_TFT/h/LN/$EXPERIMENTO"

echo "Barra:       GLOBAL (8 barras pooled, sin recorte de outliers en train)"
echo "Variante:    LN"
echo "Experimento: $EXPERIMENTO"
echo "Modelos:     $CARPETA_MODELOS"

# ── Entorno ────────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1
source /home/modules/spack/opt/spack/linux-rocky9-zen4/gcc-14.2.0/miniconda3-24.7.1-jwxiannln4jlkqta37mvoxkrrf4tumwh/etc/profile.d/conda.sh
conda activate tft_amd_gpu_env
export HSA_OVERRIDE_GFX_VERSION=9.0.10

python train_tft.py \
    --carpeta-modelos "$CARPETA_MODELOS" \
    --carpeta-logs    "$CARPETA_LOGS" \
    --barras          GLOBAL \
    --solo-precios \
    --verbose
