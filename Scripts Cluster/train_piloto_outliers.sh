#!/bin/bash
#SBATCH --job-name=piloto_out
#SBATCH --array=0-3                           # 4 jobs: 4 barras, arquitectura LN
#SBATCH --output=slurm_%a_%j_out.txt
#SBATCH --error=slurm_%a_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210

# ── Piloto: train sin recorte automatico de outliers (solo se excluyeron ──
# manualmente las 3 horas absurdas de P.AZUCAR, 2020-03-19 17-19h). Val/test
# ya evaluaban contra la serie cruda desde el Paso 1; ahora train tambien la
# ve casi completa. Dataloaders generados localmente con
# Scripts/(scratchpad)/preparar_piloto_outliers.py, que reentreno Prophet
# (CPU) con el nuevo train y rehizo el merge de clima/features solares.
# Si el MAE en regimen alto mejora contra los modelos actuales, se entrenan
# las 8 barras completas con el mismo tratamiento.

BARRAS=(ATACAMA CHARRUA P.AZUCAR TARAPACA)
BARRA="${BARRAS[$SLURM_ARRAY_TASK_ID]}"

BASE_DIR="/home/minas01/BMitchell"
EXPERIMENTO="pred_sol_clima_piloto_sin_outliers"
CARPETA_MODELOS="$BASE_DIR/Multi-Modelos_TFT/h/LN/$EXPERIMENTO"
CARPETA_LOGS="$BASE_DIR/Logs_TFT/h/LN/$EXPERIMENTO"

echo "Barra:       $BARRA"
echo "Experimento: $EXPERIMENTO (piloto sin recorte de outliers en train)"
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
