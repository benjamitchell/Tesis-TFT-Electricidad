#!/bin/bash
#SBATCH --job-name=h24_pc
#SBATCH --array=0-9                           # 10 jobs: 5 barras x {precios, residuos}
#SBATCH --output=slurm_h24pc_%a_%j_out.txt
#SBATCH --error=slurm_h24pc_%a_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210
#SBATCH --exclude=gn004

# ── P+C (Prophet + clima) a horizonte day-ahead, h=24, MIMO, LN ──
# Para comparar S+C contra P+C a igual horizonte (S+C h=24 ya entrenado con
# train_h24.sh). Mismas 5 barras e hiperparametros que S+C h=24. Los dataloaders
# se generaron localmente con Scripts/preparar_h24_pc.py a partir de
# pred_prophet_clima (h=1), cambiando solo max_prediction_length a 24.
#
# P+C necesita dos modelos por barra: el TFT de precios y el TFT de residuos de
# Prophet (despues se combinan localmente con Stacking Optimization). Se
# entrenan como jobs separados para correr en paralelo.

BARRAS=(ATACAMA CHARRUA P.MONTT P.AZUCAR TARAPACA)
MODELOS=(--solo-precios --solo-residuos)

IDX_MODELO=$((SLURM_ARRAY_TASK_ID % 2))
IDX_BARRA=$((SLURM_ARRAY_TASK_ID / 2))

BARRA="${BARRAS[$IDX_BARRA]}"
FLAG="${MODELOS[$IDX_MODELO]}"

BASE_DIR="/home/minas01/BMitchell"
EXPERIMENTO="pred_prophet_clima_h24"
CARPETA_MODELOS="$BASE_DIR/Multi-Modelos_TFT/h/LN/$EXPERIMENTO"
CARPETA_LOGS="$BASE_DIR/Logs_TFT/h/LN/$EXPERIMENTO"

echo "Barra:       $BARRA"
echo "Modelo:      $FLAG"
echo "Experimento: $EXPERIMENTO (P+C, h=24, MIMO, LN)"
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
    $FLAG \
    --verbose
