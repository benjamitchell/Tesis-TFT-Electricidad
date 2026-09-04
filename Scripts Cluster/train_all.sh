#!/bin/bash
#SBATCH --job-name=h_dyt_pro
#SBATCH --array=0-13                          # 16 jobs: 7 barras × 2 tipos
#SBATCH --output=slurm_%a_%j_out.txt
#SBATCH --error=slurm_%a_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210

# ── Configuración del experimento ──────────────────────────────────────────
VARIANTE="DyT"                   
EXPERIMENTO="pred_sol_clima_6"
FRECUENCIA="h"                 

# Derivado automáticamente
BASE_DIR="/home/minas01/BMitchell"
CARPETA_MODELOS="$BASE_DIR/Multi-Modelos_TFT/$FRECUENCIA/$VARIANTE/$EXPERIMENTO"
CARPETA_LOGS="$BASE_DIR/Logs_TFT/$FRECUENCIA/$VARIANTE/$EXPERIMENTO"

if [ "$VARIANTE" = "DyT" ]; then
    SCRIPT="train_dyt.py"
else
    SCRIPT="train_tft.py"
fi

# ── Mapeo de task ID → barra + tipo ───────────────────────────────────────
BARRAS=(ATACAMA CARDONES CHARRUA CRUCERO QUILLOTA)

# 0-7  → precios  (IDX_TIPO=0)
# 8-15 → residuos (IDX_TIPO=1)
IDX_BARRA=$((SLURM_ARRAY_TASK_ID % 8))
IDX_TIPO=$((SLURM_ARRAY_TASK_ID / 8))
BARRA="${BARRAS[$IDX_BARRA]}"

if [ $IDX_TIPO -eq 1 ]; then
    FLAG="--solo-residuos"
else
    FLAG="--solo-precios"
fi

echo "Variante:    $VARIANTE"
echo "Experimento: $EXPERIMENTO"
echo "Script:      $SCRIPT"
echo "Barra:       $BARRA  ($FLAG)"
echo "Modelos:     $CARPETA_MODELOS"

# ── Entorno ────────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1
source /home/modules/spack/opt/spack/linux-rocky9-zen4/gcc-14.2.0/miniconda3-24.7.1-jwxiannln4jlkqta37mvoxkrrf4tumwh/etc/profile.d/conda.sh
conda activate tft_amd_gpu_env
export HSA_OVERRIDE_GFX_VERSION=9.0.10

python "$SCRIPT" \
    --carpeta-modelos "$CARPETA_MODELOS" \
    --carpeta-logs    "$CARPETA_LOGS" \
    --barras          "$BARRA" \
    $FLAG \
    --verbose
