#!/bin/bash
#SBATCH --job-name=piloto_out_resto
#SBATCH --array=0-3                           # 4 jobs: las 4 barras que faltaban del piloto, arquitectura LN
#SBATCH --output=slurm_%a_%j_out.txt
#SBATCH --error=slurm_%a_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210

# ── Piloto sin recorte de outliers, extension a las 4 barras restantes del
# SEN (Cardones, Crucero, P.Montt, Quillota) -- completa las 8 barras del
# piloto original de 4 (Atacama, Charrua, P.Azucar, Tarapaca, ver
# train_piloto_outliers.sh). Dataloaders generados localmente con
# Scripts/preparar_piloto_outliers_resto.py, que agrega estas 4 barras al
# mismo pred_sol_clima_piloto_sin_outliers sin tocar las 4 ya entrenadas.

BARRAS=(CARDONES CRUCERO P.MONTT QUILLOTA)
BARRA="${BARRAS[$SLURM_ARRAY_TASK_ID]}"

BASE_DIR="/home/minas01/BMitchell"
EXPERIMENTO="pred_sol_clima_piloto_sin_outliers"
CARPETA_MODELOS="$BASE_DIR/Multi-Modelos_TFT/h/LN/$EXPERIMENTO"
CARPETA_LOGS="$BASE_DIR/Logs_TFT/h/LN/$EXPERIMENTO"

echo "Barra:       $BARRA"
echo "Experimento: $EXPERIMENTO (piloto sin recorte de outliers en train, resto de barras)"
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
