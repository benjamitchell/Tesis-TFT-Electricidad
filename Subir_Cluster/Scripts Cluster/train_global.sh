#!/bin/bash
#SBATCH --job-name=global
#SBATCH --output=slurm_%j_out.txt
#SBATCH --error=slurm_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210

# ── Modelo TFT GLOBAL: una unica instancia entrenada sobre las ocho barras
# juntas (serie_id como covariable estatica real, no constante como en el
# resto del pipeline), en vez de un modelo separado por barra. Un solo job,
# no un array -- los dataloaders ya vienen con las 8 barras mezcladas
# (Scripts/preparar_modelo_global.py).

BASE_DIR="/home/minas01/BMitchell"
EXPERIMENTO="pred_sol_clima_global"
CARPETA_MODELOS="$BASE_DIR/Multi-Modelos_TFT/h/LN/$EXPERIMENTO"
CARPETA_LOGS="$BASE_DIR/Logs_TFT/h/LN/$EXPERIMENTO"

echo "Barra:       GLOBAL (8 barras pooled)"
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
