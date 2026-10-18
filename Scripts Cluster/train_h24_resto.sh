#!/bin/bash
#SBATCH --job-name=h24_resto
#SBATCH --array=0-11                          # 12 jobs: 3 barras x 4 modelos
#SBATCH --output=slurm_h24resto_%a_%j_out.txt
#SBATCH --error=slurm_h24resto_%a_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210
#SBATCH --exclude=gn004

# ── Modelos a 24 horas para las 3 barras que faltaban (CARDONES, CRUCERO, QUILLOTA) ──
# Completa a 8 barras las pruebas a 24 horas (heterocedasticidad, Conformal
# Prediction, LN vs DyT, S+C vs P+C) y la comparacion con el CEN a igual horizonte.
# Dataloaders generados localmente con Scripts/preparar_h24_resto.py, con las mismas
# configuraciones que los modelos h24 de las otras cinco barras.
#
# Cuatro modelos por barra:
#   0: S+C con LayerNorm          (train_tft.py,    --solo-precios)
#   1: S+C con DyT                (train_dyt_v2.py, --solo-precios)
#   2: P+C, modelo de precios     (train_tft.py,    --solo-precios)
#   3: P+C, modelo de residuos    (train_tft.py,    --solo-residuos)
# train_dyt_v2.py es una copia del train_dyt.py local (con --seed), subida con otro
# nombre para no pisar el train_dyt.py que ya esta en el cluster.

BARRAS=(CARDONES CRUCERO QUILLOTA)
SCRIPTS=(train_tft.py train_dyt_v2.py train_tft.py train_tft.py)
CARPETAS=(LN/pred_sol_clima_h24_resto DyT/pred_sol_clima_h24_resto LN/pred_prophet_clima_h24_resto LN/pred_prophet_clima_h24_resto)
FLAGS=(--solo-precios --solo-precios --solo-precios --solo-residuos)

IDX_MODELO=$((SLURM_ARRAY_TASK_ID % 4))
IDX_BARRA=$((SLURM_ARRAY_TASK_ID / 4))

BARRA="${BARRAS[$IDX_BARRA]}"
SCRIPT="${SCRIPTS[$IDX_MODELO]}"
CARPETA="${CARPETAS[$IDX_MODELO]}"
FLAG="${FLAGS[$IDX_MODELO]}"

BASE_DIR="/home/minas01/BMitchell"
CARPETA_MODELOS="$BASE_DIR/Multi-Modelos_TFT/h/$CARPETA"
CARPETA_LOGS="$BASE_DIR/Logs_TFT/h/$CARPETA"

echo "Barra:   $BARRA"
echo "Script:  $SCRIPT $FLAG"
echo "Modelos: $CARPETA_MODELOS"

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
