#!/bin/bash
#SBATCH --job-name=h24_bigcap
#SBATCH --array=0-1                           # 2 jobs: ATACAMA y P.MONTT, solo LN
#SBATCH --output=slurm_%a_%j_out.txt
#SBATCH --error=slurm_%a_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210

# ── S+C h=24 (MIMO) con mas capacidad, para ver si el degradado del dia-ahead
# completo (ver evaluar_h24.py) mejora con un modelo mas grande. Prueba acotada
# a 2 barras (ATACAMA: caso solar tipico, P.MONTT: el mas dificil) y solo LN,
# para no comprometer mas tiempo de cluster del necesario. Unico cambio vs.
# pred_sol_clima_h24: hidden_size 32->64, cont_size 8->16 (tft_config.json),
# aislado como unica variable -- mismos dataloaders/scalers/datos que el h24
# original, solo copiados a esta carpeta nueva.

BARRAS=(ATACAMA P.MONTT)
BARRA="${BARRAS[$SLURM_ARRAY_TASK_ID]}"

BASE_DIR="/home/minas01/BMitchell"
EXPERIMENTO="pred_sol_clima_h24_bigcap"
CARPETA_MODELOS="$BASE_DIR/Multi-Modelos_TFT/h/LN/$EXPERIMENTO"
CARPETA_LOGS="$BASE_DIR/Logs_TFT/h/LN/$EXPERIMENTO"

echo "Barra:       $BARRA"
echo "Variante:    LN"
echo "Experimento: $EXPERIMENTO (h=24, MIMO, hidden_size=64/cont_size=16)"
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
