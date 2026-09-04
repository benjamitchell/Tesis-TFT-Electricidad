#!/bin/bash
#SBATCH --job-name=h24
#SBATCH --array=0-9                           # 10 jobs: 5 barras x 2 arquitecturas
#SBATCH --output=slurm_%a_%j_out.txt
#SBATCH --error=slurm_%a_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210

# ── S+C a horizonte day-ahead, h=24, MIMO (P1 #9 de la revision de Jofre) ──
# Los dataloaders (max_prediction_length=24) ya se generaron localmente con
# Scripts/preparar_h24.py, reusando el datasets_norm/scalers ya calculados de
# pred_sol_clima (sin recomputar features solares/climaticas). Aca solo se
# entrena; no hace falta tocar Preprocesamiento_Transformer.py ni el notebook.
#
# P.AZUCAR y TARAPACA se agregaron para poder comparar limpio (2x2: con/sin
# outliers en train x h=1/h=24) contra el piloto de outliers, que usa esas
# mismas 4 barras (ATACAMA, CHARRUA, P.AZUCAR, TARAPACA). P.MONTT se deja
# como caso control, sin sacarla.

BARRAS=(ATACAMA CHARRUA P.MONTT P.AZUCAR TARAPACA)
VARIANTES=(LN DyT)

N_VARIANTES=${#VARIANTES[@]}
IDX_VARIANTE=$((SLURM_ARRAY_TASK_ID % N_VARIANTES))
IDX_BARRA=$((SLURM_ARRAY_TASK_ID / N_VARIANTES))

BARRA="${BARRAS[$IDX_BARRA]}"
VARIANTE="${VARIANTES[$IDX_VARIANTE]}"

BASE_DIR="/home/minas01/BMitchell"
EXPERIMENTO="pred_sol_clima_h24"
CARPETA_MODELOS="$BASE_DIR/Multi-Modelos_TFT/h/$VARIANTE/$EXPERIMENTO"
CARPETA_LOGS="$BASE_DIR/Logs_TFT/h/$VARIANTE/$EXPERIMENTO"

if [ "$VARIANTE" = "DyT" ]; then
    SCRIPT="train_dyt.py"
else
    SCRIPT="train_tft.py"
fi

echo "Barra:       $BARRA"
echo "Variante:    $VARIANTE"
echo "Experimento: $EXPERIMENTO (h=24, MIMO)"
echo "Script:      $SCRIPT"
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
    --solo-precios \
    --verbose
