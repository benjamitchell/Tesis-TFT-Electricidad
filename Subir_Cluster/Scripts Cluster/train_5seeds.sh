#!/bin/bash
#SBATCH --job-name=5seeds
#SBATCH --array=0-29                          # 30 jobs: 3 barras x 2 arq x 5 semillas
#SBATCH --output=slurm_%a_%j_out.txt
#SBATCH --error=slurm_%a_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mi210

# ── Estudio de robustez: >=5 semillas por arquitectura (P1 #7 de la revision  ──
# ── de Jofre), para separar el efecto LayerNorm/DyT del efecto de init.      ──
# Reusa los dataloaders ya generados de pred_sol_clima (S+C), solo cambia la
# semilla de pytorch_lightning.seed_everything antes de entrenar. Cada semilla
# se guarda en su propia carpeta, sin pisar el pipeline principal.

BARRAS=(ATACAMA CHARRUA P.MONTT)              # norte / centro / Puerto Montt (sugerido por Jofre)
VARIANTES=(LN DyT)
SEMILLAS=(1 2 3 4 5)

N_VARIANTES=${#VARIANTES[@]}
N_SEMILLAS=${#SEMILLAS[@]}

IDX_SEMILLA=$((SLURM_ARRAY_TASK_ID % N_SEMILLAS))
IDX_VARIANTE=$(((SLURM_ARRAY_TASK_ID / N_SEMILLAS) % N_VARIANTES))
IDX_BARRA=$((SLURM_ARRAY_TASK_ID / (N_SEMILLAS * N_VARIANTES)))

BARRA="${BARRAS[$IDX_BARRA]}"
VARIANTE="${VARIANTES[$IDX_VARIANTE]}"
SEMILLA="${SEMILLAS[$IDX_SEMILLA]}"

BASE_DIR="/home/minas01/BMitchell"
EXPERIMENTO_ORIGEN="pred_sol_clima"           # de aca se reusan los dataloaders ya generados
CARPETA_ORIGEN="$BASE_DIR/Multi-Modelos_TFT/h/$VARIANTE/$EXPERIMENTO_ORIGEN"

CARPETA_MODELOS="$BASE_DIR/Multi-Modelos_TFT/h/$VARIANTE/pred_sol_clima_seeds/seed${SEMILLA}"
CARPETA_LOGS="$BASE_DIR/Logs_TFT/h/$VARIANTE/pred_sol_clima_seeds/seed${SEMILLA}"

if [ "$VARIANTE" = "DyT" ]; then
    SCRIPT="train_dyt.py"
else
    SCRIPT="train_tft.py"
fi

echo "Barra:       $BARRA"
echo "Variante:    $VARIANTE"
echo "Semilla:     $SEMILLA"
echo "Script:      $SCRIPT"
echo "Dataloaders: $CARPETA_ORIGEN"
echo "Modelos:     $CARPETA_MODELOS"

# ── Entorno ────────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1
source /home/modules/spack/opt/spack/linux-rocky9-zen4/gcc-14.2.0/miniconda3-24.7.1-jwxiannln4jlkqta37mvoxkrrf4tumwh/etc/profile.d/conda.sh
conda activate tft_amd_gpu_env
export HSA_OVERRIDE_GFX_VERSION=9.0.10

python "$SCRIPT" \
    --carpeta-modelos     "$CARPETA_MODELOS" \
    --carpeta-logs        "$CARPETA_LOGS" \
    --dataloaders-precios "$CARPETA_ORIGEN/dataloaders_precios.pkl" \
    --dataloaders-residuos "$CARPETA_ORIGEN/dataloaders_residuos.pkl" \
    --tft-config          "$CARPETA_ORIGEN/tft_config.json" \
    --barras              "$BARRA" \
    --seed                "$SEMILLA" \
    --solo-precios \
    --verbose
