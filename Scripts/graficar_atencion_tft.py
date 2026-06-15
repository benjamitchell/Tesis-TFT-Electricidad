"""
Gráfico combinado de atención TFT (4x2) por experimento.

CONFIGURACIÓN:
    - arquitectura:       "LN" o "DyT"
    - lista_experimentos: ["Multi-TFT_Precios", "Multi-TFT_Residuos"]
    - lista_barras:       8 barras

ORDEN DE PROCESAMIENTO: experimento → barra
    (carga el modelo de cada barra uno a la vez y libera memoria
    antes de pasar al siguiente, para evitar crashes por RAM/VRAM)

SALIDA (por experimento):
    - PNG: Resultados/Features/{experimento}/atencion_combinada_{experimento}.png
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
import gc

sys.path.append(r'C:\Users\56977\OneDrive\Escritorio\Tesis - copia')
from Modulos.TFT_Model import cargar_modelo_entrenado

# ── Configuración ──────────────────────────────────────────────────────────
freq               = "h"
max_pred           = 1
max_encoder        = 168
arquitectura       = "LN"               # "LN" o "DyT"
lista_experimentos = ["Multi-TFT_Precios", "Multi-TFT_Residuos"]
lista_barras       = ['ATACAMA', 'CARDONES', 'CHARRUA', 'CRUCERO',
                      'P.AZUCAR', 'P.MONTT', 'QUILLOTA', 'TARAPACA']

carpeta_modelos = (
    f"Multi-Modelos_TFT/{freq}/{arquitectura}/"
    f"pred_{max_pred}_{max_encoder}_cluster_con_clima"
)
carpeta_salida_base = os.path.join(carpeta_modelos, "Resultados", "Features")

# ── Procesamiento: experimento → barra ─────────────────────────────────────
for experimento in lista_experimentos:
    print(f"\n{'#'*60}")
    print(f"#  {experimento}")
    print(f"{'#'*60}")

    dl_file = "dataloaders_precios.pkl" if "Precios" in experimento else "dataloaders_residuos.pkl"
    with open(f'{carpeta_modelos}/{dl_file}', 'rb') as f:
        dataloaders = pickle.load(f)

    n_cols = 2
    n_rows = (len(lista_barras) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, barra in enumerate(lista_barras):
        print(f"\n  {barra} — {experimento}")

        modelo, _ = cargar_modelo_entrenado(barra, experimento, carpeta_modelos=carpeta_modelos)
        if modelo is None:
            print(f"  [SKIP] No se encontró modelo para {barra}")
            axes[idx].set_visible(False)
            continue
        modelo.eval()

        with torch.no_grad():
            raw_predictions = modelo.predict(dataloaders[barra]['test'], mode="raw", return_x=True)
            interpretation  = modelo.interpret_output(raw_predictions.output, reduction="mean")

        attention = interpretation["attention"].detach().cpu().clone()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        x = np.arange(-max_encoder, attention.size(0) - max_encoder)

        ax = axes[idx]
        ax.plot(x, attention, linewidth=1.2)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_title(f"Barra {barra}", fontweight='bold')
        ax.set_xlabel("Time index")
        ax.set_ylabel("Atención")
        ax.grid(True, alpha=0.3)

        # ── Liberar memoria antes de la siguiente barra ──────────────────
        del modelo, raw_predictions, interpretation, attention
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Atención — TFT_{arquitectura} — {experimento}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    carpeta_salida = os.path.join(carpeta_salida_base, experimento)
    os.makedirs(carpeta_salida, exist_ok=True)
    out_path = os.path.join(carpeta_salida, f"atencion_combinada_{experimento}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Guardado: {out_path}")
    plt.close(fig)

    del dataloaders
    gc.collect()

print("\nAnálisis de atención completado.")
