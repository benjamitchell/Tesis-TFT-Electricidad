#!/usr/bin/env python3
"""
fig_phase_shift_en.py - Genera fig_phase_shift_panel_en.pdf para el paper IEEE: los 4 casos
de mayor error absoluto en validacion de S+C-LN en Atacama (sin traslape de ventanas),
en ventanas de +-48h, contra la serie cruda.

La version en espanol de la tesis viene de un codigo que no quedo guardado; esta
reconstruccion sigue la misma metodologia, pero los 4 casos elegidos no son
necesariamente las mismas horas de esa figura.

Las otras figuras en ingles del paper las generan los mismos scripts que su version en
espanol: Scripts/cp_hora_mes_90.py (cobertura y ancho por hora) y
Scripts/fig_cp_ejemplo_90.py (intervalo LW-ACP de ejemplo).

Uso:
    python Scripts/fig_phase_shift_en.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA

RUTA_EVAL = "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"
RUTA_CRUDA = "Datos/h/2020-2026.csv"
CARPETA_SALIDA = "Escrito/imagenes"
BARRA = "ATACAMA"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    df_crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])
    serie = df_crudo.loc[df_crudo["Barra"] == BARRA].groupby("Fecha")["Valor"].mean()

    d = datos[BARRA]["val"]["Precios"]["datos"]
    mv = pd.DataFrame({"fecha": pd.to_datetime(d["fechas"]), "y_pred": np.asarray(d["y_pred"])})
    mv = mv.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
    mv["y_real"] = serie.reindex(mv["fecha"]).values
    mv = mv.dropna(subset=["y_real"])
    ini_falla, fin_falla = VENTANA_PRECIO_FALLA
    margen = pd.Timedelta(hours=12)
    mv = mv.loc[~((mv["fecha"] >= ini_falla - margen) & (mv["fecha"] <= fin_falla + margen))]
    mv["error"] = (mv["y_real"] - mv["y_pred"]).abs()

    # 4 mayores errores, exigiendo que las ventanas +-48h no se traslapen demasiado
    candidatos = mv.sort_values("error", ascending=False)
    elegidos = []
    for _, row in candidatos.iterrows():
        if all(abs((row["fecha"] - f).total_seconds()) > 4 * 24 * 3600 for f in elegidos):
            elegidos.append(row["fecha"])
        if len(elegidos) == 4:
            break

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, centro in zip(axes.flat, elegidos):
        ini, fin = centro - pd.Timedelta(hours=48), centro + pd.Timedelta(hours=48)
        ventana = mv[(mv["fecha"] >= ini) & (mv["fecha"] <= fin)]
        ax.plot(ventana["fecha"], ventana["y_real"], color="tab:blue", label="Real")
        ax.plot(ventana["fecha"], ventana["y_pred"], color="tab:red", linestyle="--", label="Predicted")
        ax.axvline(centro, color="darkorange", linestyle=":", linewidth=1.5)
        err = mv.loc[mv["fecha"] == centro, "error"].values[0]
        ax.set_title(f"{centro:%Y-%m-%d %H:%M}, error={err:.1f} USD/MWh", fontsize=9)
        ax.set_ylabel("USD/MWh")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        ax.tick_params(axis="x", rotation=30)
    axes.flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Extreme error cases, S+C-LN TFT, Atacama (validation set)")
    fig.tight_layout()
    ruta = os.path.join(CARPETA_SALIDA, "fig_phase_shift_panel_en.pdf")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {ruta}  (casos: {[str(c) for c in elegidos]})")


if __name__ == "__main__":
    main()
