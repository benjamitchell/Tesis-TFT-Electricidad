#!/usr/bin/env python3
"""Regenera el ejemplo de intervalo LW-ACP (Atacama, 4-11 marzo 2026) al 90% de
cobertura nominal, en espanol (tesis, presentacion) y en ingles (paper IEEE)."""
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
from Modulos.Conformal_Prediction_Wrapper import lw_acp_online_from_calibration
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA

RUTA_EVAL = "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"
BARRA, ALPHA, GAMMA = "ATACAMA", 0.10, 0.05
INI, FIN = pd.Timestamp("2026-03-04"), pd.Timestamp("2026-03-11")
TEXTOS = {
    "fig_cp_ejemplo_intervalo.pdf": ("Intervalo LW-ACP (90%)", "Real", "Predicción S+C-LN",
                                     "Ejemplo de intervalo de predicción LW-ACP, Atacama"),
    "fig_cp_ejemplo_intervalo_en.pdf": ("LW-ACP interval (90%)", "Real", "S+C-LN prediction",
                                        "Example LW-ACP prediction interval, Atacama"),
}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def marco(datos, conjunto, serie):
    d = datos[BARRA][conjunto]["Precios"]["datos"]
    m = pd.DataFrame({"fecha": pd.to_datetime(d["fechas"]), "y_pred": np.asarray(d["y_pred"])})
    m = m.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
    m["y_real"] = serie.reindex(m["fecha"]).values
    m = m.dropna(subset=["y_real"])
    ini, fin = VENTANA_PRECIO_FALLA
    return m.loc[~((m["fecha"] >= ini) & (m["fecha"] <= fin))].reset_index(drop=True)


def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    df_crudo = pd.read_csv("Datos/h/2020-2026.csv", sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])
    serie = df_crudo.loc[df_crudo["Barra"] == BARRA].groupby("Fecha")["Valor"].mean()
    mv, mt = marco(datos, "val", serie), marco(datos, "test", serie)

    r = lw_acp_online_from_calibration(mv["y_real"].values, mv["y_pred"].values, mv["fecha"].values,
                                       mt["y_real"].values, mt["y_pred"].values, mt["fecha"].values,
                                       alpha=ALPHA, gamma=GAMMA)
    v = pd.DataFrame({"fecha": mt["fecha"].values, "y_real": r["y_test"], "y_pred": mt["y_pred"].values,
                      "lower": r["lower"], "upper": r["upper"]})
    v = v[(v["fecha"] >= INI) & (v["fecha"] < FIN)]
    fuera = int(((v["y_real"] < v["lower"]) | (v["y_real"] > v["upper"])).sum())
    print(f"Ventana: {len(v)} observaciones, {fuera} fuera del intervalo ({100 * fuera / len(v):.1f}%), "
          f"ancho medio {np.mean(v['upper'] - v['lower']):.2f}")

    for archivo, (et_int, et_real, et_pred, titulo) in TEXTOS.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.fill_between(v["fecha"], v["lower"], v["upper"], color="tab:blue", alpha=0.2, label=et_int)
        ax.plot(v["fecha"], v["y_real"], color="black", linewidth=1.2, label=et_real)
        ax.plot(v["fecha"], v["y_pred"], color="tab:red", linestyle="--", linewidth=1, label=et_pred)
        ax.set_ylabel("USD/MWh")
        ax.set_title(titulo)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        ax.legend(fontsize=8)
        fig.tight_layout()
        ruta = os.path.join("Escrito/imagenes", archivo)
        fig.savefig(ruta, bbox_inches="tight")
        plt.close(fig)
        print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
