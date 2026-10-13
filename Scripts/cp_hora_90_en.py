#!/usr/bin/env python3
"""Regenera fig_cp_cobertura_ancho_hora_en.pdf (etiquetas en ingles) al 90% de
cobertura nominal, para el paper IEEE. Misma logica que Scripts/cp_hora_mes_90.py."""
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Conformal_Prediction_Wrapper import (split_cp_from_calibration,
                                                  aci_from_calibration,
                                                  lw_acp_online_from_calibration)
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA

RUTA_EVAL = "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"
RUTA_CRUDA = "Datos/h/2020-2026.csv"
CARPETA_SALIDA = "Escrito/imagenes"
BARRAS = ["ATACAMA", "CARDONES", "CHARRUA", "CRUCERO", "P.AZUCAR", "P.MONTT", "QUILLOTA", "TARAPACA"]
ALPHA, GAMMA = 0.10, 0.05

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def marco(datos, barra, conjunto, serie):
    d = datos[barra][conjunto]["Precios"]["datos"]
    m = pd.DataFrame({"fecha": pd.to_datetime(d["fechas"]), "y_pred": np.asarray(d["y_pred"])})
    m = m.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
    m["y_real"] = serie.reindex(m["fecha"]).values
    m = m.dropna(subset=["y_real"])
    ini, fin = VENTANA_PRECIO_FALLA
    return m.loc[~((m["fecha"] >= ini) & (m["fecha"] <= fin))].reset_index(drop=True)


def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    df_crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])

    acumulado = {m: {"cov": np.zeros(24), "width": np.zeros(24), "n": np.zeros(24)}
                 for m in ("Split CP", "ACI", "LW-ACP")}

    for barra in BARRAS:
        serie = df_crudo.loc[df_crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        mv = marco(datos, barra, "val", serie)
        mt = marco(datos, barra, "test", serie)
        horas_t = mt["fecha"].dt.hour.values

        r_scp = split_cp_from_calibration(mv["y_real"].values, mv["y_pred"].values,
                                          mt["y_real"].values, mt["y_pred"].values, alpha=ALPHA)
        r_aci = aci_from_calibration(mv["y_real"].values, mv["y_pred"].values,
                                     mt["y_real"].values, mt["y_pred"].values, alpha=ALPHA, gamma=GAMMA)
        r_lwacp = lw_acp_online_from_calibration(mv["y_real"].values, mv["y_pred"].values, mv["fecha"].values,
                                                 mt["y_real"].values, mt["y_pred"].values, mt["fecha"].values,
                                                 alpha=ALPHA, gamma=GAMMA)

        for nombre, r in (("Split CP", r_scp), ("ACI", r_aci), ("LW-ACP", r_lwacp)):
            y, lo, up = r["y_test"], r["lower"], r["upper"]
            dentro = (y >= lo) & (y <= up)
            for h in range(24):
                mask = horas_t == h
                if mask.sum() == 0:
                    continue
                acumulado[nombre]["cov"][h] += dentro[mask].sum()
                acumulado[nombre]["width"][h] += (up[mask] - lo[mask]).sum()
                acumulado[nombre]["n"][h] += mask.sum()
        print(f"  [{barra}] ok")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colores = {"Split CP": "tab:gray", "ACI": "tab:orange", "LW-ACP": "tab:blue"}
    for nombre in ("Split CP", "ACI", "LW-ACP"):
        n = acumulado[nombre]["n"]
        cov_h = acumulado[nombre]["cov"] / n
        width_h = acumulado[nombre]["width"] / n
        ax1.plot(range(24), cov_h, marker="o", markersize=3, label=nombre, color=colores[nombre])
        ax2.plot(range(24), width_h, marker="o", markersize=3, label=nombre, color=colores[nombre])

    ax1.axhline(0.90, color="black", linestyle=":", linewidth=1)
    ax1.set_xlabel("Hour of day"); ax1.set_ylabel("Empirical coverage"); ax1.set_title("Coverage by hour")
    ax2.set_xlabel("Hour of day"); ax2.set_ylabel("Mean interval width (USD/MWh)"); ax2.set_title("Width by hour")
    ax1.legend(fontsize=8); ax1.set_xticks(range(0, 24, 3)); ax2.set_xticks(range(0, 24, 3))
    fig.tight_layout()
    ruta = os.path.join(CARPETA_SALIDA, "fig_cp_cobertura_ancho_hora_en.pdf")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
