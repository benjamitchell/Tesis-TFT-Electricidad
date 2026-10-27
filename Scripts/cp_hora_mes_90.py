#!/usr/bin/env python3
"""Recalcula cobertura por hora y por mes (Split CP, ACI, LW-ACP) al 90% de cobertura
nominal (alpha=0.10, gamma=0.05), y regenera la figura de cobertura y ancho por hora en
espanol (fig_cp_cobertura_ancho_hora.pdf, tesis y presentacion) y en ingles
(fig_cp_cobertura_ancho_hora_en.pdf, paper IEEE). La logica es la reconstruccion fiel
(confirmada por los numeros publicados en la tesis al 95%) del codigo original que genero
la figura en espanol."""
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cp_metricas_completas import marco

RUTA_EVAL = "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"
RUTA_CRUDA = "Datos/h/2020-2026.csv"
CARPETA_SALIDA = "Escrito/imagenes"
BARRAS = ["ATACAMA", "CARDONES", "CHARRUA", "CRUCERO", "P.AZUCAR", "P.MONTT", "QUILLOTA", "TARAPACA"]
ALPHA, GAMMA = 0.10, 0.05

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})



def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    df_crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])

    acumulado_h = {m: {"cov": np.zeros(24), "width": np.zeros(24), "n": np.zeros(24)}
                   for m in ("Split CP", "ACI", "LW-ACP")}
    acumulado_mes = {m: {} for m in ("Split CP", "LW-ACP")}

    for barra in BARRAS:
        serie = df_crudo.loc[df_crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        mv = marco(datos, barra, "val", serie)
        mt = marco(datos, barra, "test", serie)
        horas_t = mt["fecha"].dt.hour.values
        meses_t = mt["fecha"].dt.month.values

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
                acumulado_h[nombre]["cov"][h] += dentro[mask].sum()
                acumulado_h[nombre]["width"][h] += (up[mask] - lo[mask]).sum()
                acumulado_h[nombre]["n"][h] += mask.sum()

            if nombre in acumulado_mes:
                for mes in range(1, 13):
                    mask = meses_t == mes
                    if mask.sum() == 0:
                        continue
                    acumulado_mes[nombre].setdefault(mes, {"cov": 0, "n": 0})
                    acumulado_mes[nombre][mes]["cov"] += dentro[mask].sum()
                    acumulado_mes[nombre][mes]["n"] += mask.sum()
        print(f"[{barra}] ok")

    # transicion vs resto (7,8,19,20,21h)
    horas_transicion = [7, 8, 19, 20, 21]
    for nombre in ("Split CP", "ACI", "LW-ACP"):
        n = acumulado_h[nombre]["n"]
        cov_h = acumulado_h[nombre]["cov"] / n
        sd_h = cov_h.std()
        cov_trans = acumulado_h[nombre]["cov"][horas_transicion].sum() / acumulado_h[nombre]["n"][horas_transicion].sum()
        resto = [h for h in range(24) if h not in horas_transicion]
        cov_resto = acumulado_h[nombre]["cov"][resto].sum() / acumulado_h[nombre]["n"][resto].sum()
        print(f"{nombre}: SD_hora={sd_h:.4f}  cov_transicion={cov_trans:.4f}  cov_resto={cov_resto:.4f}")

    print()
    for nombre in ("Split CP", "LW-ACP"):
        covs = {mes: v["cov"] / v["n"] for mes, v in acumulado_mes[nombre].items()}
        vals = np.array(list(covs.values()))
        peor_mes = min(covs, key=covs.get)
        mejor_mes = max(covs, key=covs.get)
        print(f"{nombre}: SD_mes={vals.std():.4f}  min={covs[peor_mes]:.4f}(mes {peor_mes})  max={covs[mejor_mes]:.4f}(mes {mejor_mes})  rango=[{vals.min():.3f},{vals.max():.3f}]")

    # Misma figura en espanol (tesis, presentacion) y en ingles (paper IEEE)
    colores = {"Split CP": "tab:gray", "ACI": "tab:orange", "LW-ACP": "tab:blue"}
    textos = {
        "fig_cp_cobertura_ancho_hora.pdf": (
            {"Split CP": "Split CP", "ACI": "ACI", "LW-ACP": "LW-ACP"}, "Hora del día",
            "Cobertura empírica", "Cobertura por hora", "Ancho medio del intervalo (USD/MWh)", "Ancho por hora"),
        "fig_cp_cobertura_ancho_hora_en.pdf": (
            {"Split CP": "Split CP", "ACI": "ACI", "LW-ACP": "LW-ACP"}, "Hour of day",
            "Empirical coverage", "Coverage by hour", "Mean interval width (USD/MWh)", "Width by hour"),
    }
    for archivo, (etiquetas, et_hora, et_cov, tit_cov, et_ancho, tit_ancho) in textos.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for nombre in ("Split CP", "ACI", "LW-ACP"):
            n = acumulado_h[nombre]["n"]
            cov_h = acumulado_h[nombre]["cov"] / n
            width_h = acumulado_h[nombre]["width"] / n
            ax1.plot(range(24), cov_h, marker="o", markersize=3, label=etiquetas[nombre], color=colores[nombre])
            ax2.plot(range(24), width_h, marker="o", markersize=3, label=etiquetas[nombre], color=colores[nombre])

        ax1.axhline(0.90, color="black", linestyle=":", linewidth=1)
        ax1.set_xlabel(et_hora); ax1.set_ylabel(et_cov); ax1.set_title(tit_cov)
        ax2.set_xlabel(et_hora); ax2.set_ylabel(et_ancho); ax2.set_title(tit_ancho)
        ax1.legend(fontsize=8); ax1.set_xticks(range(0, 24, 3)); ax2.set_xticks(range(0, 24, 3))
        fig.tight_layout()
        ruta = os.path.join(CARPETA_SALIDA, archivo)
        fig.savefig(ruta, bbox_inches="tight")
        plt.close(fig)
        print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
