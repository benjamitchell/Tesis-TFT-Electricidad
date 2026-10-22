#!/usr/bin/env python3
"""
cp_granularidad.py - Conformal Prediction por granularidad (Dia, Horario, 15 min),
Split CP y LW-ACP (gamma=0.05), para ATACAMA, CARDONES y P.MONTT, LN y DyT, contra la
serie cruda de cada granularidad. Calcula 95% (para verificar contra la tabla ya
reportada) y 90% (nivel principal del capitulo de CP).

Uso:
    python Scripts/cp_granularidad.py
"""
import os
import sys
import glob
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Conformal_Prediction_Wrapper import split_cp_from_calibration, lw_acp_online_from_calibration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cp_metricas_completas import marco

BARRAS = ["ATACAMA", "CARDONES", "P.MONTT"]
GAMMA = 0.05
GRANULARIDADES = {"Día": "D", "Horario": "h", "15 min": "15min"}


def series_crudas():
    h = pd.read_csv("Datos/h/2020-2026.csv", sep=";")
    h["Fecha"] = pd.to_datetime(h["Fecha"])
    m = pd.read_csv("Datos/Min/2024-2026_15min.csv", sep=";", decimal=",")
    m["Fecha"] = pd.to_datetime(m["Fecha"])
    # La serie diaria es un producto propio del Coordinador (archivos mensuales
    # CmgBarras), no el promedio diario de la serie horaria.
    d = pd.concat([pd.read_csv(f, sep=";", decimal=",") for f in glob.glob("Datos/D/20*/*.csv")])
    d["Barra"] = d["barra"].str.replace("_", "").str.replace("220", "").str.strip()
    d["Fecha"] = pd.to_datetime(dict(year=d["anio"], month=d["mes"], day=d["dia"]))
    out = {}
    for barra in BARRAS:
        out[("h", barra)] = h.loc[h["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        out[("D", barra)] = d.loc[d["Barra"] == barra].groupby("Fecha")["valor"].mean()
        out[("15min", barra)] = m.loc[m["Barra"] == barra].groupby("Fecha")["Valor"].mean()
    return out



def main():
    crudas = series_crudas()
    filas = []
    for barra in BARRAS:
        for arq in ("DyT", "LN"):
            for nombre, carpeta in GRANULARIDADES.items():
                with open(f"Multi-Modelos_TFT/{carpeta}/{arq}/pred_sol_clima/Resultados/eval_TFT_{arq}.pkl", "rb") as f:
                    datos = pickle.load(f)[0]
                serie = crudas[(carpeta, barra)]
                mv, mt = marco(datos, barra, "val", serie), marco(datos, barra, "test", serie)
                fila = {"Barra": barra, "Arq": arq, "Granularidad": nombre}
                for alpha, niv in ((0.05, "95"), (0.10, "90")):
                    r = split_cp_from_calibration(mv["y_real"].values, mv["y_pred"].values,
                                                  mt["y_real"].values, mt["y_pred"].values, alpha=alpha)
                    fila[f"SCP_cov_{niv}"], fila[f"SCP_w_{niv}"] = round(r["coverage"], 4), round(r["width"], 2)
                    if carpeta != "D":
                        r = lw_acp_online_from_calibration(mv["y_real"].values, mv["y_pred"].values, mv["fecha"].values,
                                                           mt["y_real"].values, mt["y_pred"].values, mt["fecha"].values,
                                                           alpha=alpha, gamma=GAMMA)
                        fila[f"LW_cov_{niv}"], fila[f"LW_w_{niv}"] = round(r["coverage"], 4), round(r["width"], 2)
                filas.append(fila)
    df = pd.DataFrame(filas)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    df.to_csv("Scripts/resultado_cp_granularidad.csv", index=False)
    print("\nGuardado: Scripts/resultado_cp_granularidad.csv")


if __name__ == "__main__":
    main()
