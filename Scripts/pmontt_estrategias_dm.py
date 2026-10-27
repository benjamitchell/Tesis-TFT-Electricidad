#!/usr/bin/env python3
"""
pmontt_estrategias_dm.py - Comparacion de las cuatro estrategias de Puerto Montt (H+C, P+C, S+C y
S+H+C) en ambas arquitecturas, Seccion 4.8: metricas de prueba de cada estrategia y test de
Diebold-Mariano entre pares, en las mismas dos versiones que las Tablas de LN contra DyT y de P+C
contra S+C (horaria con 24 rezagos y diaria con truncamiento de Andrews).

P+C es la mejor estrategia de Stacking elegida en validacion (mismo protocolo que
stacking_validacion_tablas.py); H+C, S+C y S+H+C son predicciones directas del TFT. Todo contra la
serie cruda, excluyendo VENTANA_PRECIO_FALLA. Las metricas de cada estrategia usan sus propias
fechas de prueba; el DM de cada par, la interseccion de fechas de ambas.

Uso:
    python Scripts/pmontt_estrategias_dm.py
"""
import os
import sys
import pickle
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA
from stacking_validacion_tablas import RUTA_CRUDA
from dm_sensibilidad_rezagos import cargar, evaluar, holm

BARRA = "P.MONTT"
DIRECTAS = {"H+C": "pred_hidrico_clima", "S+H+C": "pred_sol_hidrico_clima_PMONTT"}
ORDEN = ["H+C", "P+C", "S+C", "S+H+C"]


def main():
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    serie = crudo.loc[crudo["Barra"] == BARRA].groupby("Fecha")["Valor"].mean()
    ini, fin = VENTANA_PRECIO_FALLA
    err = cargar()

    filas_met, filas_dm = [], []
    for arq in ("LN", "DyT"):
        e = err[(arq, BARRA)]
        pred = {"P+C": pd.Series(e["PC"], index=e["fechas"]), "S+C": pd.Series(e["SC"], index=e["fechas"])}
        for nombre, carpeta in DIRECTAS.items():
            datos, _ = pickle.load(open(f"Multi-Modelos_TFT/h/{arq}/{carpeta}/Resultados/eval_TFT_{arq}.pkl", "rb"))
            d = datos[BARRA]["test"]["Precios"]["datos"]
            s = pd.Series(np.asarray(d["y_pred"], dtype=float), index=pd.to_datetime(d["fechas"]))
            s = s[~s.index.duplicated()]
            y = serie.reindex(s.index).values
            y[(s.index >= ini) & (s.index <= fin)] = np.nan
            pred[nombre] = pd.Series(np.abs(y - s.values), index=s.index)
        # metricas por estrategia (errores absolutos ya calculados contra la serie cruda)
        for nombre in ORDEN:
            a = pred[nombre].dropna()
            y = serie.reindex(a.index).values
            filas_met.append({"Estrategia": nombre, "Arq": arq, "n": len(a), "MAE": a.mean(),
                              "RMSE": np.sqrt((a ** 2).mean()),
                              "R2": 1 - (a ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                              "P90": np.percentile(a, 90), "P95": np.percentile(a, 95), "Max": a.max()})
        for n1, n2 in combinations(ORDEN, 2):
            idx = pred[n1].dropna().index.intersection(pred[n2].dropna().index)
            r = evaluar(pred[n1][idx].values, pred[n2][idx].values, idx)
            filas_dm.append({"Arq": arq, "Comparacion": f"{n1} vs {n2}", **r})

    met = pd.DataFrame(filas_met).sort_values("MAE")
    dm = pd.DataFrame(filas_dm)
    dm["holm_24"] = holm(dm["p_24"].values)
    dm["holm_diario"] = holm(dm["p_diario"].values)
    met.to_csv("Scripts/resultado_pmontt_estrategias.csv", index=False)
    dm.to_csv("Scripts/resultado_pmontt_dm.csv", index=False)

    pd.set_option("display.width", 250)
    print(met.round(3).to_string(index=False))
    print("\nDM < 0 favorece a la primera estrategia")
    print(dm[["Arq", "Comparacion", "n", "MAE_a", "MAE_b", "DM_24", "p_24", "holm_24", "lag_diario",
              "DM_diario", "p_diario", "holm_diario"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
