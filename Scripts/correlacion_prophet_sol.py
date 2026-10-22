#!/usr/bin/env python3
"""
correlacion_prophet_sol.py - Mide cuanto se solapan las covariables conocidas de P+C
(componentes de Prophet: trend, yearly, weekly, daily) con las de S+C (elevacion_solar,
cos_elevacion, es_horario_verano), sobre el conjunto de entrenamiento de las 8 barras.

Dos medidas:
  1. Correlacion de Pearson entre cada componente de Prophet y cada variable solar.
  2. R^2 con que las variables solares (juntas, de forma no lineal) reconstruyen cada
     componente de Prophet: HistGradientBoosting, validacion cruzada de 5 bloques
     contiguos (sin barajar, para no filtrar informacion entre horas vecinas).
Un R^2 cercano a 1 indica que el componente es redundante con la posicion solar; cercano a
0, que aporta informacion que la posicion solar no tiene.

Uso:
    python Scripts/correlacion_prophet_sol.py
"""
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

RUTA = "Multi-Modelos_TFT/h/LN/pred_prophet_clima/datasets_norm.pkl"
PROPHET = ["trend", "yearly", "weekly", "daily"]
SOLAR = ["elevacion_solar", "cos_elevacion", "es_horario_verano"]


def main():
    with open(RUTA, "rb") as f:
        datasets_norm = pickle.load(f)

    filas_corr, filas_r2 = [], []
    for barra, d in datasets_norm.items():
        df = d["train"][PROPHET + SOLAR].dropna()
        c = df.corr().loc[PROPHET, SOLAR]
        for comp in PROPHET:
            filas_corr.append({"Barra": barra, "Componente": comp, **c.loc[comp].to_dict()})
            r2 = cross_val_score(HistGradientBoostingRegressor(max_iter=200), df[SOLAR], df[comp],
                                 cv=KFold(5, shuffle=False), scoring="r2").mean()
            filas_r2.append({"Barra": barra, "Componente": comp, "R2": r2})
        print(f"[{barra}] listo", flush=True)

    corr = pd.DataFrame(filas_corr)
    r2 = pd.DataFrame(filas_r2)
    pd.set_option("display.width", 200)
    print("\n=== Correlacion de Pearson, promedio de 8 barras (min--max entre barras) ===")
    for comp in PROPHET:
        g = corr[corr.Componente == comp]
        print(f"{comp:8s} " + "  ".join(f"{s}: {g[s].mean():+.2f} ({g[s].min():+.2f}..{g[s].max():+.2f})" for s in SOLAR))
    print("\n=== R^2 de reconstruir cada componente de Prophet con las variables solares (CV 5 bloques) ===")
    print(r2.pivot(index="Barra", columns="Componente", values="R2")[PROPHET].round(3).to_string())
    print("\nPromedio:", r2.groupby("Componente")["R2"].mean()[PROPHET].round(3).to_dict())

    corr.to_csv("Scripts/resultado_correlacion_prophet_sol.csv", index=False)
    r2.to_csv("Scripts/resultado_r2_prophet_sol.csv", index=False)
    print("\nGuardado: Scripts/resultado_correlacion_prophet_sol.csv y Scripts/resultado_r2_prophet_sol.csv")


if __name__ == "__main__":
    main()
