#!/usr/bin/env python3
"""
h24_cp_metodos.py - Compara los cuatro metodos de Conformal Prediction de la tesis
(Split CP, ACI, LW Split CP y LW-ACP, gamma=0.05) sobre el modelo day-ahead S+C-LN,
calibrando cada metodo por separado en cada paso del horizonte (h=1..24): calibracion
en validacion y evaluacion en prueba de ese mismo paso, contra la serie cruda,
excluyendo VENTANA_PRECIO_FALLA. 90% de cobertura nominal, el nivel principal.

Lee el cache de Scripts/extraer_predicciones_h24.py.

Uso:
    python Scripts/h24_cp_metodos.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA
from cp_metricas_completas import evaluar_metodo, pinball

CACHE = "Multi-Modelos_TFT/h/LN/pred_sol_clima_h24/Resultados/predicciones_por_paso.pkl"
RUTA_CRUDA = "Datos/h/2020-2026.csv"
ALPHA = 0.10
METODOS = ["Split CP", "ACI", "LW Split CP", "LW-ACP"]


def marco(d, serie):
    m = pd.DataFrame({"fecha": pd.to_datetime(d["fechas"]), "y_pred": np.asarray(d["y_pred"])})
    m = m.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
    m["y_real"] = serie.reindex(m["fecha"]).values
    m = m.dropna(subset=["y_real"])
    ini, fin = VENTANA_PRECIO_FALLA
    return m.loc[~((m["fecha"] >= ini) & (m["fecha"] <= fin))].reset_index(drop=True)


def main():
    with open(CACHE, "rb") as f:
        preds = pickle.load(f)
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])

    filas = []
    for barra, por_paso in preds.items():
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        for h, d in por_paso.items():
            mv, mt = marco(d["val"], serie), marco(d["test"], serie)
            for metodo in METODOS:
                r = evaluar_metodo(metodo, mv, mt, ALPHA)
                y, lo, up = r["y_test"], r["lower"], r["upper"]
                cov = float(np.mean((y >= lo) & (y <= up)))
                winkler = float(np.mean((up - lo) + (2 / ALPHA) * np.maximum(lo - y, 0)
                                        + (2 / ALPHA) * np.maximum(y - up, 0)))
                pb = (pinball(y, lo, ALPHA / 2) + pinball(y, up, 1 - ALPHA / 2)) / 2
                filas.append({"Barra": barra, "h": h, "Metodo": metodo, "Cobertura": cov,
                              "Desv": abs(cov - (1 - ALPHA)), "Ancho": float(np.mean(up - lo)),
                              "Winkler": winkler, "Pinball": pb})
        print(f"[{barra}] listo", flush=True)

    df = pd.DataFrame(filas)
    pd.set_option("display.width", 200)
    print("\n=== Promedio de las 5 barras y los 24 pasos, 90% nominal ===")
    print(df.groupby("Metodo")[["Cobertura", "Desv", "Ancho", "Winkler", "Pinball"]].mean()
          .loc[METODOS].round(4).to_string())

    print("\n=== Por paso (promedio de 5 barras): Winkler por metodo ===")
    print(df.pivot_table(index="h", columns="Metodo", values="Winkler").loc[:, METODOS].round(2).to_string())

    ganador = df.loc[df.groupby(["Barra", "h"])["Winkler"].idxmin()]
    print("\n=== Metodo con menor Winkler en cada combinacion barra-paso (de 120) ===")
    print(ganador["Metodo"].value_counts().to_string())
    print("\nPor barra:")
    print(ganador.groupby("Barra")["Metodo"].value_counts().unstack(fill_value=0).to_string())

    print("\n=== Promedio por barra (24 pasos) ===")
    print(df.pivot_table(index="Barra", columns="Metodo", values=["Winkler", "Desv"]).round(3).to_string())

    df.to_csv("Scripts/resultado_h24_cp_metodos.csv", index=False)
    print("\nGuardado: Scripts/resultado_h24_cp_metodos.csv")


if __name__ == "__main__":
    main()
