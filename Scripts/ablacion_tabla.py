#!/usr/bin/env python3
"""
ablacion_tabla.py - Tabla de la ablacion de features (Seccion de ablacion del Capitulo 4): TFT de
precios LN sin features exogenas, con las de P+C (componentes de Prophet y clima) y con las de S+C
(posicion solar y clima), en prueba, contra la serie cruda y sin la ventana de precio de falla.

Como P+C y S+C comparten las variables climaticas, su diferencia aisla el aporte de la posicion
solar frente a los componentes de Prophet; la diferencia con el modelo sin features mide el aporte
de cada conjunto de variables exogenas completo. Se agrega el test de Diebold-Mariano horario
(Newey-West, 24 rezagos) de cada modelo con features contra el modelo sin features y de S+C
contra P+C.

Uso:
    python Scripts/ablacion_tabla.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA
from dm_sensibilidad_rezagos import evaluar, RUTA_CRUDA

RUTAS = {"Sin features": "Multi-Modelos_TFT/h/LN/pred_precios_ablacion/Resultados/eval_TFT_LN.pkl",
         "P+C": "Multi-Modelos_TFT/h/LN/pred_prophet_clima/Resultados/eval_TFT_LN.pkl",
         "S+C": "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"}
BARRAS = ["ATACAMA", "CRUCERO", "P.MONTT", "TARAPACA"]


def main():
    ev = {m: pickle.load(open(r, "rb"))[0] for m, r in RUTAS.items()}
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    ini, fin = VENTANA_PRECIO_FALLA
    filas = []
    for barra in BARRAS:
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        pred = {}
        for m in RUTAS:
            d = ev[m][barra]["test"]["Precios"]["datos"]
            p = pd.Series(np.asarray(d["y_pred"]), index=pd.to_datetime(d["fechas"]))
            pred[m] = p[~p.index.duplicated()]
        idx = pred["Sin features"].index
        for m in ("P+C", "S+C"):
            idx = idx.intersection(pred[m].index)
        idx = idx[~((idx >= ini) & (idx <= fin))]
        y = serie.reindex(idx).values
        ok = ~np.isnan(y)
        idx, y = idx[ok], y[ok]
        err = {m: y - pred[m][idx].values for m in RUTAS}
        fila = {"Barra": barra}
        for m, e in err.items():
            fila[f"MAE {m}"] = np.abs(e).mean()
            fila[f"RMSE {m}"] = np.sqrt((e ** 2).mean())
            fila[f"R2 {m}"] = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
        for a, b in (("Sin features", "P+C"), ("Sin features", "S+C"), ("P+C", "S+C")):
            r = evaluar(np.abs(err[a]), np.abs(err[b]), idx)
            fila[f"DM {a} vs {b}"], fila[f"p {a} vs {b}"] = r["DM_24"], r["p_24"]
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv("Scripts/resultado_ablacion_tabla.csv", index=False)
    pd.set_option("display.width", 250)
    print("DM > 0 favorece al segundo modelo")
    print(df.round(3).T.to_string())


if __name__ == "__main__":
    main()
