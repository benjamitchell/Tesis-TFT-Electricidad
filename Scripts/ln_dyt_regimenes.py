#!/usr/bin/env python3
"""
ln_dyt_regimenes.py - Contrasta la hipotesis de que DyT maneja mejor los spikes del costo marginal
(Seccion 2.4.5): compara S+C-LN y S+C-DyT en prueba en dos regimenes, las horas de precio alto
(serie cruda >= su percentil 95) y las horas de salto brusco (las 5% con mayor cambio absoluto
respecto a la hora anterior), con Diebold-Mariano horario (Newey-West, 24 rezagos) y diario, y
Holm-Bonferroni sobre las 8 barras de cada regimen.

Uso:
    python Scripts/ln_dyt_regimenes.py
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
from dm_sensibilidad_rezagos import evaluar, holm, RUTA_CRUDA

RUTAS = {"LN": "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl",
         "DyT": "Multi-Modelos_TFT/h/DyT/pred_sol_clima/Resultados/eval_TFT_DyT.pkl"}


def main():
    ev = {arq: pickle.load(open(ruta, "rb"))[0] for arq, ruta in RUTAS.items()}
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    ini, fin = VENTANA_PRECIO_FALLA
    filas = []
    for barra in ev["LN"]:
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        pred = {}
        for arq in RUTAS:
            d = ev[arq][barra]["test"]["Precios"]["datos"]
            p = pd.Series(np.asarray(d["y_pred"]), index=pd.to_datetime(d["fechas"]))
            pred[arq] = p[~p.index.duplicated()]
        idx = pred["LN"].index.intersection(pred["DyT"].index)
        idx = idx[~((idx >= ini) & (idx <= fin))]
        y = serie.reindex(idx).values
        ok = ~np.isnan(y)
        idx, y = idx[ok], y[ok]
        e_ln, e_dyt = np.abs(y - pred["LN"][idx].values), np.abs(y - pred["DyT"][idx].values)
        salto = np.abs(np.diff(y))
        regimenes = {"Precio alto": y >= np.percentile(y, 95),
                     "Salto brusco": np.r_[False, salto >= np.percentile(salto, 95)]}
        for regimen, m in regimenes.items():
            r = evaluar(e_ln[m], e_dyt[m], idx[m])
            filas.append({"Regimen": regimen, "Barra": barra, "n": int(m.sum()), "MAE_LN": r["MAE_a"],
                          "MAE_DyT": r["MAE_b"], "DM_24": r["DM_24"], "p_24": r["p_24"], "p_diario": r["p_diario"]})
    df = pd.DataFrame(filas)
    for c in ("p_24", "p_diario"):
        df[f"holm_{c[2:]}"] = df.groupby("Regimen")[c].transform(lambda p: holm(p.values))
    df.to_csv("Scripts/resultado_ln_dyt_regimenes.csv", index=False)
    pd.set_option("display.width", 200)
    print("DM < 0 favorece a LN")
    print(df.round(3).to_string(index=False))
    print(df.groupby("Regimen")[["MAE_LN", "MAE_DyT"]].mean().round(2))


if __name__ == "__main__":
    main()
