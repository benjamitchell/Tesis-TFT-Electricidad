#!/usr/bin/env python3
"""
h24_ln_vs_dyt.py - Test de Diebold-Mariano (varianza Newey-West) entre los modelos
day-ahead S+C con LayerNorm y con DyT, conjunto de prueba, contra la serie cruda,
excluyendo VENTANA_PRECIO_FALLA, para las 5 barras con modelo h24.

Dos niveles:
  1. Por barra, sobre el error absoluto promedio de los 24 pasos de cada pronostico
     emitido (un valor por hora de emision). Pronosticos consecutivos comparten 23
     horas, asi que se usa h=24 (truncamiento de Newey-West en 25).
  2. Por barra y paso, sobre el error absoluto de ese paso, con h igual al paso.
Holm-Bonferroni sobre las 5 barras en cada nivel (y en cada paso).

DM < 0 favorece a LN (primer argumento). Lee los caches de
Scripts/extraer_predicciones_h24.py.

Uso:
    python Scripts/h24_ln_vs_dyt.py
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
from Modulos.Comparativa import diebold_mariano_desde_errores
from test_heterocedasticidad import holm

CACHE = "Multi-Modelos_TFT/h/{arq}/pred_sol_clima_h24/Resultados/predicciones_por_paso.pkl"
RUTA_CRUDA = "Datos/h/2020-2026.csv"


def errores_por_paso(preds_barra, serie):
    """DataFrame con indice = hora de emision del pronostico y una columna por paso."""
    ini, fin = VENTANA_PRECIO_FALLA
    cols = {}
    for h, d in preds_barra.items():
        f = pd.to_datetime(d["test"]["fechas"])
        e = pd.Series(np.abs(serie.reindex(f).values - np.asarray(d["test"]["y_pred"])), index=f)
        e = e[~e.index.duplicated()]
        e[(e.index >= ini) & (e.index <= fin)] = np.nan
        e.index = e.index - pd.Timedelta(hours=h)
        cols[h] = e
    return pd.DataFrame(cols)


def main():
    preds = {}
    for arq in ("LN", "DyT"):
        with open(CACHE.format(arq=arq), "rb") as f:
            preds[arq] = pickle.load(f)
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])

    filas_dia, filas_paso = [], []
    for barra in preds["LN"]:
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        e_ln = errores_por_paso(preds["LN"][barra], serie)
        e_dyt = errores_por_paso(preds["DyT"][barra], serie)
        comunes = e_ln.dropna().index.intersection(e_dyt.dropna().index)
        m_ln, m_dyt = e_ln.loc[comunes].mean(axis=1), e_dyt.loc[comunes].mean(axis=1)
        r = diebold_mariano_desde_errores(m_ln.values, m_dyt.values, h=24)
        filas_dia.append({"Barra": barra, "n": len(comunes), "MAE_LN": m_ln.mean(), "MAE_DyT": m_dyt.mean(),
                          "DM": r["DM"], "p": r["p_value"]})
        for h in e_ln.columns:
            a, b = e_ln[h], e_dyt[h]
            idx = a.dropna().index.intersection(b.dropna().index)
            r = diebold_mariano_desde_errores(a.loc[idx].values, b.loc[idx].values, h=h)
            filas_paso.append({"Barra": barra, "h": h, "MAE_LN": a.loc[idx].mean(), "MAE_DyT": b.loc[idx].mean(),
                               "DM": r["DM"], "p": r["p_value"]})
        print(f"[{barra}] listo", flush=True)

    dia = pd.DataFrame(filas_dia)
    sig = holm(dict(zip(dia.index, dia["p"])))
    dia["sig_holm"] = dia.index.isin(sig)
    dia["favorece"] = np.where(dia["DM"] < 0, "LN", "DyT")
    pd.set_option("display.width", 200)
    print("\n=== Por barra: error promedio de los 24 pasos de cada pronostico (DM<0 favorece a LN) ===")
    print(dia.round(4).to_string(index=False))

    paso = pd.DataFrame(filas_paso)
    paso["sig_holm"] = False
    for h, g in paso.groupby("h"):
        paso.loc[list(holm(dict(zip(g.index, g["p"])))), "sig_holm"] = True
    paso["favorece"] = np.where(paso["DM"] < 0, "LN", "DyT")
    print("\n=== Por paso: combinaciones barra-paso significativas tras Holm, segun a quien favorecen (de 120) ===")
    print(paso[paso["sig_holm"]]["favorece"].value_counts().to_string())
    print("\nPor barra (pasos significativos tras Holm a favor de cada arquitectura, y pasos con menor MAE nominal):")
    resumen = paso.groupby("Barra").apply(lambda g: pd.Series({
        "sig_LN": int(((g["favorece"] == "LN") & g["sig_holm"]).sum()),
        "sig_DyT": int(((g["favorece"] == "DyT") & g["sig_holm"]).sum()),
        "LN_menor_MAE": int((g["MAE_LN"] < g["MAE_DyT"]).sum()),
    }))
    print(resumen.to_string())

    dia.to_csv("Scripts/resultado_h24_ln_vs_dyt_dia.csv", index=False)
    paso.to_csv("Scripts/resultado_h24_ln_vs_dyt_paso.csv", index=False)
    print("\nGuardado: Scripts/resultado_h24_ln_vs_dyt_dia.csv y _paso.csv")


if __name__ == "__main__":
    main()
