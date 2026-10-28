#!/usr/bin/env python3
"""
dm_winkler_lwacp_aci.py - Test de Diebold-Mariano entre LW-ACP y ACI sobre el interval score de
Winkler hora a hora (S+C-LN, 8 barras, gamma=0.05), al 90% y al 95% de cobertura nominal.

Misma configuracion que cp_metricas_completas.py (calibracion en validacion, serie cruda, sin la
ventana de precio de falla) y las mismas dos versiones del DM que el resto del capitulo: horaria con
varianza Newey-West de 24 rezagos y diaria (Winkler promedio de cada dia, truncamiento de Andrews).
Holm-Bonferroni sobre las 8 barras de cada nivel y version.

Uso:
    python Scripts/dm_winkler_lwacp_aci.py
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
from cp_metricas_completas import RUTA_EVAL, BARRAS, marco, evaluar_metodo
from dm_sensibilidad_rezagos import evaluar, holm


def winkler_horario(y, lo, up, alpha):
    return (up - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - up, 0)


def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    crudo = pd.read_csv("Datos/h/2020-2026.csv", sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])

    filas = []
    for barra in BARRAS:
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        mv, mt = marco(datos, barra, "val", serie), marco(datos, barra, "test", serie)
        fechas = pd.DatetimeIndex(mt["fecha"])
        for alpha, nivel in ((0.10, "90%"), (0.05, "95%")):
            w = {}
            for metodo in ("LW-ACP", "ACI"):
                r = evaluar_metodo(metodo, mv, mt, alpha)
                w[metodo] = winkler_horario(r["y_test"], r["lower"], r["upper"], alpha)
            res = evaluar(w["LW-ACP"], w["ACI"], fechas)
            filas.append({"Barra": barra, "Nivel": nivel, "W_LWACP": res["MAE_a"], "W_ACI": res["MAE_b"],
                          "DM_24": res["DM_24"], "p_24": res["p_24"], "lag_diario": res["lag_diario"],
                          "DM_diario": res["DM_diario"], "p_diario": res["p_diario"]})

    df = pd.DataFrame(filas)
    for col in ("p_24", "p_diario"):
        df[f"holm_{col[2:]}"] = df.groupby("Nivel")[col].transform(lambda p: holm(p.values))
    df.to_csv("Scripts/resultado_dm_winkler_lwacp_aci.csv", index=False)
    pd.set_option("display.width", 200)
    print("DM < 0 favorece a LW-ACP")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
