#!/usr/bin/env python3
"""
dm_winkler_lwacp_aci.py - Test de Diebold-Mariano contra ACI de LW-ACP y de su variante robusta
(piloto_lw_escala_robusta.py, cuantil y piso elegidos en validacion) sobre el interval score de
Winkler hora a hora (S+C-LN, 8 barras, gamma=0.05), al 90% y al 95% de cobertura nominal.

Misma configuracion que cp_metricas_completas.py (calibracion en validacion, serie cruda, sin la
ventana de precio de falla) y las mismas dos versiones del DM que el resto del capitulo: horaria con
varianza Newey-West de 24 rezagos y diaria (Winkler promedio de cada dia, truncamiento de Andrews).
Holm-Bonferroni sobre las 8 barras de cada nivel, comparacion y version.

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
from piloto_lw_escala_robusta import lw_acp, elegir_piso


def winkler_horario(y, lo, up, alpha):
    return (up - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - up, 0)


def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    crudo = pd.read_csv("Datos/h/2020-2026.csv", sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])

    marcos = {}
    for barra in BARRAS:
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        marcos[barra] = (marco(datos, barra, "val", serie), marco(datos, barra, "test", serie))

    filas = []
    for alpha, nivel in ((0.10, "90%"), (0.05, "95%")):
        (cuantil, piso), _ = elegir_piso(marcos, alpha)
        print(f"[{nivel}] variante robusta: cuantil {cuantil}, piso {piso}", flush=True)
        for barra, (mv, mt) in marcos.items():
            fechas = pd.DatetimeIndex(mt["fecha"])
            res = {m: evaluar_metodo(m, mv, mt, alpha) for m in ("LW-ACP", "ACI")}
            res["LW-ACP robusto"] = lw_acp(mv, mt, alpha, piso, cuantil=cuantil)
            w = {m: winkler_horario(r["y_test"], r["lower"], r["upper"], alpha) for m, r in res.items()}
            for metodo in ("LW-ACP", "LW-ACP robusto"):
                r = evaluar(w[metodo], w["ACI"], fechas)
                filas.append({"Comparacion": f"{metodo} vs ACI", "Barra": barra, "Nivel": nivel,
                              "W_metodo": r["MAE_a"], "W_ACI": r["MAE_b"], "DM_24": r["DM_24"], "p_24": r["p_24"],
                              "lag_diario": r["lag_diario"], "DM_diario": r["DM_diario"], "p_diario": r["p_diario"]})

    df = pd.DataFrame(filas)
    for col in ("p_24", "p_diario"):
        df[f"holm_{col[2:]}"] = df.groupby(["Comparacion", "Nivel"])[col].transform(lambda p: holm(p.values))
    df.to_csv("Scripts/resultado_dm_winkler_lwacp_aci.csv", index=False)
    pd.set_option("display.width", 200)
    print("DM < 0 favorece al metodo frente a ACI")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
