#!/usr/bin/env python3
"""
seleccion_cuantil_lw_h24.py - Igual que Scripts/seleccion_cuantil_lw.py, pero sobre el modelo
S+C-LN a 24 horas (8 barras x 24 pasos), con la adaptacion online por hora del dia de
Scripts/h24_cp_metodos.py, al 90% de cobertura nominal.

Para cada cuantil de la escala horaria se elige el piso en validacion (calibrando con la
primera mitad y midiendo el Winkler en la segunda, promedio de las 192 combinaciones barra-
paso) y se reporta el Winkler en prueba, junto al de LW-ACP en su version principal.

Salida: Scripts/resultado_seleccion_cuantil_lw_h24.csv
Uso:
    python Scripts/seleccion_cuantil_lw_h24.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_h24 import cargar_preds_sc
from cp_metricas_completas import winkler
from h24_cp_metodos import marco
from piloto_lw_escala_robusta import lw_acp, PISOS, CUANTILES

RUTA_CRUDA = "Datos/h/2020-2026.csv"
ALPHA = 0.10


def main():
    preds = cargar_preds_sc("LN")
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    marcos = []
    for barra, por_paso in preds.items():
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        marcos += [(marco(d["val"], serie), marco(d["test"], serie)) for d in por_paso.values()]

    w_base = np.mean([winkler(*(lambda r: (r["y_test"], r["lower"], r["upper"]))(lw_acp(mv, mt, ALPHA, por_hora=True)), ALPHA)
                      for mv, mt in marcos])
    print(f"LW-ACP version principal (prueba): {w_base:.2f}", flush=True)

    filas = []
    for p in CUANTILES:
        w_val = {}
        for piso in PISOS:
            ws = []
            for mv, _ in marcos:
                mitad = len(mv) // 2
                r = lw_acp(mv.iloc[:mitad].reset_index(drop=True), mv.iloc[mitad:].reset_index(drop=True),
                           ALPHA, piso, por_hora=True, cuantil=p)
                ws.append(winkler(r["y_test"], r["lower"], r["upper"], ALPHA))
            w_val[piso] = float(np.mean(ws))
        piso = min(w_val, key=w_val.get)
        w_test = np.mean([winkler(*(lambda r: (r["y_test"], r["lower"], r["upper"]))(
            lw_acp(mv, mt, ALPHA, piso, por_hora=True, cuantil=p)), ALPHA) for mv, mt in marcos])
        filas.append({"Cuantil": p, "Piso": piso, "Winkler_val": w_val[piso], "Winkler_test": float(w_test)})
        print(f"cuantil {p}: piso {piso}, Winkler validacion {w_val[piso]:.2f}, prueba {w_test:.2f}", flush=True)

    df = pd.DataFrame(filas)
    mejor = df.sort_values("Winkler_val").iloc[0]
    print(f"\nElegido en validacion: cuantil {mejor.Cuantil}, piso {mejor.Piso}; prueba {mejor.Winkler_test:.2f} "
          f"(LW-ACP principal {w_base:.2f})")
    df.to_csv("Scripts/resultado_seleccion_cuantil_lw_h24.csv", index=False)
    print("Guardado: Scripts/resultado_seleccion_cuantil_lw_h24.csv")


if __name__ == "__main__":
    main()
