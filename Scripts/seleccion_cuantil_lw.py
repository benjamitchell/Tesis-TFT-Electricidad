#!/usr/bin/env python3
"""
seleccion_cuantil_lw.py - Verifica si el cuantil al nivel nominal (P90 al 90%, P95 al 95%)
es la mejor escala horaria para LW-ACP robusto (Scripts/piloto_lw_escala_robusta.py), a 1 hora
y 8 barras.

Se busca en validacion, sin mirar la prueba, la combinacion de cuantil de la escala horaria y
piso absoluto con menor Winkler promedio: calibrando con la primera mitad de la validacion y
evaluando en la segunda, igual que la eleccion del piso. Luego se reporta en prueba el
Winkler de cada cuantil con su mejor piso en validacion, para ver si la eleccion de
validacion coincide con el nivel nominal.

Salida: Scripts/resultado_seleccion_cuantil_lw.csv
Uso:
    python Scripts/seleccion_cuantil_lw.py
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
from cp_metricas_completas import marco, winkler, RUTA_EVAL, BARRAS
from piloto_lw_escala_robusta import lw_acp, PISOS, RUTA_CRUDA

CUANTILES = {0.10: [0.5, 0.75, 0.9, 0.95], 0.05: [0.5, 0.75, 0.9, 0.95, 0.975]}


def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    marcos = {}
    for barra in BARRAS:
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        marcos[barra] = (marco(datos, barra, "val", serie), marco(datos, barra, "test", serie))

    filas = []
    for alpha, cuantiles in CUANTILES.items():
        nivel = f"{int(round(100 * (1 - alpha)))}%"
        for p in cuantiles:
            w_val = {}
            for piso in PISOS:
                ws = []
                for mv, _ in marcos.values():
                    mitad = len(mv) // 2
                    r = lw_acp(mv.iloc[:mitad].reset_index(drop=True), mv.iloc[mitad:].reset_index(drop=True),
                               alpha, piso, cuantil=p)
                    ws.append(winkler(r["y_test"], r["lower"], r["upper"], alpha))
                w_val[piso] = float(np.mean(ws))
            piso = min(w_val, key=w_val.get)
            w_test = []
            for mv, mt in marcos.values():
                r = lw_acp(mv, mt, alpha, piso, cuantil=p)
                w_test.append(winkler(r["y_test"], r["lower"], r["upper"], alpha))
            filas.append({"Nivel": nivel, "Cuantil": p, "Piso": piso, "Winkler_val": w_val[piso],
                          "Winkler_test": float(np.mean(w_test))})
            print(f"[{nivel}] cuantil {p}: piso {piso}, Winkler validacion {w_val[piso]:.2f}, "
                  f"prueba {np.mean(w_test):.2f}", flush=True)

    df = pd.DataFrame(filas)
    for nivel, d in df.groupby("Nivel"):
        mejor = d.sort_values("Winkler_val").iloc[0]
        print(f"\n{nivel}: cuantil elegido en validacion = {mejor.Cuantil} (piso {mejor.Piso})")
    df.to_csv("Scripts/resultado_seleccion_cuantil_lw.csv", index=False)
    print("\nGuardado: Scripts/resultado_seleccion_cuantil_lw.csv")


if __name__ == "__main__":
    main()
