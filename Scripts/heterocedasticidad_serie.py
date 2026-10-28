#!/usr/bin/env python3
"""
heterocedasticidad_serie.py - Tests de heterocedasticidad sobre la serie del costo marginal
(Seccion de analisis de estacionariedad de la Metodologia), complementarios a los que la
Seccion 4.6.3 aplica sobre los residuos del modelo.

  - Levene (centrado en la mediana): igualdad de varianza del costo marginal entre las 24 horas.
  - ARCH-LM de Engle con 24 rezagos: agrupamiento temporal de la volatilidad, sobre la desviacion
    del costo marginal respecto a la media de su hora del dia (para que el ciclo diario de la media
    no se confunda con el de la varianza).

Serie cruda horaria 2020-2026 de cada barra, sin la ventana de precio de falla.

Uso:
    python Scripts/heterocedasticidad_serie.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import levene
from statsmodels.stats.diagnostic import het_arch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA

RUTA_CRUDA = "Datos/h/2020-2026.csv"
NLAGS_ARCH = 24


def main():
    df = pd.read_csv(RUTA_CRUDA, sep=";")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    ini, fin = VENTANA_PRECIO_FALLA
    filas = []
    for barra in sorted(df["Barra"].unique()):
        s = df.loc[df["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        s = s[~((s.index >= ini) & (s.index <= fin))].dropna()
        hora = s.index.hour
        f_lev, p_lev = levene(*[s[hora == h].values for h in range(24)], center="median")
        desv = s - s.groupby(hora).transform("mean")
        lm, p_arch, _, _ = het_arch(desv.values, nlags=NLAGS_ARCH)
        sd_h = s.groupby(hora).std()
        filas.append({"Barra": barra, "Levene": f_lev, "p_Levene": p_lev, "ARCH_LM": lm, "p_ARCH": p_arch,
                      "SD_hora_min": sd_h.min(), "SD_hora_max": sd_h.max(), "n": len(s)})
    res = pd.DataFrame(filas)
    res.to_csv("Scripts/resultado_heterocedasticidad_serie.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
