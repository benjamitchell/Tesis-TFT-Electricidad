#!/usr/bin/env python3
"""
h24_heterocedasticidad.py - Repite la verificacion de heterocedasticidad de la tesis
(Scripts/test_heterocedasticidad.py) sobre el modelo day-ahead S+C-LN, para cada paso
del horizonte (h=1..24) y las 8 barras con modelo h24:

  1. Levene (centrado en la mediana) entre los 24 grupos de hora del dia de la hora
     pronosticada.
  2. ARCH-LM de Engle (nlags=24) sobre la serie de residuos del paso en orden
     cronologico.

Residuos de VALIDACION contra la serie cruda, excluyendo VENTANA_PRECIO_FALLA, igual que
en h=1. Holm-Bonferroni sobre las 8 barras de cada paso. Lee el cache de
Scripts/extraer_predicciones_h24.py.

Uso:
    python Scripts/h24_heterocedasticidad.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import levene
from statsmodels.stats.diagnostic import het_arch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA
from cache_h24 import cargar_preds_sc
from test_heterocedasticidad import holm

RUTA_CRUDA = "Datos/h/2020-2026.csv"
NLAGS_ARCH = 24


def main():
    preds = cargar_preds_sc("LN")
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    ini, fin = VENTANA_PRECIO_FALLA

    filas = []
    for barra, por_paso in preds.items():
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        for h, d in por_paso.items():
            m = pd.DataFrame({"fecha": pd.to_datetime(d["val"]["fechas"]), "y_pred": np.asarray(d["val"]["y_pred"])})
            m = m.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
            m["y_real"] = serie.reindex(m["fecha"]).values
            m = m.dropna(subset=["y_real"])
            m = m.loc[~((m["fecha"] >= ini) & (m["fecha"] <= fin))].reset_index(drop=True)
            m["residuo"] = m["y_real"] - m["y_pred"]
            hora = m["fecha"].dt.hour

            grupos = [g for g in (m.loc[hora == k, "residuo"].values for k in range(24)) if len(g) >= 10]
            f_lev, p_lev = levene(*grupos, center="median")
            sd = m.groupby(hora)["residuo"].std()
            lm, p_arch, _, _ = het_arch(m["residuo"].values, nlags=NLAGS_ARCH)
            filas.append({"Barra": barra, "h": h, "n": len(m), "Levene_F": f_lev, "Levene_p": p_lev,
                          "SD_min": sd.min(), "SD_max": sd.max(), "ARCH": lm, "ARCH_p": p_arch})
        print(f"[{barra}] listo")

    df = pd.DataFrame(filas)
    for prueba in ("Levene", "ARCH"):
        col = "Levene_p" if prueba == "Levene" else "ARCH_p"
        df[f"{prueba}_sig_holm"] = False
        for h, g in df.groupby("h"):
            sig = holm(dict(zip(g.index, g[col])))
            df.loc[list(sig), f"{prueba}_sig_holm"] = True

    pd.set_option("display.width", 220)
    resumen = df.groupby("Barra").agg(
        Levene_F_min=("Levene_F", "min"), Levene_F_max=("Levene_F", "max"), Levene_p_max=("Levene_p", "max"),
        Levene_sig=("Levene_sig_holm", "sum"),
        SD_min=("SD_min", "min"), SD_max=("SD_max", "max"),
        ARCH_min=("ARCH", "min"), ARCH_max=("ARCH", "max"), ARCH_p_max=("ARCH_p", "max"),
        ARCH_sig=("ARCH_sig_holm", "sum"))
    print("\n=== Por barra, sobre los 24 pasos (sig = pasos significativos tras Holm, de 24) ===")
    print(resumen.round(4).to_string())
    print("\n=== Pasos representativos ===")
    print(df[df["h"].isin([1, 6, 12, 18, 24])].round(4).to_string(index=False))
    print(f"\nTotal: Levene significativo en {int(df['Levene_sig_holm'].sum())}/{len(df)}, "
          f"ARCH-LM en {int(df['ARCH_sig_holm'].sum())}/{len(df)} combinaciones barra-paso (Holm por paso).")

    df.to_csv("Scripts/resultado_h24_heterocedasticidad.csv", index=False)
    print("Guardado: Scripts/resultado_h24_heterocedasticidad.csv")


if __name__ == "__main__":
    main()
