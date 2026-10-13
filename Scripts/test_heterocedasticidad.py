#!/usr/bin/env python3
"""
test_heterocedasticidad.py - Verificacion empirica formal de heterocedasticidad en los
residuos de S+C-LN, las ocho barras del SEN, dos pruebas complementarias:

  1. Test de Levene (homogeneidad de varianza entre grupos), agrupando los residuos
     de VALIDACION por hora del dia (24 grupos) -- valida directamente la premisa de
     LW Split CP / LW-ACP (heterocedasticidad horaria explotada via sigma_h).
  2. Test ARCH-LM de Engle (autocorrelacion de los residuos al cuadrado en el tiempo,
     nlags=24 ~ un dia), sobre la serie de residuos de VALIDACION en orden cronologico
     -- el mismo test usado en la literatura internacional (Benhmad & Percebois 2018,
     Fleten et al. 2013, Bello-Rodriguez & Beltran-Ahumada 2010) para precios
     electricos, lo que hace el resultado directamente comparable; valida la premisa
     de la adaptacion online de alpha_t en ACI/LW-ACP (agrupamiento temporal de la
     volatilidad, no solo por hora del dia).

Residuos = y_real (serie cruda) - y_pred (S+C-LN), excluyendo VENTANA_PRECIO_FALLA.
Se usa el conjunto de VALIDACION porque es el mismo conjunto de calibracion que usa
el pipeline de CP para estimar sigma_h (Modulos/Conformal_Prediction_Wrapper.py).

Uso:
    python Scripts/test_heterocedasticidad.py
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA

RUTA_EVAL = "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"
RUTA_CRUDA = "Datos/h/2020-2026.csv"
BARRAS = ["ATACAMA", "CARDONES", "CHARRUA", "CRUCERO", "P.AZUCAR", "P.MONTT", "QUILLOTA", "TARAPACA"]
NLAGS_ARCH = 24


def holm(pvals, alpha=0.05):
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    sig = set()
    for i, (k, p) in enumerate(items):
        if p < alpha / (m - i):
            sig.add(k)
        else:
            break
    return sig


def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    df_crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])
    ini_falla, fin_falla = VENTANA_PRECIO_FALLA

    filas = []
    for barra in BARRAS:
        serie = df_crudo.loc[df_crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        d = datos[barra]["val"]["Precios"]["datos"]
        m = pd.DataFrame({"fecha": pd.to_datetime(d["fechas"]), "y_pred": np.asarray(d["y_pred"])})
        m = m.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
        m["y_real"] = serie.reindex(m["fecha"]).values
        m = m.dropna(subset=["y_real"])
        m = m.loc[~((m["fecha"] >= ini_falla) & (m["fecha"] <= fin_falla))].reset_index(drop=True)
        m["residuo"] = m["y_real"] - m["y_pred"]
        m["hora"] = m["fecha"].dt.hour

        # 1. Levene por hora del dia
        grupos = [m.loc[m["hora"] == h, "residuo"].values for h in range(24)]
        grupos = [g for g in grupos if len(g) >= 10]
        stat_lev, p_lev = levene(*grupos, center="median")
        sd_por_hora = m.groupby("hora")["residuo"].std()

        # 2. ARCH-LM (Engle) sobre la serie de residuos en orden cronologico
        stat_arch, p_arch, _, _ = het_arch(m["residuo"].values, nlags=NLAGS_ARCH)

        filas.append({
            "Barra": barra, "n": len(m),
            "Levene_stat": stat_lev, "Levene_p": p_lev,
            "SD_hora_min": sd_por_hora.min(), "SD_hora_max": sd_por_hora.max(),
            "ARCH_stat": stat_arch, "ARCH_p": p_arch,
        })
        print(f"[{barra}] Levene F={stat_lev:.2f} p={p_lev:.2e} | "
              f"SD por hora: [{sd_por_hora.min():.2f}, {sd_por_hora.max():.2f}] | "
              f"ARCH-LM={stat_arch:.1f} p={p_arch:.2e}")

    df = pd.DataFrame(filas)

    p_lev = {r["Barra"]: r["Levene_p"] for _, r in df.iterrows()}
    p_arch = {r["Barra"]: r["ARCH_p"] for _, r in df.iterrows()}
    hb_lev = holm(p_lev)
    hb_arch = holm(p_arch)

    print("\n=== Resumen ===")
    print(f"Levene (heterocedasticidad por hora): {sum(p < 0.05 for p in p_lev.values())}/8 significativos sin corregir, "
          f"{len(hb_lev)}/8 tras Holm-Bonferroni")
    print(f"ARCH-LM (agrupamiento temporal de volatilidad): {sum(p < 0.05 for p in p_arch.values())}/8 significativos sin corregir, "
          f"{len(hb_arch)}/8 tras Holm-Bonferroni")

    os.makedirs("Resultados", exist_ok=True)
    ruta_out = "Scripts/resultado_heterocedasticidad.csv"
    df.to_csv(ruta_out, index=False)
    print(f"\nGuardado: {ruta_out}")


if __name__ == "__main__":
    main()
