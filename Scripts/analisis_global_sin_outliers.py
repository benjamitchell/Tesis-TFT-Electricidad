#!/usr/bin/env python3
"""
analisis_global_sin_outliers.py - Compara los cuatro modelos S+C-LN de la matriz
(outliers en train x granularidad), sobre el mismo test y la serie cruda como referencia:

    barra_con   por barra, train con recorte z>=4      (pred_sol_clima, el de la tesis)
    barra_sin   por barra, train sin recorte           (pred_sol_clima_piloto_sin_outliers)
    global_con  global (8 barras pooled), con recorte  (pred_sol_clima_global)
    global_sin  global (8 barras pooled), sin recorte  (pred_sol_clima_global_sin_outliers)

Por modelo: MAE/RMSE, errores extremos (p95, p99, top-20, maximo) y CP (Split CP y
LW-ACP gamma=0.05, alpha=0.10, calibrando en val). Por pares (global_sin contra cada uno de los otros
tres): Diebold-Mariano (MAE y MSE) y bootstrap por dias del delta de MAE en regimen p95.

Los dos modelos con recorte se evaluan con entradas crudas en val/test, igual que los dos sin
recorte (Scripts/inferencia_entradas_crudas.py, con y sin --global), para que la matriz solo
difiera en el entrenamiento.

Requiere haber corrido antes Scripts/analisis_piloto_outliers.py (cache de predicciones
por barra), Scripts/inferencia_entradas_crudas.py y Scripts/evaluar_modelo_global.py sobre
la carpeta global sin recorte.

Uso:
    python Scripts/analisis_global_sin_outliers.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Modulos.Comparativa import diebold_mariano
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA
from Modulos.Conformal_Prediction_Wrapper import (split_cp_from_calibration,
                                                  lw_acp_online_from_calibration)
from analisis_piloto_outliers import bootstrap_delta_regimen, metricas_cp, RUTA_CACHE, ALPHA, GAMMA
from evaluar_piloto_outliers import BARRAS
from inferencia_entradas_crudas import RUTA_BARRA, RUTA_GLOBAL

RUTA_SERIE_CRUDA = "Datos/h/2020-2026.csv"
CARPETA_GLOBAL_SIN = "Multi-Modelos_TFT/h/LN/pred_sol_clima_global_sin_outliers"
RUTA_GLOBAL_SIN = os.path.join(CARPETA_GLOBAL_SIN, "Resultados", "eval_TFT_GLOBAL.pkl")
CARPETA_SALIDA = os.path.join(CARPETA_GLOBAL_SIN, "Resultados")

MODELOS = ["barra_con", "barra_sin", "global_con", "global_sin"]
PARES = [("global_sin", "global_con"), ("global_sin", "barra_sin"), ("global_sin", "barra_con")]


def cargar_predicciones():
    with open(RUTA_CACHE, "rb") as f:
        cache = pickle.load(f)
    preds = {"barra_sin": cache["piloto"]}
    # los modelos con recorte, con entradas crudas en val/test (inferencia_entradas_crudas.py)
    for nombre, ruta in (("barra_con", RUTA_BARRA), ("global_con", RUTA_GLOBAL)):
        with open(ruta, "rb") as f:
            preds[nombre] = pickle.load(f)
    with open(RUTA_GLOBAL_SIN, "rb") as f:
        datos, _ = pickle.load(f)
    preds["global_sin"] = {}
    for barra in BARRAS:
        preds["global_sin"][barra] = {
            k: (pd.to_datetime(datos[barra][k]["Precios"]["datos"]["fechas"]),
                np.asarray(datos[barra][k]["Precios"]["datos"]["y_pred"]))
            for k in ("val", "test")}
    return preds


def marco_comun(preds, barra, conjunto, serie):
    """DataFrame con fecha, y (cruda) y una columna de prediccion por modelo, en la
    interseccion de fechas de los cuatro modelos, sin la ventana de precio de falla."""
    m = None
    for nombre in MODELOS:
        f, p = preds[nombre][barra][conjunto]
        d = pd.DataFrame({"fecha": f, nombre: p}).drop_duplicates("fecha")
        m = d if m is None else m.merge(d, on="fecha", how="inner")
    m = m.sort_values("fecha")
    m["y"] = serie.reindex(m["fecha"]).values
    ini, fin = VENTANA_PRECIO_FALLA
    m = m.dropna(subset=["y"])
    return m.loc[~((m["fecha"] >= ini) & (m["fecha"] <= fin))].reset_index(drop=True)


def main():
    preds = cargar_predicciones()
    df_crudo = pd.read_csv(RUTA_SERIE_CRUDA, sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])

    filas_mod, filas_cp, filas_par = [], [], []

    for barra in BARRAS:
        serie = df_crudo.loc[df_crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        mt = marco_comun(preds, barra, "test", serie)
        mv = marco_comun(preds, barra, "val", serie)
        y = mt["y"].values
        u95, u99 = np.percentile(y, 95), np.percentile(y, 99)
        m95, m99, m0 = y >= u95, y >= u99, y == 0

        for nombre in MODELOS:
            e = np.abs(y - mt[nombre].values)
            filas_mod.append({
                "Barra": barra, "Modelo": nombre, "n": len(y),
                "MAE": e.mean(), "RMSE": np.sqrt((e ** 2).mean()),
                "MAE_p95": e[m95].mean(), "MAE_p99": e[m99].mean(),
                "Top20": np.sort(e)[-20:].mean(), "Max": e.max(),
            })
            r_split = split_cp_from_calibration(mv["y"].values, mv[nombre].values,
                                                y, mt[nombre].values, alpha=ALPHA)
            r_lw = lw_acp_online_from_calibration(mv["y"].values, mv[nombre].values, mv["fecha"].values,
                                                  y, mt[nombre].values, mt["fecha"].values,
                                                  alpha=ALPHA, gamma=GAMMA)
            for metodo, r in (("Split CP", r_split), ("LW-ACP", r_lw)):
                fila = {"Barra": barra, "Modelo": nombre, "Metodo": metodo}
                fila.update(metricas_cp(y, r["lower"], r["upper"], m95, m0))
                filas_cp.append(fila)

        for nuevo, ref in PARES:
            dm_mae = diebold_mariano(y, mt[nuevo].values, mt[ref].values, h=1, loss="mae", max_lag=24)
            dm_mse = diebold_mariano(y, mt[nuevo].values, mt[ref].values, h=1, loss="mse", max_lag=24)
            par = pd.DataFrame({"fecha": mt["fecha"], "y": y, "p_act": mt[ref].values, "p_pil": mt[nuevo].values})
            obs95, lo95, hi95, p95 = bootstrap_delta_regimen(par, m95)
            filas_par.append({
                "Par": f"{nuevo} vs {ref}", "Barra": barra,
                "MAE_nuevo": np.abs(y - mt[nuevo].values).mean(), "MAE_ref": np.abs(y - mt[ref].values).mean(),
                "DM_MAE": dm_mae["DM"], "p_MAE": dm_mae["p_value"],
                "DM_MSE": dm_mse["DM"], "p_MSE": dm_mse["p_value"],
                "d_p95": obs95, "IC95_lo": lo95, "IC95_hi": hi95, "p_boot_p95": p95,
            })

    df_mod, df_cp, df_par = map(pd.DataFrame, (filas_mod, filas_cp, filas_par))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    print("\n=== A. MAE por barra (test, serie cruda, mismas fechas para los 4) ===")
    print(df_mod.pivot(index="Barra", columns="Modelo", values="MAE")[MODELOS].round(3).to_string())
    print("\nPromedio de las 8 barras por modelo:")
    print(df_mod.groupby("Modelo")[["MAE", "RMSE", "MAE_p95", "MAE_p99", "Top20", "Max"]]
          .mean().loc[MODELOS].round(3).to_string())
    print("\nMAE en regimen p95 por barra:")
    print(df_mod.pivot(index="Barra", columns="Modelo", values="MAE_p95")[MODELOS].round(2).to_string())
    print("\nError maximo por barra:")
    print(df_mod.pivot(index="Barra", columns="Modelo", values="Max")[MODELOS].round(1).to_string())

    print("\n=== B. Pares: nuevo (a) vs referencia (b); DM<0 y d_p95<0 favorecen a global_sin ===")
    for par, g in df_par.groupby("Par", sort=False):
        print(f"\n--- {par} ---")
        print(g.drop(columns="Par").round(4).to_string(index=False))
        gana = int(((g["DM_MAE"] < 0) & (g["p_MAE"] < 0.05)).sum())
        pierde = int(((g["DM_MAE"] > 0) & (g["p_MAE"] < 0.05)).sum())
        mej95 = int((g["d_p95"] < 0).sum())
        sig95 = int(((g["d_p95"] < 0) & (g["p_boot_p95"] < 0.05)).sum())
        print(f"MAE general (DM 5%): gana {gana}/8, pierde {pierde}/8, empate {8 - gana - pierde}/8 | "
              f"mejor en {int((g['DM_MAE'] < 0).sum())}/8 (signo p={binomtest(int((g['DM_MAE'] < 0).sum()), 8, 0.5).pvalue:.3f})")
        print(f"MAE p95: mejor en {mej95}/8 (bootstrap 5%: {sig95}/8, signo p={binomtest(mej95, 8, 0.5).pvalue:.3f})")

    print("\n=== C. Conformal Prediction (promedio de las 8 barras, alpha=0.10) ===")
    print(df_cp.groupby(["Metodo", "Modelo"])[["Cobertura", "Ancho", "Winkler", "Cob_p95", "Ancho_p95", "Cob_cero"]]
          .mean().reindex(MODELOS, level="Modelo").round(3).to_string())

    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    df_mod.to_csv(os.path.join(CARPETA_SALIDA, "matriz_modelos.csv"), index=False)
    df_par.to_csv(os.path.join(CARPETA_SALIDA, "pares_global_sin.csv"), index=False)
    df_cp.to_csv(os.path.join(CARPETA_SALIDA, "cp_matriz_modelos.csv"), index=False)
    print(f"\nGuardado en {CARPETA_SALIDA}/")


if __name__ == "__main__":
    main()
