#!/usr/bin/env python3
"""
analisis_piloto_outliers.py - Analisis estadistico del piloto de train sin recorte de
outliers vs. el modelo actual (train con recorte z>=4), S+C-LN, 8 barras:

  1. Diebold-Mariano (Newey-West) sobre el test, perdida MAE y MSE.
  2. Errores extremos: MAE en regimen p95/p99/cero, top-20 errores, maximo, y un
     bootstrap por bloques (dias) del delta de MAE en regimen alto.
  3. Conformal Prediction (Split CP y LW-ACP gamma=0.05, alpha=0.10, calibrando en val) con
     cobertura/ancho/Winkler globales y condicionales al regimen alto.

Referencia unica para ambos modelos: la serie cruda Datos/h/2020-2026.csv (misma que
recalcular_metricas_serie_cruda), excluyendo VENTANA_PRECIO_FALLA. Las predicciones se
cachean en Resultados/preds_piloto_vs_actual.pkl (la inferencia en CPU es lo lento).

Uso:
    python Scripts/analisis_piloto_outliers.py
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
from Modulos.Evaluacion_TFT import (extraer_todas_predicciones_modelo, VENTANA_PRECIO_FALLA)
from Modulos.Conformal_Prediction_Wrapper import (split_cp_from_calibration,
                                                  lw_acp_online_from_calibration)
from evaluar_piloto_outliers import EXPERIMENTO_PRECIOS, BARRAS, ORIGEN_ACTUAL, ORIGEN_PILOTO
from Modulos.TFT_Model import cargar_modelo_entrenado
from cp_metricas_completas import winkler

RUTA_SERIE_CRUDA = "Datos/h/2020-2026.csv"
CARPETA_SALIDA = os.path.join(ORIGEN_PILOTO, "Resultados")
RUTA_CACHE = os.path.join(CARPETA_SALIDA, "preds_piloto_vs_actual.pkl")
ALPHA = 0.10
GAMMA = 0.05
B_BOOT = 5000
rng = np.random.default_rng(0)


def generar_predicciones():
    """Devuelve {origen: {barra: {'val': (fechas, y_pred), 'test': (fechas, y_pred)}}}."""
    cache = {}
    for nombre, origen in (("actual", ORIGEN_ACTUAL), ("piloto", ORIGEN_PILOTO)):
        with open(os.path.join(origen, "dataloaders_precios.pkl"), "rb") as f:
            dls = pickle.load(f)
        with open(os.path.join(origen, "datasets_norm.pkl"), "rb") as f:
            dsn = pickle.load(f)
        with open(os.path.join(origen, "scalers.pkl"), "rb") as f:
            scl = pickle.load(f)
        cache[nombre] = {}
        for barra in BARRAS:
            print(f"[inferencia] {nombre} / {barra}", flush=True)
            modelo, _ = cargar_modelo_entrenado(barra, EXPERIMENTO_PRECIOS, origen, verbose=False)
            res = extraer_todas_predicciones_modelo(barra, modelo, dls[barra]["test"], dsn, scl)
            cache[nombre][barra] = {k: (pd.to_datetime(res[k]["fechas"]), np.asarray(res[k]["y_pred"]))
                                    for k in ("val", "test")}
        del dls, dsn, scl
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    with open(RUTA_CACHE, "wb") as f:
        pickle.dump(cache, f)
    return cache


def alinear(cache, barra, conjunto, serie):
    fa, pa = cache["actual"][barra][conjunto]
    fp, pp = cache["piloto"][barra][conjunto]
    da = pd.DataFrame({"fecha": fa, "p_act": pa}).drop_duplicates("fecha")
    dp = pd.DataFrame({"fecha": fp, "p_pil": pp}).drop_duplicates("fecha")
    m = da.merge(dp, on="fecha", how="inner").sort_values("fecha")
    m["y"] = serie.reindex(m["fecha"]).values
    ini, fin = VENTANA_PRECIO_FALLA
    m = m.dropna(subset=["y"])
    m = m.loc[~((m["fecha"] >= ini) & (m["fecha"] <= fin))].reset_index(drop=True)
    return m


def bootstrap_delta_regimen(m, mask):
    """Delta de MAE (piloto - actual) en las horas de `mask`, bootstrap por dias."""
    d = np.abs(m["y"] - m["p_pil"]).values - np.abs(m["y"] - m["p_act"]).values
    dias = m["fecha"].dt.normalize().values
    df = pd.DataFrame({"dia": dias, "d": np.where(mask, d, 0.0), "n": mask.astype(int)})
    g = df.groupby("dia").agg(s=("d", "sum"), n=("n", "sum"))
    s, n = g["s"].values, g["n"].values
    obs = s.sum() / n.sum()
    idx = rng.integers(0, len(g), size=(B_BOOT, len(g)))
    est = s[idx].sum(axis=1) / np.maximum(n[idx].sum(axis=1), 1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    p = min(1.0, 2 * min((est >= 0).mean(), (est <= 0).mean()))
    return obs, lo, hi, p




def metricas_cp(y, lo, up, mask_alto, mask_cero):
    dentro = (y >= lo) & (y <= up)
    out = {"Cobertura": dentro.mean(), "Ancho": np.mean(up - lo), "Winkler": winkler(y, lo, up, ALPHA)}
    out["Cob_p95"] = dentro[mask_alto].mean() if mask_alto.any() else np.nan
    out["Ancho_p95"] = np.mean((up - lo)[mask_alto]) if mask_alto.any() else np.nan
    out["Cob_cero"] = dentro[mask_cero].mean() if mask_cero.any() else np.nan
    return out


def main():
    if os.path.exists(RUTA_CACHE):
        print(f"Usando cache: {RUTA_CACHE}")
        with open(RUTA_CACHE, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = generar_predicciones()

    df_crudo = pd.read_csv(RUTA_SERIE_CRUDA, sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])

    filas_dm, filas_ext, filas_cp = [], [], []

    for barra in BARRAS:
        serie = df_crudo.loc[df_crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        mt = alinear(cache, barra, "test", serie)
        mv = alinear(cache, barra, "val", serie)
        y = mt["y"].values
        e_act = np.abs(y - mt["p_act"].values)
        e_pil = np.abs(y - mt["p_pil"].values)

        # 1. Diebold-Mariano (a = piloto: DM<0 favorece al piloto)
        dm_mae = diebold_mariano(y, mt["p_pil"].values, mt["p_act"].values, h=1, loss="mae", max_lag=24)
        dm_mse = diebold_mariano(y, mt["p_pil"].values, mt["p_act"].values, h=1, loss="mse", max_lag=24)
        filas_dm.append({
            "Barra": barra, "n": len(mt),
            "MAE_actual": e_act.mean(), "MAE_piloto": e_pil.mean(),
            "DM_MAE": dm_mae["DM"], "p_MAE": dm_mae["p_value"],
            "RMSE_actual": np.sqrt((e_act ** 2).mean()), "RMSE_piloto": np.sqrt((e_pil ** 2).mean()),
            "DM_MSE": dm_mse["DM"], "p_MSE": dm_mse["p_value"],
        })

        # 2. Errores extremos
        u95, u99 = np.percentile(y, 95), np.percentile(y, 99)
        m95, m99, m0 = y >= u95, y >= u99, y == 0
        obs95, lo95, hi95, p95 = bootstrap_delta_regimen(mt, m95)
        obs99, lo99, hi99, p99 = bootstrap_delta_regimen(mt, m99)
        top20_act = np.sort(e_act)[-20:].mean()
        top20_pil = np.sort(e_pil)[-20:].mean()
        filas_ext.append({
            "Barra": barra,
            "MAE_p95_act": e_act[m95].mean(), "MAE_p95_pil": e_pil[m95].mean(),
            "d_p95": obs95, "IC95_lo": lo95, "IC95_hi": hi95, "p_boot_p95": p95,
            "MAE_p99_act": e_act[m99].mean(), "MAE_p99_pil": e_pil[m99].mean(),
            "d_p99": obs99, "p_boot_p99": p99,
            "MAE_cero_act": e_act[m0].mean() if m0.any() else np.nan,
            "MAE_cero_pil": e_pil[m0].mean() if m0.any() else np.nan,
            "Top20_act": top20_act, "Top20_pil": top20_pil,
            "Max_act": e_act.max(), "Max_pil": e_pil.max(),
            "N>100_act": int((e_act > 100).sum()), "N>100_pil": int((e_pil > 100).sum()),
        })

        # 3. Conformal Prediction (calibracion en val, serie cruda)
        for nombre, col in (("actual", "p_act"), ("piloto", "p_pil")):
            r_split = split_cp_from_calibration(mv["y"].values, mv[col].values,
                                                y, mt[col].values, alpha=ALPHA)
            r_lw = lw_acp_online_from_calibration(mv["y"].values, mv[col].values, mv["fecha"].values,
                                                  y, mt[col].values, mt["fecha"].values,
                                                  alpha=ALPHA, gamma=GAMMA)
            for metodo, r in (("Split CP", r_split), ("LW-ACP", r_lw)):
                fila = {"Barra": barra, "Modelo": nombre, "Metodo": metodo}
                fila.update(metricas_cp(y, r["lower"], r["upper"], m95, m0))
                filas_cp.append(fila)

    df_dm = pd.DataFrame(filas_dm)
    df_ext = pd.DataFrame(filas_ext)
    df_cp = pd.DataFrame(filas_cp)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    print("\n=== 1. Diebold-Mariano, piloto (a) vs actual (b); DM<0 favorece al piloto ===")
    print(df_dm.round(4).to_string(index=False))
    g = df_dm["DM_MAE"] < 0
    for etiqueta, col in (("MAE", "p_MAE"), ("MSE", "p_MSE")):
        gana = int(((df_dm[f"DM_{etiqueta}"] < 0) & (df_dm[col] < 0.05)).sum())
        pierde = int(((df_dm[f"DM_{etiqueta}"] > 0) & (df_dm[col] < 0.05)).sum())
        print(f"Perdida {etiqueta}: piloto gana significativamente en {gana}/8, "
              f"pierde en {pierde}/8, empate en {8 - gana - pierde}/8 (5%).")
    st = binomtest(int(g.sum()), 8, 0.5).pvalue
    print(f"Test de signo sobre MAE general: piloto mejor en {int(g.sum())}/8, p={st:.3f}")

    print("\n=== 2. Errores extremos (test, serie cruda) ===")
    print(df_ext.round(3).to_string(index=False))
    mej95 = int((df_ext["d_p95"] < 0).sum())
    sig95 = int(((df_ext["d_p95"] < 0) & (df_ext["p_boot_p95"] < 0.05)).sum())
    mej99 = int((df_ext["d_p99"] < 0).sum())
    print(f"Delta MAE p95 < 0 en {mej95}/8 barras (significativo bootstrap 5% en {sig95}/8); "
          f"sign test p={binomtest(mej95, 8, 0.5).pvalue:.3f}")
    print(f"Delta MAE p99 < 0 en {mej99}/8 barras; sign test p={binomtest(mej99, 8, 0.5).pvalue:.3f}")
    print(f"Top-20 errores: piloto menor en {int((df_ext['Top20_pil'] < df_ext['Top20_act']).sum())}/8; "
          f"Maximo: piloto menor en {int((df_ext['Max_pil'] < df_ext['Max_act']).sum())}/8")

    print(f"\n=== 3. Conformal Prediction (alpha={ALPHA}; calibracion en val, serie cruda) ===")
    print(df_cp.round(3).to_string(index=False))
    print("\nPromedio de las 8 barras:")
    print(df_cp.groupby(["Metodo", "Modelo"])[["Cobertura", "Ancho", "Winkler", "Cob_p95", "Ancho_p95", "Cob_cero"]]
          .mean().round(3).to_string())

    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    df_dm.to_csv(os.path.join(CARPETA_SALIDA, "dm_piloto_vs_actual.csv"), index=False)
    df_ext.to_csv(os.path.join(CARPETA_SALIDA, "extremos_piloto_vs_actual.csv"), index=False)
    df_cp.to_csv(os.path.join(CARPETA_SALIDA, "cp_piloto_vs_actual.csv"), index=False)
    print(f"\nGuardado en {CARPETA_SALIDA}/")


if __name__ == "__main__":
    main()
