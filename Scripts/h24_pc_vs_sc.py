#!/usr/bin/env python3
"""
h24_pc_vs_sc.py - Compara las estrategias P+C y S+C a horizonte day-ahead (h=24), LN,
en las 8 barras con modelo h24, replicando el protocolo de la tesis a h=1
(Modulos/Stacking_Optimization_TFT.stacking_optimization) en cada paso del horizonte:

  - Seis estrategias de P+C: Prophet solo, TFT precios solo, Prophet + TFT precios,
    Prophet + TFT residuos (suma directa y con pesos) y (Prophet + residuos) + TFT
    precios. Pesos por grid search de 41 puntos sobre ENTRENAMIENTO, contra la serie
    usada en el entrenamiento (datasets_norm), igual que a h=1.
  - Seleccion de la mejor estrategia por barra en VALIDACION (MAE promedio de los 24
    pasos contra la serie cruda), sin mirar prueba.
  - Comparacion en PRUEBA contra S+C h24 (cache de Scripts/extraer_predicciones_h24.py),
    contra la serie cruda, excluyendo VENTANA_PRECIO_FALLA: MAE promedio de los 24 pasos
    y Diebold-Mariano sobre el error promedio de los 24 pasos de cada pronostico emitido
    (h=24, Newey-West truncado en 25), con Holm sobre las 8 barras.

Uso:
    python Scripts/h24_pc_vs_sc.py
"""
import os
import sys
import gc
import json
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.TFT_Model import TFTBridge
from Modulos.Evaluacion_TFT import extraer_todas_predicciones_modelo, inferir_raw_modelo, VENTANA_PRECIO_FALLA
from Modulos.Stacking_Optimization_TFT import grid_search_modelos
from Modulos.Comparativa import diebold_mariano_desde_errores
from test_heterocedasticidad import holm
from cache_h24 import cargar_preds_sc, cargar_preds_pc

CARPETA_NORM = "Multi-Modelos_TFT/h/LN/pred_prophet_clima"
RUTA_CRUDA = "Datos/h/2020-2026.csv"
# (carpeta de la tanda de entrenamiento, barras)
TANDAS = [("Multi-Modelos_TFT/h/LN/pred_prophet_clima_h24", ["ATACAMA", "CHARRUA", "P.AZUCAR", "P.MONTT", "TARAPACA"]),
          ("Multi-Modelos_TFT/h/LN/pred_prophet_clima_h24_resto", ["CARDONES", "CRUCERO", "QUILLOTA"])]
BARRAS = [b for _, bs in TANDAS for b in bs]
N_HORIZONTES = 24
ESTRATEGIAS = ["Prophet solo", "TFT precios solo", "Prophet + TFT precios",
               "Prophet + residuos (directo)", "Prophet + residuos (opt)", "(Prophet + residuos) + TFT precios"]


def cargar_modelo(carpeta, experimento, barra):
    ruta = os.path.join(carpeta, experimento, barra, f"{experimento}_{barra}_config.json")
    with open(ruta, encoding="utf-8") as f:
        ckpt = json.load(f)["model_path"]
    if not os.path.exists(ckpt):
        ckpt = os.path.join(carpeta, experimento, barra, os.path.basename(ckpt))
    _orig_zeros = torch.zeros

    def _safe_zeros(*a, **kw):
        d = kw.get("device")
        if d is not None and "cuda" in str(d):
            kw["device"] = "cpu"
        return _orig_zeros(*a, **kw)

    torch.zeros = _safe_zeros
    try:
        modelo = TFTBridge.load_from_checkpoint(ckpt, map_location=lambda storage, loc: storage, weights_only=False)
    finally:
        torch.zeros = _orig_zeros
    modelo.eval()
    for m in modelo.modules():
        if hasattr(m, "_device"):
            m._device = torch.device("cpu")
    return modelo


def extraer(carpeta, barras):
    """Predicciones por paso de precios y residuos de P+C h24, en los tres conjuntos.
    Cache en <carpeta>/Resultados/predicciones_por_paso.pkl."""
    cache = os.path.join(carpeta, "Resultados", "predicciones_por_paso.pkl")
    if os.path.exists(cache):
        return
    with open(os.path.join(carpeta, "dataloaders_precios.pkl"), "rb") as f:
        dl_precios = pickle.load(f)
    with open(os.path.join(carpeta, "dataloaders_residuos.pkl"), "rb") as f:
        dl_residuos = pickle.load(f)
    with open(os.path.join(CARPETA_NORM, "datasets_norm.pkl"), "rb") as f:
        datasets_norm = pickle.load(f)
    with open(os.path.join(carpeta, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    out = {}
    for barra in barras:
        out[barra] = {}
        for tipo, exp, dls, clave in (("precios", "Multi-TFT_Precios", dl_precios, "y_real"),
                                      ("residuos", "Multi-TFT_Residuos", dl_residuos, "residuo")):
            modelo = cargar_modelo(carpeta, exp, barra)
            raw = inferir_raw_modelo(modelo, dls[barra]["test"])
            for h in range(1, N_HORIZONTES + 1):
                r = extraer_todas_predicciones_modelo(barra, modelo, dls[barra]["test"], datasets_norm, scalers,
                                                      target_key=clave, scaler_key=clave, horizonte=h, raw=raw)
                for c in ("train", "val", "test"):
                    out[barra].setdefault(h, {}).setdefault(c, {})[tipo] = pd.Series(
                        np.asarray(r[c]["y_pred"]), index=pd.to_datetime(r[c]["fechas"]))
            print(f"[{barra}] {tipo} listo", flush=True)
            del raw, modelo
            gc.collect()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f)


def prophet_y_objetivo(datasets_norm, scalers, barra):
    """Prediccion de Prophet (yhat) y precio de entrenamiento (serie del pipeline), por fecha."""
    df = pd.concat([datasets_norm[barra][c] for c in ("train", "val", "test")], ignore_index=True)
    idx = pd.to_datetime(df["ds"])
    st = scalers[barra]["scalers_target"]
    yhat = st["yhat"].inverse_transform(df[["yhat"]].values).ravel()
    y = st["y_real"].inverse_transform(df[["y_real"]].values).ravel()
    return pd.Series(yhat, index=idx).groupby(level=0).first(), pd.Series(y, index=idx).groupby(level=0).first()


def estrategias_paso(d, prophet, y_pipe):
    """Predicciones de las seis estrategias en train/val/test para un paso del horizonte."""
    marcos = {}
    for c in ("train", "val", "test"):
        m = pd.DataFrame({"precios": d[c]["precios"], "residuos": d[c]["residuos"]}).dropna()
        m["prophet"] = prophet.reindex(m.index).values
        m["y_pipe"] = y_pipe.reindex(m.index).values
        marcos[c] = m.dropna()
    tr = marcos["train"]
    w_pt = grid_search_modelos(tr["y_pipe"].values, tr["prophet"].values, tr["precios"].values, resolution=41)["w1"]
    w_pr = grid_search_modelos(tr["y_pipe"].values, tr["prophet"].values, tr["residuos"].values, resolution=41)["w1"]
    w_prt = grid_search_modelos(tr["y_pipe"].values, (tr["prophet"] + tr["residuos"]).values, tr["precios"].values,
                                resolution=41)["w1"]
    preds = {}
    for c in ("val", "test"):
        m = marcos[c]
        preds[c] = pd.DataFrame({
            "Prophet solo": m["prophet"],
            "TFT precios solo": m["precios"],
            "Prophet + TFT precios": w_pt * m["prophet"] + (1 - w_pt) * m["precios"],
            "Prophet + residuos (directo)": m["prophet"] + m["residuos"],
            "Prophet + residuos (opt)": w_pr * m["prophet"] + (1 - w_pr) * m["residuos"],
            "(Prophet + residuos) + TFT precios": w_prt * (m["prophet"] + m["residuos"]) + (1 - w_prt) * m["precios"],
        })
    return preds


def main():
    scalers = {}
    for carpeta, barras in TANDAS:
        extraer(carpeta, barras)
        with open(os.path.join(carpeta, "scalers.pkl"), "rb") as f:
            scalers.update({b: v for b, v in pickle.load(f).items() if b in barras})
    pc, sc = cargar_preds_pc(), cargar_preds_sc("LN")
    with open(os.path.join(CARPETA_NORM, "datasets_norm.pkl"), "rb") as f:
        datasets_norm = pickle.load(f)
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    ini, fin = VENTANA_PRECIO_FALLA

    def error_abs(serie_pred, serie):
        e = (serie.reindex(serie_pred.index) - serie_pred).abs()
        e[(e.index >= ini) & (e.index <= fin)] = np.nan
        return e

    filas, filas_val = [], []
    for barra in BARRAS:
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        prophet, y_pipe = prophet_y_objetivo(datasets_norm, scalers, barra)

        # errores por paso (indice = hora de emision) de cada estrategia y de S+C
        err = {c: {e: {} for e in ESTRATEGIAS} for c in ("val", "test")}
        err_sc = {}
        for h in range(1, N_HORIZONTES + 1):
            preds = estrategias_paso(pc[barra][h], prophet, y_pipe)
            for c in ("val", "test"):
                for e in ESTRATEGIAS:
                    ea = error_abs(preds[c][e], serie)
                    ea.index = ea.index - pd.Timedelta(hours=h)
                    err[c][e][h] = ea
            s = pd.Series(np.asarray(sc[barra][h]["test"]["y_pred"]), index=pd.to_datetime(sc[barra][h]["test"]["fechas"]))
            s = s[~s.index.duplicated()]
            ea = error_abs(s, serie)
            ea.index = ea.index - pd.Timedelta(hours=h)
            err_sc[h] = ea

        mae_val = {e: pd.DataFrame(err["val"][e]).mean().mean() for e in ESTRATEGIAS}
        mejor = min(mae_val, key=mae_val.get)
        for e in ESTRATEGIAS:
            filas_val.append({"Barra": barra, "Estrategia": e, "MAE_val": mae_val[e],
                              "MAE_test": pd.DataFrame(err["test"][e]).mean().mean(), "Elegida": e == mejor})

        d_pc = pd.DataFrame(err["test"][mejor])
        d_sc = pd.DataFrame(err_sc)
        comunes = d_pc.dropna().index.intersection(d_sc.dropna().index)
        m_pc, m_sc = d_pc.loc[comunes].mean(axis=1), d_sc.loc[comunes].mean(axis=1)
        r = diebold_mariano_desde_errores(m_sc.values, m_pc.values, h=24)
        filas.append({"Barra": barra, "Estrategia_PC": mejor, "n": len(comunes), "MAE_SC": m_sc.mean(),
                      "MAE_PC": m_pc.mean(), "DM": r["DM"], "p": r["p_value"]})
        print(f"[{barra}] estrategia P+C elegida en validacion: {mejor}", flush=True)

    res = pd.DataFrame(filas)
    sig = holm(dict(zip(res.index, res["p"])))
    res["sig_holm"] = res.index.isin(sig)
    res["favorece"] = np.where(res["DM"] < 0, "S+C", "P+C")
    pd.set_option("display.width", 220)
    print("\n=== Estrategias P+C por barra (MAE promedio 24 pasos, serie cruda) ===")
    print(pd.DataFrame(filas_val).round(3).to_string(index=False))
    print("\n=== S+C vs P+C a 24 horas, prueba (DM<0 favorece a S+C) ===")
    print(res.round(4).to_string(index=False))

    pd.DataFrame(filas_val).to_csv("Scripts/resultado_h24_pc_estrategias.csv", index=False)
    res.to_csv("Scripts/resultado_h24_pc_vs_sc.csv", index=False)
    print("\nGuardado: Scripts/resultado_h24_pc_estrategias.csv y Scripts/resultado_h24_pc_vs_sc.csv")


if __name__ == "__main__":
    main()
