#!/usr/bin/env python3
"""
errores_extremos_h24.py - Extiende el analisis de errores extremos ("desfase de fase"
vs "evento estructural", Seccion 4.6.9/4.5.8 de la tesis) al horizonte day-ahead
(h=24), para ver si el mismo mecanismo de desfase persiste a medida que crece el paso
del horizonte dentro de la ventana day-ahead.

Metodologia identica a la usada para h=1 (misma descripcion que ya esta en el escrito,
Seccion de errores extremos): para cada paso h=1..24, se toman los 20 errores absolutos
mas grandes del conjunto de VALIDACION de Atacama (misma barra ilustrativa usada en el
analisis de h=1), se recalcula el error tras desplazar la prediccion entre -3 y +3 horas
respecto a su posicion original, y se retiene el desplazamiento que minimiza el error. Si
la reduccion de error alcanzada supera el 60%, el caso se clasifica como "desfase de
fase"; en caso contrario, "evento estructural".

Reusa la misma inferencia eficiente de Scripts/evaluar_h24.py (una sola pasada por
barra, reusada para los 24 pasos via el parametro raw=).

Uso:
    python Scripts/errores_extremos_h24.py
"""
import os
import sys
import json
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.TFT_Model import TFTBridge
from Modulos.Evaluacion_TFT import extraer_todas_predicciones_modelo, inferir_raw_modelo, VENTANA_PRECIO_FALLA

EXPERIMENTO_PRECIOS = "Multi-TFT_Precios"
CARPETA = "Multi-Modelos_TFT/h/LN/pred_sol_clima_h24"
CARPETA_NORM = "Multi-Modelos_TFT/h/LN/pred_sol_clima"
BARRA = "ATACAMA"
N_HORIZONTES = 24
RUTA_CRUDA = "Datos/h/2020-2026.csv"
TOP_N = 20
DESPLAZAMIENTOS = range(-3, 4)
UMBRAL_REDUCCION = 0.60
HORAS_TRANSICION = {7, 8, 19, 20, 21}


def cargar_modelo(barra):
    ruta_config = os.path.join(CARPETA, EXPERIMENTO_PRECIOS, barra, f"{EXPERIMENTO_PRECIOS}_{barra}_config.json")
    with open(ruta_config, encoding="utf-8") as f:
        config = json.load(f)
    ckpt = config["model_path"]
    if not os.path.exists(ckpt):
        ckpt = os.path.join(CARPETA, EXPERIMENTO_PRECIOS, barra, os.path.basename(ckpt))

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


def clasificar(fechas, y_pred, y_real_cont):
    """y_real_cont: pd.Series horaria continua (reindexada), para poder consultar t+shift
    aunque esa hora no haya quedado en el conjunto de validacion de este paso."""
    e = np.abs(y_pred - y_real_cont.reindex(fechas).values)
    idx_top = np.argsort(e)[::-1][:TOP_N]

    filas = []
    for i in idx_top:
        t = fechas[i]
        e_orig = e[i]
        mejor_red, mejor_shift = -np.inf, 0
        for s in DESPLAZAMIENTOS:
            t_shift = t + pd.Timedelta(hours=s)
            if t_shift not in y_real_cont.index or pd.isna(y_real_cont.loc[t_shift]):
                continue
            e_shift = abs(y_pred[i] - y_real_cont.loc[t_shift])
            red = 1 - e_shift / e_orig if e_orig > 0 else 0
            if red > mejor_red:
                mejor_red, mejor_shift = red, s
        es_desfase = mejor_red > UMBRAL_REDUCCION
        filas.append({
            "fecha": t, "error_original": e_orig, "mejor_desplazamiento": mejor_shift,
            "reduccion": mejor_red, "desfase": es_desfase,
            "hora": t.hour, "transicion_solar": t.hour in HORAS_TRANSICION,
        })
    return pd.DataFrame(filas)


def main():
    df_crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])
    serie = df_crudo.loc[df_crudo["Barra"] == BARRA].groupby("Fecha")["Valor"].mean()
    ini_falla, fin_falla = VENTANA_PRECIO_FALLA
    serie = serie.copy()
    serie.loc[(serie.index >= ini_falla) & (serie.index <= fin_falla)] = np.nan

    with open(os.path.join(CARPETA, "dataloaders_precios.pkl"), "rb") as f:
        dataloaders_precios = pickle.load(f)
    with open(os.path.join(CARPETA_NORM, "datasets_norm.pkl"), "rb") as f:
        datasets_norm = pickle.load(f)
    with open(os.path.join(CARPETA, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    modelo = cargar_modelo(BARRA)
    raw = inferir_raw_modelo(modelo, dataloaders_precios[BARRA]["test"])

    filas_resumen = []
    detalle_todos = {}
    for h in range(1, N_HORIZONTES + 1):
        resultados = extraer_todas_predicciones_modelo(
            BARRA, modelo, dataloaders_precios[BARRA]["test"], datasets_norm, scalers,
            horizonte=h, raw=raw,
        )
        val = resultados["val"]
        fechas = pd.to_datetime(val["fechas"])
        y_pred = np.asarray(val["y_pred"])
        y_real = serie.reindex(fechas).values
        mask = ~np.isnan(y_real)
        fechas, y_pred, y_real = fechas[mask], y_pred[mask], y_real[mask]

        det = clasificar(fechas, y_pred, serie)
        detalle_todos[h] = det

        n_desfase = int(det["desfase"].sum())
        n_transicion = int(det["transicion_solar"].sum())
        red_mediana = det.loc[det["desfase"], "reduccion"].median() if n_desfase > 0 else np.nan
        pct_positivo = float((det.loc[det["desfase"], "mejor_desplazamiento"] > 0).mean()) if n_desfase > 0 else np.nan

        e_val = np.abs(y_pred - y_real)
        mae = float(np.mean(e_val))
        p95 = float(np.percentile(e_val, 95))
        maximo = float(np.max(e_val))

        filas_resumen.append({
            "h": h, "MAE": round(mae, 2), "P95": round(p95, 2), "Maximo": round(maximo, 2),
            "Desfase": n_desfase, "Transicion_solar": n_transicion,
            "Reduccion_mediana_%": round(100 * red_mediana, 1) if not np.isnan(red_mediana) else np.nan,
            "Pct_desplazamiento_positivo_%": round(100 * pct_positivo, 1) if not np.isnan(pct_positivo) else np.nan,
        })
        print(f"h={h:2d}  MAE={mae:6.2f}  P95={p95:6.2f}  Max={maximo:7.2f}  "
              f"Desfase={n_desfase:2d}/20  Transicion={n_transicion:2d}/20  "
              f"RedMed={red_mediana*100 if not np.isnan(red_mediana) else float('nan'):5.1f}%  "
              f"PosShift={pct_positivo*100 if not np.isnan(pct_positivo) else float('nan'):5.1f}%")

    df_resumen = pd.DataFrame(filas_resumen)
    df_resumen.to_csv("Scripts/resultado_errores_extremos_h24.csv", index=False)
    print("\nGuardado: Scripts/resultado_errores_extremos_h24.csv")

    with pd.ExcelWriter("Scripts/resultado_errores_extremos_h24_detalle.xlsx") as writer:
        for h, det in detalle_todos.items():
            det.to_excel(writer, sheet_name=f"h{h}", index=False)
    print("Guardado: Scripts/resultado_errores_extremos_h24_detalle.xlsx")


if __name__ == "__main__":
    main()
