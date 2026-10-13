#!/usr/bin/env python3
"""
analisis_h24_errores_cp.py - Analisis de errores y Conformal Prediction "bien hechos"
para el horizonte day-ahead (h=24), S+C-LN, las 5 barras con modelo h24 entrenado
(ATACAMA, CHARRUA, P.AZUCAR, P.MONTT, TARAPACA). Reusa la misma inferencia eficiente
de Scripts/evaluar_h24.py (una sola pasada por barra, reusada para los 24 pasos).

Para cada paso h=1..24:
  - MAE general y MAE en el regimen alto (percentil 95 de la serie cruda de ese split),
    contra la serie cruda, para ver si el "error que importa" crece igual de rapido que
    el MAE promedio o no.
  - Split CP calibrado EN ESE PASO: score de no conformidad de validacion -> cuantil ->
    intervalo evaluado en el test de ese mismo paso. Un cuantil y un ancho por paso,
    en vez de uno solo mezclando los 24 (consistente con la logica de LW-ACP, que ya
    pondera por hora del dia; aqui se pondera por paso del horizonte). alpha=0.05 y
    alpha=0.10 (ver nota sobre colas pesadas, Scripts/resultado_cp_90_vs_95.csv).

No toca el escrito ni el pipeline existente; resultados solo para este chat/analisis.

Uso:
    python Scripts/analisis_h24_errores_cp.py
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.TFT_Model import TFTBridge
from Modulos.Evaluacion_TFT import extraer_todas_predicciones_modelo, inferir_raw_modelo, VENTANA_PRECIO_FALLA
from Modulos.Conformal_Prediction_Wrapper import split_cp_from_calibration

EXPERIMENTO_PRECIOS = "Multi-TFT_Precios"
CARPETA = "Multi-Modelos_TFT/h/LN/pred_sol_clima_h24"
CARPETA_NORM = "Multi-Modelos_TFT/h/LN/pred_sol_clima"
BARRAS = ["ATACAMA", "CHARRUA", "P.AZUCAR", "P.MONTT", "TARAPACA"]
N_HORIZONTES = 24
RUTA_CRUDA = "Datos/h/2020-2026.csv"
ALPHAS = {0.05: "95%", 0.10: "90%"}


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


def y_cruda(serie, fechas):
    fechas = pd.to_datetime(fechas)
    y = serie.reindex(fechas).values
    ini, fin = VENTANA_PRECIO_FALLA
    mask_falla = (fechas >= ini) & (fechas <= fin)
    return y, ~np.isnan(y) & ~mask_falla


def main():
    df_crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])

    with open(os.path.join(CARPETA, "dataloaders_precios.pkl"), "rb") as f:
        dataloaders_precios = pickle.load(f)
    with open(os.path.join(CARPETA_NORM, "datasets_norm.pkl"), "rb") as f:
        datasets_norm = pickle.load(f)
    with open(os.path.join(CARPETA, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    filas = []
    for barra in BARRAS:
        print(f"\n=== {barra} ===")
        serie = df_crudo.loc[df_crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        modelo = cargar_modelo(barra)
        raw = inferir_raw_modelo(modelo, dataloaders_precios[barra]["test"])

        for h in range(1, N_HORIZONTES + 1):
            resultados = extraer_todas_predicciones_modelo(
                barra, modelo, dataloaders_precios[barra]["test"], datasets_norm, scalers,
                horizonte=h, raw=raw,
            )
            val, test = resultados["val"], resultados["test"]

            y_val, m_val = y_cruda(serie, val["fechas"])
            y_test, m_test = y_cruda(serie, test["fechas"])
            yp_val, yp_test = np.asarray(val["y_pred"]), np.asarray(test["y_pred"])

            y_val, yp_val = y_val[m_val], yp_val[m_val]
            y_test, yp_test = y_test[m_test], yp_test[m_test]

            e_test = np.abs(y_test - yp_test)
            u95 = np.percentile(y_test, 95)
            mae_gen = float(np.mean(e_test))
            mae_p95 = float(np.mean(e_test[y_test >= u95])) if (y_test >= u95).any() else np.nan

            fila = {"Barra": barra, "h": h, "MAE": round(mae_gen, 3), "MAE_p95": round(mae_p95, 3)}
            for alpha, etiqueta in ALPHAS.items():
                r = split_cp_from_calibration(y_val, yp_val, y_test, yp_test, alpha=alpha)
                fila[f"Ancho_{etiqueta}"] = round(r["width"], 2)
                fila[f"Cobertura_{etiqueta}"] = round(r["coverage"], 4)
            filas.append(fila)
            print(f"  h={h:2d}  MAE={mae_gen:6.2f}  MAE_p95={mae_p95:6.2f}  "
                  f"Ancho95={fila['Ancho_95%']:6.2f} (cov={fila['Cobertura_95%']:.3f})  "
                  f"Ancho90={fila['Ancho_90%']:6.2f} (cov={fila['Cobertura_90%']:.3f})")

        del raw, modelo
        gc.collect()

    df = pd.DataFrame(filas)
    pd.set_option("display.width", 200)

    print("\n=== Resumen: paso 1 vs paso 24 ===")
    for barra in BARRAS:
        d = df[df.Barra == barra]
        p1, p24 = d[d.h == 1].iloc[0], d[d.h == 24].iloc[0]
        print(f"{barra:10s}  MAE: {p1['MAE']:.2f} -> {p24['MAE']:.2f} ({100*(p24['MAE']-p1['MAE'])/p1['MAE']:+.1f}%)   "
              f"Ancho95: {p1['Ancho_95%']:.2f} -> {p24['Ancho_95%']:.2f} ({100*(p24['Ancho_95%']-p1['Ancho_95%'])/p1['Ancho_95%']:+.1f}%)   "
              f"Ancho90: {p1['Ancho_90%']:.2f} -> {p24['Ancho_90%']:.2f} ({100*(p24['Ancho_90%']-p1['Ancho_90%'])/p1['Ancho_90%']:+.1f}%)")

    print("\n=== Promedio por paso (las 5 barras) ===")
    print(df.groupby("h")[["MAE", "MAE_p95", "Ancho_95%", "Cobertura_95%", "Ancho_90%", "Cobertura_90%"]].mean().round(3).to_string())

    df.to_csv("Scripts/resultado_h24_errores_cp.csv", index=False)
    print("\nGuardado: Scripts/resultado_h24_errores_cp.csv")


if __name__ == "__main__":
    main()
