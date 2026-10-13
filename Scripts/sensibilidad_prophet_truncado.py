#!/usr/bin/env python3
"""
sensibilidad_prophet_truncado.py - Camino barato (sin GPU) para medir el efecto de
truncar las predicciones de Prophet en 0 USD/MWh sobre el Stacking, SIN reentrenar el
modelo de residuos del TFT (que seguiria entrenado sobre el objetivo con Prophet sin
truncar). Reusa las predicciones base ya guardadas en estrategias_prophet_{arq}_raw.pkl
(Prophet_Solo, TFT_Precios_Solo, TFT_Residuos_Solo) y rehace solo las estrategias que
dependen de Prophet, con el mismo protocolo que el pipeline real: pesos ajustados en
train (grid_search_modelos, igual que Modulos/Stacking_Optimization_TFT.py), estrategia
ganadora elegida por MAE en VALIDACION, test reportado una sola vez. y_real se toma de
la serie cruda (no del pkl) para ser consistente con el resto de la tesis.

Resultado solo para este chat, no se escribe en la tesis.

Uso:
    python Scripts/sensibilidad_prophet_truncado.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Stacking_Optimization_TFT import grid_search_modelos, calcular_metricas
from Modulos.Comparativa import diebold_mariano
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA

RUTAS = {
    "LN": "Multi-Modelos_TFT/h/LN/pred_prophet_clima/Resultados/estrategias_prophet_LN_raw.pkl",
    "DyT": "Multi-Modelos_TFT/h/DyT/pred_prophet_clima/Resultados/estrategias_prophet_DyT_raw.pkl",
}
RUTA_CRUDA = "Datos/h/2020-2026.csv"
BARRAS = ["ATACAMA", "CARDONES", "CHARRUA", "CRUCERO", "P.AZUCAR", "P.MONTT", "QUILLOTA", "TARAPACA"]


def y_cruda(serie, fechas):
    fechas = pd.to_datetime(fechas)
    y = serie.reindex(fechas).values
    ini, fin = VENTANA_PRECIO_FALLA
    mask_falla = (fechas >= ini) & (fechas <= fin)
    return y, ~np.isnan(y) & ~mask_falla


def procesar(arq, df_crudo):
    with open(RUTAS[arq], "rb") as f:
        obj = pickle.load(f)

    filas = []
    for barra in BARRAS:
        d = obj[barra]
        sp = d["_datos_split"]
        serie = df_crudo.loc[df_crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()

        prophet = {c: np.asarray(d["Prophet_Solo"][f"prediccion_{c}"]).ravel() for c in ("train", "val", "test")}
        tft_p   = {c: np.asarray(d["TFT_Precios_Solo"][f"prediccion_{c}"]).ravel() for c in ("train", "val", "test")}
        tft_r   = {c: np.asarray(d["TFT_Residuos_Solo"][f"prediccion_{c}"]).ravel() for c in ("train", "val", "test")}
        fechas  = {c: pd.DatetimeIndex(sp[f"fechas_{c}"]) for c in ("train", "val", "test")}

        y, m = {}, {}
        for c in ("train", "val", "test"):
            y[c], m[c] = y_cruda(serie, fechas[c])
            n = min(len(y[c]), len(prophet[c]), len(tft_p[c]), len(tft_r[c]))
            y[c], m[c] = y[c][:n], m[c][:n]
            prophet[c], tft_p[c], tft_r[c] = prophet[c][:n], tft_p[c][:n], tft_r[c][:n]

        prophet_clip = {c: np.maximum(prophet[c], 0.0) for c in ("train", "val", "test")}

        def mean_or(a, b):
            return float(np.mean(np.abs(a - b)))

        # TFT precios solo: referencia, no cambia con el truncamiento
        mae_tft_test = mean_or(y["test"][m["test"]], tft_p["test"][m["test"]])

        for etiqueta, prophet_dict in (("original", prophet), ("truncado", prophet_clip)):
            # Estrategia "(P+R) + TFT precios", la ganadora en 7/8 barras en la tesis
            # (usa el residuo, que se entreno contra el Prophet SIN truncar: puede haber
            # un desajuste de reconstruccion si se trunca solo aqui, ver abajo)
            pr = {c: prophet_dict[c] + tft_r[c] for c in ("train", "val", "test")}
            r = grid_search_modelos(y["train"][m["train"]], pr["train"][m["train"]], tft_p["train"][m["train"]],
                                    resolution=41, metrica="MAE", w_min=0.0)
            w1, w2 = r["w1"], r["w2"]
            pred_val = w1 * pr["val"] + w2 * tft_p["val"]
            pred_test = w1 * pr["test"] + w2 * tft_p["test"]
            mae_val = mean_or(y["val"][m["val"]], pred_val[m["val"]])
            mae_test = mean_or(y["test"][m["test"]], pred_test[m["test"]])

            # Estrategia "Prophet + TFT precios" (SIN residuo): no tiene el desajuste de
            # reconstruccion anterior, porque Prophet nunca se combino con un residuo
            # entrenado sobre su version sin truncar. Aisla el efecto limpio de truncar.
            r2 = grid_search_modelos(y["train"][m["train"]], prophet_dict["train"][m["train"]], tft_p["train"][m["train"]],
                                     resolution=41, metrica="MAE", w_min=0.0)
            w1b, w2b = r2["w1"], r2["w2"]
            pred_test_b = w1b * prophet_dict["test"] + w2b * tft_p["test"]
            mae_test_sin_residuo = mean_or(y["test"][m["test"]], pred_test_b[m["test"]])

            filas.append({
                "Arq": arq, "Barra": barra, "Prophet": etiqueta,
                "w_PR": round(w1, 3), "w_TFT": round(w2, 3),
                "MAE_val_PR+TFT": round(mae_val, 4), "MAE_test_PR+TFT": round(mae_test, 4),
                "MAE_test_TFTsolo": round(mae_tft_test, 4),
                "Gana_estrategia": "PR+TFT" if mae_val < mae_tft_test else "TFT_solo",
                "MAE_test_final": round(min(mae_test, mae_tft_test) if mae_val >= mae_tft_test else mae_test, 4),
                "w_Pclean": round(w1b, 3), "MAE_test_Prophet+TFT_sin_residuo": round(mae_test_sin_residuo, 4),
                "pred_test": pred_test, "mask_test": m["test"], "y_test": y["test"],
            })
    return filas


def main():
    df_crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])

    todas = []
    for arq in RUTAS:
        if not os.path.exists(RUTAS[arq]):
            print(f"[AVISO] no encontrado: {RUTAS[arq]}")
            continue
        todas += procesar(arq, df_crudo)

    df = pd.DataFrame(todas)
    cols_mostrar = ["Arq", "Barra", "Prophet", "w_PR", "w_TFT", "MAE_val_PR+TFT",
                    "MAE_test_PR+TFT", "MAE_test_TFTsolo", "Gana_estrategia", "MAE_test_final",
                    "w_Pclean", "MAE_test_Prophet+TFT_sin_residuo"]
    pd.set_option("display.width", 200)
    print(df[cols_mostrar].to_string(index=False))

    print("\n=== Comparacion original vs truncado (MAE_test_final, estrategia ganadora por VALIDACION) ===")
    piv = df.pivot_table(index=["Arq", "Barra"], columns="Prophet", values="MAE_test_final")
    piv["Delta_%"] = 100 * (piv["truncado"] - piv["original"]) / piv["original"]
    print(piv.round(4).to_string())

    # Test de Diebold-Mariano: estrategia ganadora original vs truncada, por barra/arq
    print("\n=== Diebold-Mariano: P+C truncado (a) vs P+C original (b); DM<0 favorece al truncado ===")
    for arq in df["Arq"].unique():
        for barra in BARRAS:
            fo = df[(df.Arq == arq) & (df.Barra == barra) & (df.Prophet == "original")].iloc[0]
            ft = df[(df.Arq == arq) & (df.Barra == barra) & (df.Prophet == "truncado")].iloc[0]
            msk = fo["mask_test"] & ft["mask_test"]
            res = diebold_mariano(fo["y_test"][msk], ft["pred_test"][msk], fo["pred_test"][msk], h=1, loss="mae")
            print(f"{arq:4s} {barra:10s} DM={res['DM']:7.3f}  p={res['p_value']:.4f}")

    df[cols_mostrar].to_csv("Scripts/resultado_sensibilidad_prophet.csv", index=False)
    print("\nGuardado: Scripts/resultado_sensibilidad_prophet.csv")


if __name__ == "__main__":
    main()
