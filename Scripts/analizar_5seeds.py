#!/usr/bin/env python3
"""
analizar_5seeds.py - Agrega los resultados del estudio de >=5 semillas por
arquitectura (train_5seeds.sh) en una tabla de media +/- desviacion estandar
por barra y arquitectura, y compara LN vs DyT.

Corre localmente una vez que los 30 checkpoints de train_5seeds.sh ya estan
sincronizados desde el cluster. Reusa el patron de inferencia ya validado en
Modulos/TFT_Model.py (cargar_modelo_entrenado) + Modulos/Evaluacion_TFT.py, y
evalua contra la serie cruda (Datos/h/2020-2026.csv), igual que el resto del
capitulo de Resultados.

Uso:
    python Scripts/analizar_5seeds.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modulos.TFT_Model import TFTBridge
from Modulos.Parche import activar_dyt_mode, desactivar_dyt_mode
from Modulos.Evaluacion_TFT import extraer_todas_predicciones_modelo, recalcular_metricas_serie_cruda
from Modulos.Comparativa import diebold_mariano

BARRAS = ["ATACAMA", "CHARRUA", "P.MONTT"]
VARIANTES = ["LN", "DyT"]
SEMILLAS = [1, 2, 3, 4, 5]
BASE = "Multi-Modelos_TFT/h"
# El nombre de la carpeta de experimento difiere entre pipelines: LN usa
# "Multi-TFT_Precios", DyT usa solo "Precios" (ver combos en Resultados_resumen.ipynb).
EXPERIMENTO_PRECIOS = {"LN": "Multi-TFT_Precios", "DyT": "Precios"}


def cargar_modelo(carpeta_modelos, barra, variante):
    exp = EXPERIMENTO_PRECIOS[variante]
    ruta_config = os.path.join(carpeta_modelos, exp, barra, f"{exp}_{barra}_config.json")
    if not os.path.exists(ruta_config):
        return None
    import json
    with open(ruta_config, encoding="utf-8") as f:
        config = json.load(f)
    ckpt = config["model_path"]
    if not os.path.exists(ckpt):
        # el model_path guardado suele ser la ruta absoluta del cluster; se intenta
        # resolver localmente asumiendo que la carpeta se sincronizo tal cual
        ckpt_local = os.path.join(carpeta_modelos, exp, barra, os.path.basename(ckpt))
        if os.path.exists(ckpt_local):
            ckpt = ckpt_local
        else:
            print(f"  [AVISO] checkpoint no encontrado: {ckpt}")
            return None

    _orig_zeros = torch.zeros
    def _safe_zeros(*a, **kw):
        d = kw.get("device")
        if d is not None and "cuda" in str(d):
            kw["device"] = "cpu"
        return _orig_zeros(*a, **kw)
    # DyT reemplaza nn.LayerNorm por DynamicTanh vía monkey-patch al construir el
    # modelo (Modulos/Parche.py); load_from_checkpoint reconstruye la arquitectura
    # desde cero antes de cargar el state_dict, asi que hay que activar el mismo
    # parche aca o los nombres/formas de las capas no calzan (RuntimeError).
    if variante == "DyT":
        activar_dyt_mode()
    torch.zeros = _safe_zeros
    try:
        modelo = TFTBridge.load_from_checkpoint(ckpt, map_location=lambda storage, loc: storage, weights_only=False)
    finally:
        torch.zeros = _orig_zeros
        if variante == "DyT":
            desactivar_dyt_mode()
    modelo.eval()
    for m in modelo.modules():
        if hasattr(m, "_device"):
            m._device = torch.device("cpu")
    return modelo


def main():
    filas = []
    predicciones_test = {}  # (barra, variante, seed) -> (fechas, y_pred)

    for barra in BARRAS:
        for variante in VARIANTES:
            carpeta_origen = os.path.join(BASE, variante, "pred_sol_clima")
            with open(os.path.join(carpeta_origen, "dataloaders_precios.pkl"), "rb") as f:
                dataloaders_precios = pickle.load(f)
            with open(os.path.join(carpeta_origen, "datasets_norm.pkl"), "rb") as f:
                datasets_norm = pickle.load(f)
            with open(os.path.join(carpeta_origen, "scalers.pkl"), "rb") as f:
                scalers = pickle.load(f)

            for seed in SEMILLAS:
                carpeta_modelos = os.path.join(BASE, variante, "pred_sol_clima_seeds", f"seed{seed}")
                print(f"Cargando {barra} / {variante} / seed {seed} ...")
                modelo = cargar_modelo(carpeta_modelos, barra, variante)
                if modelo is None:
                    continue

                resultados = extraer_todas_predicciones_modelo(
                    barra, modelo, dataloaders_precios[barra]["test"], datasets_norm, scalers
                )
                test = resultados["test"]
                r = recalcular_metricas_serie_cruda(barra, test["fechas"], test["y_pred"], verbose=False)
                filas.append({"Barra": barra, "Variante": variante, "Seed": seed, "MAE": r["general"]["MAE"]})
                predicciones_test[(barra, variante, seed)] = (test["fechas"], test["y_pred"])

    df = pd.DataFrame(filas)
    if df.empty:
        print("No se encontro ningun checkpoint. ¿Ya se sincronizaron los resultados del cluster?")
        return

    resumen = df.groupby(["Barra", "Variante"])["MAE"].agg(["mean", "std", "count"]).reset_index()
    print("\n=== MAE media +/- desviacion estandar, por barra y arquitectura (n=5 semillas) ===")
    print(resumen.to_string(index=False))

    print("\n=== ¿Se solapan LN y DyT dentro de +/-1 sd? ===")
    for barra in BARRAS:
        ln = resumen[(resumen.Barra == barra) & (resumen.Variante == "LN")]
        dyt = resumen[(resumen.Barra == barra) & (resumen.Variante == "DyT")]
        if ln.empty or dyt.empty:
            continue
        ln_lo, ln_hi = ln["mean"].iloc[0] - ln["std"].iloc[0], ln["mean"].iloc[0] + ln["std"].iloc[0]
        dyt_lo, dyt_hi = dyt["mean"].iloc[0] - dyt["std"].iloc[0], dyt["mean"].iloc[0] + dyt["std"].iloc[0]
        solapan = (ln_lo <= dyt_hi) and (dyt_lo <= ln_hi)
        print(f"  {barra:10} LN=[{ln_lo:.2f},{ln_hi:.2f}]  DyT=[{dyt_lo:.2f},{dyt_hi:.2f}]  -> {'SOLAPAN (equivalentes)' if solapan else 'NO solapan'}")

    df.to_csv("Scripts/resultado_5seeds.csv", index=False)
    print("\nGuardado: Scripts/resultado_5seeds.csv")


if __name__ == "__main__":
    main()
