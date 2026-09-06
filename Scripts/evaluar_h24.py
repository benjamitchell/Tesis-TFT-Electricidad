#!/usr/bin/env python3
"""
evaluar_h24.py - Evalua los modelos S+C entrenados a horizonte day-ahead h=24
(MIMO), tanto el original (train_h24.sh, 5 barras, LN+DyT) como el piloto
combinado sin-outliers+h24 (train_piloto_outliers_h24.sh, 4 barras, LN).

Para cada barra/arquitectura/experimento, extrae la prediccion de cada uno de
los 24 pasos del horizonte (extraer_todas_predicciones_modelo con
horizonte=1..24 -- fix de fecha por paso ya aplicado en Modulos/Evaluacion_TFT.py,
ver commit de este mismo dia) y la compara paso a paso contra la linea base de
persistencia a 24h (mismo valor que hace 24 horas, calcular_baselines_naive con
lag=24), calculando MASE y skill score por paso. Reporta el promedio sobre los
24 pasos y como se degrada el MAE del paso 1 al paso 24.

Corre localmente una vez que los checkpoints de train_h24.sh /
train_piloto_outliers_h24.sh ya estan sincronizados desde el cluster.

Uso:
    python Scripts/evaluar_h24.py
"""
import os
import sys
import gc
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modulos.TFT_Model import TFTBridge
from Modulos.Parche import activar_dyt_mode, desactivar_dyt_mode
from Modulos.Evaluacion_TFT import (
    extraer_todas_predicciones_modelo, recalcular_metricas_serie_cruda,
    calcular_baselines_naive, inferir_raw_modelo,
)

# El nombre de la carpeta de experimento difiere entre pipelines: LN usa
# "Multi-TFT_Precios", DyT usa solo "Precios" (ver combos en Resultados_resumen.ipynb).
EXPERIMENTO_PRECIOS = {"LN": "Multi-TFT_Precios", "DyT": "Precios"}
N_HORIZONTES = 24

# (nombre, carpeta_base, carpeta_datasets_norm, barras, arquitecturas)
# datasets_norm.pkl no se duplica en las carpetas _h24 (max_prediction_length=24
# reusa el mismo datasets_norm/scalers de la carpeta base h=1, ver LEEME.txt de
# Subir_Cluster) -- se carga desde ahi en vez de la carpeta h24.
EXPERIMENTOS = [
    ("h24_original", "Multi-Modelos_TFT/h/{arq}/pred_sol_clima_h24",
     "Multi-Modelos_TFT/h/{arq}/pred_sol_clima",
     ["ATACAMA", "CHARRUA", "P.MONTT", "P.AZUCAR", "TARAPACA"], ["LN", "DyT"]),
    ("piloto_sin_outliers_h24", "Multi-Modelos_TFT/h/{arq}/pred_sol_clima_piloto_sin_outliers_h24",
     "Multi-Modelos_TFT/h/{arq}/pred_sol_clima_piloto_sin_outliers",
     ["ATACAMA", "CHARRUA", "P.AZUCAR", "TARAPACA"], ["LN"]),
]


def cargar_modelo(carpeta_modelos, barra, variante):
    exp = EXPERIMENTO_PRECIOS[variante]
    ruta_config = os.path.join(carpeta_modelos, exp, barra, f"{exp}_{barra}_config.json")
    if not os.path.exists(ruta_config):
        print(f"  [AVISO] no encontrado: {ruta_config}")
        return None
    with open(ruta_config, encoding="utf-8") as f:
        config = json.load(f)
    ckpt = config["model_path"]
    if not os.path.exists(ckpt):
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
    # modelo (Modulos/Parche.py); hay que activarlo antes de load_from_checkpoint
    # o el state_dict no calza con la arquitectura reconstruida (RuntimeError).
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
    filas_resumen = []
    filas_detalle = []

    for nombre_exp, carpeta_tpl, carpeta_norm_tpl, barras, arqs in EXPERIMENTOS:
        for arq in arqs:
            carpeta = carpeta_tpl.format(arq=arq)
            carpeta_norm = carpeta_norm_tpl.format(arq=arq)
            if not os.path.exists(carpeta):
                print(f"[SKIP] no existe {carpeta} (¿ya se sincronizo del cluster?)")
                continue

            with open(os.path.join(carpeta, "dataloaders_precios.pkl"), "rb") as f:
                dataloaders_precios = pickle.load(f)
            with open(os.path.join(carpeta_norm, "datasets_norm.pkl"), "rb") as f:
                datasets_norm = pickle.load(f)
            with open(os.path.join(carpeta, "scalers.pkl"), "rb") as f:
                scalers = pickle.load(f)

            for barra in barras:
                print(f"\n=== {nombre_exp} / {arq} / {barra} ===")
                modelo = cargar_modelo(carpeta, barra, arq)
                if modelo is None:
                    continue

                fechas_train = datasets_norm[barra]["train"]["ds"]

                # Inferir una sola vez y reusar el resultado para los 24 horizontes --
                # llamar extraer_todas_predicciones_modelo suelto en el loop de abajo
                # re-hacia la inferencia completa sobre las ~8300h de test 24 veces por
                # nada (encontrado tras un OOM silencioso a mitad de la evaluacion).
                raw = inferir_raw_modelo(modelo, dataloaders_precios[barra]["test"])

                mae_pasos, mase_pasos = [], []
                for h in range(1, N_HORIZONTES + 1):
                    resultados = extraer_todas_predicciones_modelo(
                        barra, modelo, dataloaders_precios[barra]["test"], datasets_norm, scalers,
                        horizonte=h, raw=raw,
                    )
                    test = resultados["test"]
                    r_modelo = recalcular_metricas_serie_cruda(barra, test["fechas"], test["y_pred"], verbose=False)
                    mae_h = r_modelo["general"]["MAE"]

                    # No se pasa mae_modelo: calcular_baselines_naive solo sabe calcular
                    # MASE/skill contra 'persistencia' (lag=1); aqui se necesita contra
                    # lag=24, asi que se arma manualmente desde naive_lag24 abajo.
                    r_base = calcular_baselines_naive(
                        barra, fechas_train, test["fechas"],
                        lags_naive=(24,), lags_ar=(), verbose=False,
                    )
                    mae_base_h = r_base["naive_lag24"]["MAE"]
                    mase_h = mae_h / mae_base_h
                    skill_h = 1 - mase_h

                    mae_pasos.append(mae_h)
                    mase_pasos.append(mase_h)
                    filas_detalle.append({
                        "Experimento": nombre_exp, "Arq": arq, "Barra": barra, "Horizonte_h": h,
                        "MAE_modelo": mae_h, "MAE_persistencia24": mae_base_h,
                        "MASE": mase_h, "skill": skill_h,
                    })
                    print(f"  h={h:2d}  MAE={mae_h:6.2f}  MAE_persist24={mae_base_h:6.2f}  "
                          f"MASE={mase_h:.3f}  skill={skill_h:+.3f}")

                filas_resumen.append({
                    "Experimento": nombre_exp, "Arq": arq, "Barra": barra,
                    "MAE_promedio_24h": np.mean(mae_pasos),
                    "MAE_paso1": mae_pasos[0], "MAE_paso24": mae_pasos[-1],
                    "MASE_promedio": np.mean(mase_pasos),
                    "skill_promedio": 1 - np.mean(mase_pasos),
                })

                del raw, modelo
                gc.collect()

    df_resumen = pd.DataFrame(filas_resumen)
    df_detalle = pd.DataFrame(filas_detalle)

    if df_resumen.empty:
        print("\nNo se encontro ningun checkpoint. ¿Ya se sincronizaron los resultados del cluster?")
        return

    print("\n=== Resumen (promedio sobre los 24 pasos de horizonte) ===")
    print(df_resumen.to_string(index=False))

    print("\n=== Degradacion paso 1 -> paso 24 (MAE) ===")
    for _, fila in df_resumen.iterrows():
        delta = 100 * (fila["MAE_paso24"] - fila["MAE_paso1"]) / fila["MAE_paso1"]
        print(f"  {fila['Experimento']:25} {fila['Arq']:4} {fila['Barra']:10} "
              f"paso1={fila['MAE_paso1']:6.2f}  paso24={fila['MAE_paso24']:6.2f}  ({delta:+.1f}%)")

    df_resumen.to_csv("Scripts/resultado_h24_resumen.csv", index=False)
    df_detalle.to_csv("Scripts/resultado_h24_detalle.csv", index=False)
    print("\nGuardado: Scripts/resultado_h24_resumen.csv y Scripts/resultado_h24_detalle.csv")


if __name__ == "__main__":
    main()
