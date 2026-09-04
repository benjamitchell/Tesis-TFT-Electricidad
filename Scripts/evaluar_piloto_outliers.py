#!/usr/bin/env python3
"""
evaluar_piloto_outliers.py - Compara el piloto de train sin recorte de outliers
(train_piloto_outliers.sh) contra los modelos actuales de pred_sol_clima, para
las 4 barras del piloto (ATACAMA, CHARRUA, P.AZUCAR, TARAPACA), arquitectura LN.

Corre localmente una vez que los 4 checkpoints de train_piloto_outliers.sh ya
estan sincronizados desde el cluster en
Multi-Modelos_TFT/h/LN/pred_sol_clima_piloto_sin_outliers/Multi-TFT_Precios/<BARRA>/.

Reporta MAE general y MAE en regimen alto (percentil 95+) de ambos modelos,
mas MASE/skill score del piloto respecto al modelo actual (usando el MAE del
modelo actual como referencia, no persistencia -- la pregunta aqui es "mejoro
el tratamiento de outliers en train", no "le gana a un baseline ingenuo").

Uso:
    python Scripts/evaluar_piloto_outliers.py
"""
import os
import sys
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modulos.TFT_Model import TFTBridge
from Modulos.Evaluacion_TFT import extraer_todas_predicciones_modelo, recalcular_metricas_serie_cruda

BARRAS = ["ATACAMA", "CHARRUA", "P.AZUCAR", "TARAPACA"]
EXPERIMENTO_PRECIOS = "Multi-TFT_Precios"
ORIGEN_ACTUAL = "Multi-Modelos_TFT/h/LN/pred_sol_clima"
ORIGEN_PILOTO = "Multi-Modelos_TFT/h/LN/pred_sol_clima_piloto_sin_outliers"


def cargar_modelo(carpeta_modelos, barra):
    ruta_config = os.path.join(carpeta_modelos, EXPERIMENTO_PRECIOS, barra,
                                f"{EXPERIMENTO_PRECIOS}_{barra}_config.json")
    if not os.path.exists(ruta_config):
        print(f"  [AVISO] no encontrado: {ruta_config}")
        return None
    with open(ruta_config, encoding="utf-8") as f:
        config = json.load(f)
    ckpt = config["model_path"]
    if not os.path.exists(ckpt):
        ckpt_local = os.path.join(carpeta_modelos, EXPERIMENTO_PRECIOS, barra, os.path.basename(ckpt))
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


def evaluar(origen, barra, carpeta_modelos):
    with open(os.path.join(origen, "dataloaders_precios.pkl"), "rb") as f:
        dataloaders_precios = pickle.load(f)
    with open(os.path.join(origen, "datasets_norm.pkl"), "rb") as f:
        datasets_norm = pickle.load(f)
    with open(os.path.join(origen, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    modelo = cargar_modelo(carpeta_modelos, barra)
    if modelo is None:
        return None

    resultados = extraer_todas_predicciones_modelo(
        barra, modelo, dataloaders_precios[barra]["test"], datasets_norm, scalers
    )
    test = resultados["test"]
    return recalcular_metricas_serie_cruda(barra, test["fechas"], test["y_pred"], verbose=False)


def main():
    filas = []
    for barra in BARRAS:
        print(f"Barra: {barra}")

        print("  Modelo actual (train con recorte z>=4)...")
        r_actual = evaluar(ORIGEN_ACTUAL, barra, ORIGEN_ACTUAL)

        print("  Piloto (train sin recorte, solo excluidas 3h absurdas P.AZUCAR)...")
        r_piloto = evaluar(ORIGEN_PILOTO, barra, ORIGEN_PILOTO)

        if r_actual is None or r_piloto is None:
            print("  [SKIP] falta uno de los dos checkpoints/resultados")
            continue

        mae_actual = r_actual["general"]["MAE"]
        mae_piloto = r_piloto["general"]["MAE"]
        mase = mae_piloto / mae_actual
        skill = 1 - mase

        p95_actual = r_actual["p95_metricas"]["MAE"] if r_actual["p95_metricas"] else np.nan
        p95_piloto = r_piloto["p95_metricas"]["MAE"] if r_piloto["p95_metricas"] else np.nan
        mase_p95 = p95_piloto / p95_actual if p95_actual and not np.isnan(p95_actual) else np.nan

        filas.append({
            "Barra": barra,
            "MAE_actual": mae_actual, "MAE_piloto": mae_piloto,
            "Delta_MAE_%": 100 * (mae_piloto - mae_actual) / mae_actual,
            "MAE_p95_actual": p95_actual, "MAE_p95_piloto": p95_piloto,
            "Delta_MAE_p95_%": 100 * (p95_piloto - p95_actual) / p95_actual if p95_actual else np.nan,
            "MASE_vs_actual": mase, "skill_vs_actual": skill,
        })

    df = pd.DataFrame(filas)
    if df.empty:
        print("No se encontro ningun checkpoint. ¿Ya se sincronizaron los resultados del cluster?")
        return

    print("\n=== Piloto sin-outliers vs. modelo actual (MAE general y regimen alto p95) ===")
    print(df.to_string(index=False))

    mejoro_general = (df["Delta_MAE_%"] < 0).sum()
    mejoro_p95 = (df["Delta_MAE_p95_%"] < 0).sum()
    print(f"\nMejora en MAE general: {mejoro_general}/{len(df)} barras")
    print(f"Mejora en MAE regimen alto (p95): {mejoro_p95}/{len(df)} barras")
    print("\nSi la mayoria mejora (sobre todo en regimen alto), vale la pena extender el "
          "tratamiento a las 8 barras completas. Si no, se descarta y se mantiene el "
          "recorte z>=4 actual en train.")

    df.to_csv("Scripts/resultado_piloto_outliers.csv", index=False)
    print("\nGuardado: Scripts/resultado_piloto_outliers.csv")


if __name__ == "__main__":
    main()
