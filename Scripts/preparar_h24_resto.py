#!/usr/bin/env python3
"""
preparar_h24_resto.py - Prepara los dataloaders a horizonte day-ahead (h=24) para las tres
barras que faltaban (CARDONES, CRUCERO, QUILLOTA), con el mismo procedimiento que
Scripts/preparar_h24.py y Scripts/preparar_h24_pc.py:

  - S+C con LN y con DyT, desde pred_sol_clima/{LN,DyT}
  - P+C con LN (precios y residuos de Prophet), desde pred_prophet_clima/LN

Se guardan en carpetas "_resto" para no pisar las de las cinco barras ya entrenadas; las
configuraciones (hiperparametros, features, max_prediction_length=24) se copian de las
carpetas h24 existentes, de modo que los modelos nuevos son identicos en diseno.

Uso:
    python Scripts/preparar_h24_resto.py
"""
import io
import os
import sys
import json
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Preprocesamiento_Transformer import crear_dataloaders, guardar_experimento

BARRAS = ["CARDONES", "CRUCERO", "QUILLOTA"]
EXPERIMENTOS = [
    # (origen datasets_norm/scalers, carpeta h24 de referencia para configs, destino)
    ("Multi-Modelos_TFT/h/LN/pred_sol_clima", "Multi-Modelos_TFT/h/LN/pred_sol_clima_h24",
     "Multi-Modelos_TFT/h/LN/pred_sol_clima_h24_resto"),
    ("Multi-Modelos_TFT/h/DyT/pred_sol_clima", "Multi-Modelos_TFT/h/DyT/pred_sol_clima_h24",
     "Multi-Modelos_TFT/h/DyT/pred_sol_clima_h24_resto"),
    ("Multi-Modelos_TFT/h/LN/pred_prophet_clima", "Multi-Modelos_TFT/h/LN/pred_prophet_clima_h24",
     "Multi-Modelos_TFT/h/LN/pred_prophet_clima_h24_resto"),
]


def main():
    for origen, referencia, destino in EXPERIMENTOS:
        with open(f"{origen}/datasets_norm.pkl", "rb") as f:
            datasets_norm = pickle.load(f)
        with open(f"{origen}/scalers.pkl", "rb") as f:
            scalers = pickle.load(f)
        with open(f"{referencia}/feature_config.json", encoding="utf-8") as f:
            feature_config = json.load(f)
        with open(f"{referencia}/tft_config.json", encoding="utf-8") as f:
            tft_config = json.load(f)
        assert feature_config["max_prediction_length"] == 24 and tft_config["max_prediction_length"] == 24

        dl_precios, dl_residuos, _, _ = crear_dataloaders(
            lista_barras=BARRAS,
            datasets_norm=datasets_norm,
            known_reals=feature_config["known_reals"],
            unknown_reals_price=feature_config["unknown_reals_price"],
            unknown_reals_resid=feature_config["unknown_reals_resid"],
            max_encoder_length=feature_config["max_encoder_length"],
            max_prediction_length=24,
            batch_size=feature_config["batch_size"],
        )
        guardar_experimento(destino, dl_precios, dl_residuos, {b: scalers[b] for b in BARRAS},
                            dict(feature_config, lista_barras=BARRAS), tft_config)
        print(f"Guardado en: {destino}\n")


if __name__ == "__main__":
    main()
