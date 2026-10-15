#!/usr/bin/env python3
"""
preparar_h24_pc.py - Prepara los dataloaders de la estrategia P+C (Prophet + clima) a
horizonte day-ahead (h=24, MIMO), LN, para las mismas 5 barras del modelo S+C a 24 horas,
de modo que S+C y P+C se puedan comparar a igual horizonte.

Mismo procedimiento que Scripts/preparar_h24.py: reusa datasets_norm, scalers y
feature_config de la carpeta P+C a h=1 (pred_prophet_clima) y solo cambia
max_prediction_length a 24. Las componentes de Prophet (trend, yearly, weekly, daily,
is_holiday) son conocidas a futuro, igual que la posicion solar en S+C.

Salida: Multi-Modelos_TFT/h/LN/pred_prophet_clima_h24/ (precios y residuos)

Uso:
    python Scripts/preparar_h24_pc.py
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

BARRAS = ["ATACAMA", "CHARRUA", "P.MONTT", "P.AZUCAR", "TARAPACA"]
ORIGEN = "Multi-Modelos_TFT/h/LN/pred_prophet_clima"
DESTINO = "Multi-Modelos_TFT/h/LN/pred_prophet_clima_h24"


def main():
    with open(f"{ORIGEN}/datasets_norm.pkl", "rb") as f:
        datasets_norm = pickle.load(f)
    with open(f"{ORIGEN}/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(f"{ORIGEN}/feature_config.json", encoding="utf-8") as f:
        feature_config = json.load(f)
    with open(f"{ORIGEN}/tft_config.json", encoding="utf-8") as f:
        tft_config = json.load(f)

    print("known_reals:", feature_config["known_reals"])
    print("unknown_reals_price:", feature_config["unknown_reals_price"])
    print("unknown_reals_resid:", feature_config["unknown_reals_resid"])

    dataloaders_precios, dataloaders_residuos, _, _ = crear_dataloaders(
        lista_barras=BARRAS,
        datasets_norm=datasets_norm,
        known_reals=feature_config["known_reals"],
        unknown_reals_price=feature_config["unknown_reals_price"],
        unknown_reals_resid=feature_config["unknown_reals_resid"],
        max_encoder_length=feature_config["max_encoder_length"],
        max_prediction_length=24,
        batch_size=feature_config["batch_size"],
    )

    feature_config_h24 = dict(feature_config, max_prediction_length=24, lista_barras=BARRAS)
    tft_config_h24 = dict(tft_config, max_prediction_length=24)
    guardar_experimento(DESTINO, dataloaders_precios, dataloaders_residuos,
                        {b: scalers[b] for b in BARRAS}, feature_config_h24, tft_config_h24)
    print(f"Guardado en: {DESTINO}")


if __name__ == "__main__":
    main()
