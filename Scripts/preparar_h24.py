#!/usr/bin/env python3
"""
preparar_h24.py - Prepara los dataloaders de un experimento a horizonte day-ahead (h=24,
MIMO) a partir de una carpeta ya preprocesada a h=1: reusa su datasets_norm, scalers,
feature_config y tft_config, y solo cambia max_prediction_length a 24. No recalcula
features (ni Prophet, ni clima, ni posicion solar).

Sirve para cualquier estrategia y arquitectura (S+C o P+C, LN o DyT); los modelos a 24 horas
de la tesis se prepararon en dos tandas, la segunda con sufijo "_resto":

    # S+C, LN y DyT, cinco barras
    python Scripts/preparar_h24.py --origen Multi-Modelos_TFT/h/LN/pred_sol_clima \\
        --destino Multi-Modelos_TFT/h/LN/pred_sol_clima_h24 \\
        --barras ATACAMA CHARRUA P.MONTT P.AZUCAR TARAPACA
    (igual con DyT en vez de LN)

    # P+C (precios y residuos de Prophet), LN, las mismas cinco barras
    python Scripts/preparar_h24.py --origen Multi-Modelos_TFT/h/LN/pred_prophet_clima \\
        --destino Multi-Modelos_TFT/h/LN/pred_prophet_clima_h24 \\
        --barras ATACAMA CHARRUA P.MONTT P.AZUCAR TARAPACA

    # Las tres barras restantes, en carpetas "_resto" para no pisar las anteriores
    python Scripts/preparar_h24.py --origen Multi-Modelos_TFT/h/LN/pred_sol_clima \\
        --destino Multi-Modelos_TFT/h/LN/pred_sol_clima_h24_resto --barras CARDONES CRUCERO QUILLOTA
    (igual con DyT/pred_sol_clima y con LN/pred_prophet_clima)

Salida en --destino: dataloaders_precios.pkl, dataloaders_residuos.pkl, scalers.pkl (solo las
barras pedidas), feature_config.json y tft_config.json, listos para train_tft.py /
train_dyt.py en el cluster.
"""
import argparse
import io
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Preprocesamiento_Transformer import crear_dataloaders, guardar_experimento

HORIZONTE = 24


def main():
    parser = argparse.ArgumentParser(description="Dataloaders a h=24 desde una carpeta a h=1")
    parser.add_argument("--origen", required=True, help="carpeta a h=1 con datasets_norm.pkl y configs")
    parser.add_argument("--destino", required=True)
    parser.add_argument("--barras", nargs="+", required=True)
    args = parser.parse_args()

    with open(f"{args.origen}/datasets_norm.pkl", "rb") as f:
        datasets_norm = pickle.load(f)
    with open(f"{args.origen}/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(f"{args.origen}/feature_config.json", encoding="utf-8") as f:
        feature_config = json.load(f)
    with open(f"{args.origen}/tft_config.json", encoding="utf-8") as f:
        tft_config = json.load(f)

    print("known_reals:", feature_config["known_reals"])
    print("unknown_reals_price:", feature_config["unknown_reals_price"])
    print("unknown_reals_resid:", feature_config["unknown_reals_resid"])

    dataloaders_precios, dataloaders_residuos, _, _ = crear_dataloaders(
        lista_barras=args.barras,
        datasets_norm=datasets_norm,
        known_reals=feature_config["known_reals"],
        unknown_reals_price=feature_config["unknown_reals_price"],
        unknown_reals_resid=feature_config["unknown_reals_resid"],
        max_encoder_length=feature_config["max_encoder_length"],
        max_prediction_length=HORIZONTE,
        batch_size=feature_config["batch_size"],
    )

    guardar_experimento(args.destino, dataloaders_precios, dataloaders_residuos,
                        {b: scalers[b] for b in args.barras},
                        dict(feature_config, max_prediction_length=HORIZONTE, lista_barras=args.barras),
                        dict(tft_config, max_prediction_length=HORIZONTE))
    print(f"Guardado en: {args.destino}")


if __name__ == "__main__":
    main()
