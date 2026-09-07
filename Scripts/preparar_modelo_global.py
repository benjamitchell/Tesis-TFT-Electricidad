#!/usr/bin/env python3
"""
preparar_modelo_global.py - Arma los dataloaders de un modelo TFT GLOBAL: una
sola instancia entrenada sobre las ocho barras juntas (en vez de una instancia
por barra, como en el resto del pipeline), usando serie_id como covariable
estatica real (con las ocho barras presentes, no una sola como en el resto de
las carpetas de experimento).

Reusa el datasets_norm.pkl ya calculado de pred_sol_clima (S+C-LN, features
solares + clima, ya normalizado por barra) -- no recalcula preprocesamiento
ni features. Solo concatena las ocho barras en un unico DataFrame por split y
arma un unico TimeSeriesDataSet con group_ids=["serie_id"].

Uso:
    python Scripts/preparar_modelo_global.py
"""
import os
import sys
import json
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

ORIGEN = "Multi-Modelos_TFT/h/LN/pred_sol_clima"
CARPETA_SALIDA = "Multi-Modelos_TFT/h/LN/pred_sol_clima_global"
BARRAS = ["ATACAMA", "CARDONES", "CHARRUA", "CRUCERO", "P.AZUCAR", "P.MONTT", "QUILLOTA", "TARAPACA"]

var_clima = ["temperatura", "humedad", "velocidad_viento", "precipitacion", "nubosidad"]
var_solar = ["es_horario_verano", "elevacion_solar", "cos_elevacion"]
KNOWN_REALS = var_solar
UNKNOWN_REALS_PRICE = ["y_real"] + var_clima
MAX_ENCODER_LENGTH = 168
MAX_PREDICTION_LENGTH = 1
BATCH_SIZE = 128

print("=" * 80)
print("Preparando modelo GLOBAL (8 barras en un solo TimeSeriesDataSet)")
print("=" * 80)

with open(os.path.join(ORIGEN, "datasets_norm.pkl"), "rb") as f:
    datasets_norm = pickle.load(f)
with open(os.path.join(ORIGEN, "scalers.pkl"), "rb") as f:
    scalers = pickle.load(f)

trains, train_vals, train_val_tests = [], [], []

for barra in BARRAS:
    df_train = datasets_norm[barra]["train"].copy()
    df_val = datasets_norm[barra]["val"].copy()
    df_test = datasets_norm[barra]["test"].copy()

    df_train["serie_id"] = barra
    df_val["serie_id"] = barra
    df_test["serie_id"] = barra

    train_time_idx = df_train["time_idx"].values
    val_time_idx = df_val["time_idx"].values
    test_time_idx = df_test["time_idx"].values

    df_train_val = pd.concat([df_train, df_val], ignore_index=True).sort_values("ds").reset_index(drop=True)
    df_train_val["time_idx"] = list(train_time_idx) + list(val_time_idx)
    df_train_val["serie_id"] = barra

    df_train_val_test = pd.concat([df_train, df_val, df_test], ignore_index=True).sort_values("ds").reset_index(drop=True)
    df_train_val_test["time_idx"] = list(train_time_idx) + list(val_time_idx) + list(test_time_idx)
    df_train_val_test["serie_id"] = barra

    assert df_train_val["time_idx"].is_monotonic_increasing, f"{barra}: time_idx no continuo (train+val)"
    assert df_train_val_test["time_idx"].is_monotonic_increasing, f"{barra}: time_idx no continuo (train+val+test)"

    trains.append(df_train)
    train_vals.append(df_train_val)
    train_val_tests.append(df_train_val_test)

    print(f"  {barra}: train={len(df_train)} val={len(df_val)} test={len(df_test)}")

# Concatenar las 8 barras -- cada una conserva su propio time_idx (0..N independiente
# por barra) y su propio serie_id; TimeSeriesDataSet particiona ventanas por grupo
# (group_ids=["serie_id"]), no requiere que time_idx sea unico globalmente, solo
# monotono DENTRO de cada grupo (ya verificado arriba).
df_train_global = pd.concat(trains, ignore_index=True)
df_val_global = pd.concat(train_vals, ignore_index=True)
df_test_global = pd.concat(train_val_tests, ignore_index=True)

print(f"\nTotal filas -- train: {len(df_train_global)}  val(train+val): {len(df_val_global)}  "
      f"test(train+val+test): {len(df_test_global)}")

dataset_precios = TimeSeriesDataSet(
    df_train_global,
    time_idx="time_idx",
    target="y_real",
    group_ids=["serie_id"],
    min_encoder_length=MAX_ENCODER_LENGTH // 2,
    max_encoder_length=MAX_ENCODER_LENGTH,
    min_prediction_length=1,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    static_categoricals=["serie_id"],
    time_varying_known_reals=KNOWN_REALS,
    time_varying_unknown_reals=UNKNOWN_REALS_PRICE,
    target_normalizer=None,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

val_dataset_precios = TimeSeriesDataSet.from_dataset(dataset_precios, df_val_global, stop_randomization=True)
test_dataset_precios = TimeSeriesDataSet.from_dataset(dataset_precios, df_test_global, stop_randomization=True)

dataloaders_precios = {
    "GLOBAL": {
        "train": dataset_precios.to_dataloader(train=True, batch_size=BATCH_SIZE, shuffle=True),
        "val": val_dataset_precios.to_dataloader(train=False, batch_size=BATCH_SIZE * 10, shuffle=False),
        "test": test_dataset_precios.to_dataloader(train=False, batch_size=BATCH_SIZE * 10, shuffle=False),
    }
}
# train_tft.py carga dataloaders_residuos.pkl incondicionalmente aunque no se use
# (--solo-precios); se guarda una copia identica solo para satisfacer esa carga,
# nunca se usa para entrenar.
dataloaders_residuos = {"GLOBAL": dataloaders_precios["GLOBAL"]}

os.makedirs(CARPETA_SALIDA, exist_ok=True)
with open(os.path.join(CARPETA_SALIDA, "dataloaders_precios.pkl"), "wb") as f:
    pickle.dump(dataloaders_precios, f)
with open(os.path.join(CARPETA_SALIDA, "dataloaders_residuos.pkl"), "wb") as f:
    pickle.dump(dataloaders_residuos, f)
with open(os.path.join(CARPETA_SALIDA, "scalers.pkl"), "wb") as f:
    pickle.dump(scalers, f)
with open(os.path.join(CARPETA_SALIDA, "datasets_norm.pkl"), "wb") as f:
    pickle.dump(datasets_norm, f)

feature_config = {
    "known_reals": KNOWN_REALS, "unknown_reals_price": UNKNOWN_REALS_PRICE,
    "unknown_reals_resid": [], "max_encoder_length": MAX_ENCODER_LENGTH,
    "max_prediction_length": MAX_PREDICTION_LENGTH, "batch_size": BATCH_SIZE,
    "lista_barras": BARRAS, "nota": "Modelo GLOBAL: las 8 barras pooled en un unico TimeSeriesDataSet",
}
with open(os.path.join(CARPETA_SALIDA, "feature_config.json"), "w", encoding="utf-8") as f:
    json.dump(feature_config, f, indent=2, ensure_ascii=False)

tft_config_total = {
    "tft_config": {"lr": 0.001, "hidden_size": 32, "heads": 4, "dropout": 0.2, "cont_size": 8, "patience_lr": 5},
    "early_stop_config": {"min_delta": 0.001, "patience": 15},
    "epochs": 100, "max_encoder_length": MAX_ENCODER_LENGTH,
    "max_prediction_length": MAX_PREDICTION_LENGTH, "batch_size": BATCH_SIZE, "freq": "h",
}
with open(os.path.join(CARPETA_SALIDA, "tft_config.json"), "w", encoding="utf-8") as f:
    json.dump(tft_config_total, f, indent=2, ensure_ascii=False)

print(f"\nLISTO. Carpeta: {CARPETA_SALIDA}")
print("Entrenar con --barras GLOBAL --solo-precios (una sola corrida, no un array por barra).")
