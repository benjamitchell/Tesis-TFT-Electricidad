#!/usr/bin/env python3
"""
inferencia_entradas_crudas.py - Predicciones de los modelos S+C-LN con recorte de outliers en
entrenamiento (por barra y global) con entradas crudas en validacion y prueba, usando los mismos
checkpoints, sin reentrenar.

Los dataloaders de pred_sol_clima (y del global, que reusa su datasets_norm) se armaron con las
entradas de validacion y prueba recortadas a 3 desviaciones estandar, mientras que el piloto sin
recorte y el global sin recorte las reciben crudas. Para que esas comparaciones aislen el efecto del
recorte en entrenamiento, aqui se reemplaza y_real de val/test por la serie cruda normalizada con el
mismo scaler (train queda igual) y se repite la inferencia. Lo consumen analisis_piloto_outliers.py
y analisis_global_sin_outliers.py.

Uso:
    python Scripts/inferencia_entradas_crudas.py            # modelo por barra
    python Scripts/inferencia_entradas_crudas.py --global   # modelo global
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Modulos.Preprocesamiento_Transformer import crear_dataloaders
from Modulos.TFT_Model import cargar_modelo_entrenado
from Modulos.Evaluacion_TFT import extraer_todas_predicciones_modelo, inferir_raw_modelo

BARRAS = ["ATACAMA", "CARDONES", "CHARRUA", "CRUCERO", "P.AZUCAR", "P.MONTT", "QUILLOTA", "TARAPACA"]
ORIGEN_BARRA = "Multi-Modelos_TFT/h/LN/pred_sol_clima"
ORIGEN_GLOBAL = "Multi-Modelos_TFT/h/LN/pred_sol_clima_global"
RUTA_BARRA = os.path.join(ORIGEN_BARRA, "Resultados", "preds_entradas_crudas.pkl")
RUTA_GLOBAL = os.path.join(ORIGEN_GLOBAL, "Resultados", "preds_entradas_crudas.pkl")
EXPERIMENTO = "Multi-TFT_Precios"
KNOWN = ["es_horario_verano", "elevacion_solar", "cos_elevacion"]
UNKNOWN = ["y_real", "temperatura", "humedad", "velocidad_viento", "precipitacion", "nubosidad"]


def datasets_con_entradas_crudas(origen):
    """datasets_norm de origen con y_real de val/test reemplazado por la serie cruda normalizada."""
    crudo = pd.read_csv("Datos/h/2020-2026.csv", sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    with open(os.path.join(origen, "datasets_norm.pkl"), "rb") as f:
        dsn = pickle.load(f)
    with open(os.path.join(origen, "scalers.pkl"), "rb") as f:
        scl = pickle.load(f)
    for barra in BARRAS:
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        s = scl[barra]["scalers_target"]["y_real"]
        for c in ("val", "test"):
            df = dsn[barra][c]
            raw = serie.reindex(pd.to_datetime(df["ds"])).values
            assert not np.isnan(raw).any(), (barra, c)
            df["y_real"] = s.transform(raw.reshape(-1, 1)).ravel()
    return dsn, scl


def por_barra():
    dsn, scl = datasets_con_entradas_crudas(ORIGEN_BARRA)
    dls, _, _, _ = crear_dataloaders(BARRAS, dsn, KNOWN, UNKNOWN, [], 168, 1, 128)
    cache = {}
    for barra in BARRAS:
        print(f"[inferencia] {barra}", flush=True)
        modelo, _ = cargar_modelo_entrenado(barra, EXPERIMENTO, ORIGEN_BARRA, verbose=False)
        res = extraer_todas_predicciones_modelo(barra, modelo, dls[barra]["test"], dsn, scl)
        cache[barra] = {k: (pd.to_datetime(res[k]["fechas"]), np.asarray(res[k]["y_pred"])) for k in ("val", "test")}
    return cache, RUTA_BARRA


def modelo_global():
    # mismo armado que preparar_modelo_global.py, solo para el conjunto de prueba (train+val+test)
    dsn, scl = datasets_con_entradas_crudas(ORIGEN_GLOBAL)
    trains, tvts = [], []
    for barra in BARRAS:
        tr, va, te = (dsn[barra][k].copy() for k in ("train", "val", "test"))
        for d in (tr, va, te):
            d["serie_id"] = barra
        tvt = pd.concat([tr, va, te], ignore_index=True).sort_values("ds").reset_index(drop=True)
        tvt["time_idx"] = list(tr["time_idx"]) + list(va["time_idx"]) + list(te["time_idx"])
        tvt["serie_id"] = barra
        trains.append(tr)
        tvts.append(tvt)
    base = TimeSeriesDataSet(
        pd.concat(trains, ignore_index=True), time_idx="time_idx", target="y_real", group_ids=["serie_id"],
        min_encoder_length=84, max_encoder_length=168, min_prediction_length=1, max_prediction_length=1,
        static_categoricals=["serie_id"], time_varying_known_reals=KNOWN, time_varying_unknown_reals=UNKNOWN,
        target_normalizer=None, add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True)
    test_ds = TimeSeriesDataSet.from_dataset(base, pd.concat(tvts, ignore_index=True), stop_randomization=True)
    dl = test_ds.to_dataloader(train=False, batch_size=1280, shuffle=False)

    sys.argv = sys.argv[:1]
    from evaluar_modelo_global import filtrar_raw_por_barra
    modelo, _ = cargar_modelo_entrenado("GLOBAL", EXPERIMENTO, ORIGEN_GLOBAL, verbose=False)
    raw = inferir_raw_modelo(modelo, dl)
    sid = raw.index["serie_id"].values
    cache = {}
    for barra in BARRAS:
        rb = filtrar_raw_por_barra(raw, sid == barra)
        res = extraer_todas_predicciones_modelo(barra, modelo, None, dsn, scl, "y_real", None, 1, 0.001, raw=rb)
        cache[barra] = {k: (pd.to_datetime(res[k]["fechas"]), np.asarray(res[k]["y_pred"])) for k in ("val", "test")}
    return cache, RUTA_GLOBAL


if __name__ == "__main__":
    cache, ruta = modelo_global() if "--global" in sys.argv else por_barra()
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(cache, f)
    print(f"Guardado: {ruta}")
