#!/usr/bin/env python3
"""
evaluar_modelo_global.py - Evalua el modelo GLOBAL (un unico TFT entrenado sobre
las 8 barras pooled via serie_id, ver Scripts/preparar_modelo_global.py) y separa
las predicciones de vuelta por barra para poder comparar contra S+C-LN por-barra
(Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl).

El dataloader 'test' de GLOBAL mezcla las 8 barras en un solo TimeSeriesDataSet
(group_ids=["serie_id"]) y cubre train+val+test (igual que el dataloader 'test'
del pipeline por-barra, ver comentario de inferir_raw_modelo). Tras UNA sola
pasada de inferencia, raw.index trae la columna 'serie_id' ya decodificada a
texto por pytorch_forecasting (TimeSeriesDataSet.x_to_index), lo que permite
filtrar el resultado pooled a un dict por-barra sin volver a inferir, reusando
extraer_todas_predicciones_modelo/datasets_norm/scalers tal cual (son copias
identicas de los del pipeline por-barra, con su propio time_idx 0..N por barra).

Uso:
    python Scripts/evaluar_modelo_global.py [carpeta_modelo_global]
    (por defecto pred_sol_clima_global; para la variante sin outliers pasar
    Multi-Modelos_TFT/h/LN/pred_sol_clima_global_sin_outliers)
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import gc
import pickle
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modulos.TFT_Model import cargar_modelo_entrenado
from Modulos.Evaluacion_TFT import (
    extraer_todas_predicciones_modelo, calcular_metricas_predicciones,
    crear_tabla_resumen, inferir_raw_modelo,
)

CARPETA = sys.argv[1] if len(sys.argv) > 1 else "Multi-Modelos_TFT/h/LN/pred_sol_clima_global"
EXPERIMENTO = "Multi-TFT_Precios"
CARPETA_RESULTADOS = os.path.join(CARPETA, "Resultados")
HORIZONTE = 1
UMBRAL_DESFASE = 0.001




def filtrar_raw_por_barra(raw, mask):
    """Subconjunto de un resultado de inferencia pooled (raw), quedandose solo con
    las filas de una barra. mask: array booleano alineado fila a fila con
    raw.output/raw.y/raw.index (mismo orden que devuelve modelo.predict)."""
    prediction = raw.output.prediction if hasattr(raw.output, 'prediction') else raw.output[0]
    mask_t = torch.as_tensor(mask)

    output_filtrado = SimpleNamespace(prediction=prediction[mask_t])

    actuals = raw.y
    if isinstance(actuals, (list, tuple)):
        y_filtrado = (actuals[0][mask_t], actuals[1][mask_t] if actuals[1] is not None else None)
    else:
        y_filtrado = actuals[mask_t]

    index_filtrado = raw.index.iloc[mask].reset_index(drop=True)

    return SimpleNamespace(output=output_filtrado, y=y_filtrado, index=index_filtrado)


def main():
    print("=" * 80)
    print("EVALUACIÓN MODELO GLOBAL")
    print("=" * 80)

    with open(os.path.join(CARPETA, "dataloaders_precios.pkl"), "rb") as f:
        dataloaders_precios = pickle.load(f)
    with open(os.path.join(CARPETA, "datasets_norm.pkl"), "rb") as f:
        datasets_norm = pickle.load(f)
    with open(os.path.join(CARPETA, "scalers.pkl"), "rb") as f:
        diccionario_scalers = pickle.load(f)

    lista_barras = list(datasets_norm.keys())
    print(f"Barras a separar ({len(lista_barras)}): {lista_barras}")

    print("\nCargando modelo GLOBAL...")
    modelo, _ = cargar_modelo_entrenado("GLOBAL", EXPERIMENTO, CARPETA, verbose=False)
    if modelo is None:
        raise FileNotFoundError(f"No se encontro el modelo GLOBAL en {CARPETA}")

    print("Corriendo inferencia única sobre el dataloader pooled (8 barras)...")
    raw = inferir_raw_modelo(modelo, dataloaders_precios["GLOBAL"]["test"])
    serie_id_col = raw.index["serie_id"].values
    print(f"  Total de filas inferidas (8 barras juntas): {len(serie_id_col)}")

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    datos_evaluacion = {}

    for barra in lista_barras:
        print(f"\n[{barra}]")
        mask = (serie_id_col == barra)
        print(f"  filas de esta barra en el pooled: {int(mask.sum())}")
        raw_barra = filtrar_raw_por_barra(raw, mask)

        todas_precios = extraer_todas_predicciones_modelo(
            barra, modelo, None, datasets_norm, diccionario_scalers,
            'y_real', None, HORIZONTE, UMBRAL_DESFASE, raw=raw_barra,
        )

        datos_evaluacion[barra] = {}
        for conjunto in ['train', 'val', 'test']:
            dp = todas_precios[conjunto]
            mp = calcular_metricas_predicciones(
                dp['y_real'], dp['y_pred'], y_tft_reales=dp['y_tft_reales'],
                umbral_desfase=UMBRAL_DESFASE, verbose=True,
            )
            datos_evaluacion[barra][conjunto] = {
                'Precios': {'datos': dp, 'metricas': mp,
                            'n_validos': dp['n_validos'], 'n_total': dp['n_total']},
            }
            print(f"    n_validos/n_total: {dp['n_validos']}/{dp['n_total']}")

    del modelo, raw
    gc.collect()

    df_metricas = crear_tabla_resumen(datos_evaluacion)

    ruta_pkl = os.path.join(CARPETA_RESULTADOS, "eval_TFT_GLOBAL.pkl")
    with open(ruta_pkl, "wb") as f:
        pickle.dump((datos_evaluacion, df_metricas), f)
    ruta_csv = os.path.join(CARPETA_RESULTADOS, "metricas_TFT_GLOBAL.csv")
    df_metricas.to_csv(ruta_csv, index=False)

    print("\n" + "=" * 80)
    print(f"Guardado: {ruta_pkl}")
    print(f"Guardado: {ruta_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
