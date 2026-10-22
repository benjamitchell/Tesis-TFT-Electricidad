#!/usr/bin/env python3
"""
extraer_predicciones_h24.py - Extrae y guarda en cache las predicciones por paso (h=1..24)
de los modelos day-ahead S+C, LN y DyT, en validacion y prueba, para las 8 barras con
modelo h24 (las 5 originales en *_h24 y las 3 de train_h24_resto.sh en *_h24_resto). Una sola inferencia por modelo (mismo patron que Scripts/evaluar_h24.py).

Las pruebas de la Seccion "conclusiones a 24 horas" (heterocedasticidad, Conformal
Prediction y LN vs DyT) leen este cache en vez de re-inferir.

Salida: Multi-Modelos_TFT/h/{arq}/pred_sol_clima_h24{,_resto}/Resultados/predicciones_por_paso.pkl
        (Scripts/cache_h24.py las une)
        dict[barra][h]["val"|"test"] = {"fechas": DatetimeIndex, "y_pred": ndarray}

Uso:
    python Scripts/extraer_predicciones_h24.py
"""
import os
import sys
import gc
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluar_h24 import EXPERIMENTO_PRECIOS
from Modulos.Evaluacion_TFT import extraer_todas_predicciones_modelo, inferir_raw_modelo
from Modulos.TFT_Model import cargar_modelo_entrenado

# (sufijo de carpeta, barras)
TANDAS = [("_h24", ["ATACAMA", "CHARRUA", "P.AZUCAR", "P.MONTT", "TARAPACA"]),
          ("_h24_resto", ["CARDONES", "CRUCERO", "QUILLOTA"])]
ARQUITECTURAS = ["LN", "DyT"]
N_HORIZONTES = 24


def main():
    for arq, (sufijo, barras) in ((a, t) for a in ARQUITECTURAS for t in TANDAS):
        carpeta = f"Multi-Modelos_TFT/h/{arq}/pred_sol_clima{sufijo}"
        carpeta_norm = f"Multi-Modelos_TFT/h/{arq}/pred_sol_clima"
        salida = os.path.join(carpeta, "Resultados", "predicciones_por_paso.pkl")
        if os.path.exists(salida):
            print(f"[{arq}] ya existe {salida}, se omite")
            continue
        if not all(os.path.exists(os.path.join(carpeta, "Multi-TFT_Precios" if arq == "LN" else "Precios", b))
                   for b in barras):
            print(f"[{arq}] faltan modelos en {carpeta}, se omite")
            continue

        with open(os.path.join(carpeta, "dataloaders_precios.pkl"), "rb") as f:
            dataloaders = pickle.load(f)
        with open(os.path.join(carpeta_norm, "datasets_norm.pkl"), "rb") as f:
            datasets_norm = pickle.load(f)
        with open(os.path.join(carpeta, "scalers.pkl"), "rb") as f:
            scalers = pickle.load(f)

        resultado = {}
        for barra in barras:
            modelo, _ = cargar_modelo_entrenado(barra, EXPERIMENTO_PRECIOS[arq], carpeta,
                                                dyt=arq == "DyT", verbose=False)
            raw = inferir_raw_modelo(modelo, dataloaders[barra]["test"])
            resultado[barra] = {}
            for h in range(1, N_HORIZONTES + 1):
                r = extraer_todas_predicciones_modelo(barra, modelo, dataloaders[barra]["test"],
                                                      datasets_norm, scalers, horizonte=h, raw=raw)
                resultado[barra][h] = {c: {"fechas": r[c]["fechas"], "y_pred": r[c]["y_pred"]}
                                       for c in ("val", "test")}
            print(f"[{arq}] {barra} listo", flush=True)
            del raw, modelo
            gc.collect()

        os.makedirs(os.path.dirname(salida), exist_ok=True)
        with open(salida, "wb") as f:
            pickle.dump(resultado, f)
        print(f"[{arq}] guardado {salida}")


if __name__ == "__main__":
    main()
