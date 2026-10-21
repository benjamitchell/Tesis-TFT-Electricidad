"""
cache_h24.py - Carga las predicciones por paso de los modelos day-ahead (h=24) uniendo
las dos tandas de entrenamiento: las cinco barras originales (carpetas *_h24) y las tres
que se entrenaron despues (carpetas *_h24_resto: CARDONES, CRUCERO, QUILLOTA).

Los caches los genera Scripts/extraer_predicciones_h24.py (S+C) y Scripts/h24_pc_vs_sc.py
(P+C). Si una tanda todavia no tiene cache, se omite con un aviso.
"""
import os
import pickle

CARPETAS_SC = ["Multi-Modelos_TFT/h/{arq}/pred_sol_clima_h24", "Multi-Modelos_TFT/h/{arq}/pred_sol_clima_h24_resto"]
CARPETAS_PC = ["Multi-Modelos_TFT/h/LN/pred_prophet_clima_h24", "Multi-Modelos_TFT/h/LN/pred_prophet_clima_h24_resto"]
NOMBRE_CACHE = os.path.join("Resultados", "predicciones_por_paso.pkl")


def _unir(carpetas):
    preds = {}
    for carpeta in carpetas:
        ruta = os.path.join(carpeta, NOMBRE_CACHE)
        if not os.path.exists(ruta):
            print(f"  [AVISO] sin cache: {ruta}")
            continue
        with open(ruta, "rb") as f:
            preds.update(pickle.load(f))
    return preds


def cargar_preds_sc(arq="LN"):
    """dict[barra][h]["val"|"test"] = {"fechas", "y_pred"} del modelo S+C h24."""
    return _unir([c.format(arq=arq) for c in CARPETAS_SC])


def cargar_preds_pc():
    """dict[barra][h][conjunto]["precios"|"residuos"] = Series del modelo P+C h24."""
    return _unir(CARPETAS_PC)
