#!/usr/bin/env python3
"""
cqr_niveles.py - CQR (cross-conformal, misma logica que
Modulos/Conformal_Prediction_Wrapper.cqr_implementation) sobre la prediccion cruda de
S+C-LN, serie cruda, a 95%, 90% y 50%, para completar la comparacion de metodos de CP.
Incluye ademas la regresion cuantilica sin conformalizar de
Modulos/Uncertainty_Comparison.quantile_regression, para verificar cual de las dos
implementaciones reproduce el resultado ya reportado al 95%.

Uso:
    python Scripts/cqr_niveles.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cp_metricas_completas import marco, pinball, winkler

RUTA_EVAL = "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"
BARRAS = ["ATACAMA", "CARDONES", "CHARRUA", "CRUCERO", "P.AZUCAR", "P.MONTT", "QUILLOTA", "TARAPACA"]
NIVELES = {0.05: "95%", 0.10: "90%", 0.50: "50%"}
N_FOLDS, QR_L1 = 5, 0.01



def features(yhat, fechas):
    r = 2 * np.pi * pd.DatetimeIndex(fechas).hour / 24
    return np.column_stack([yhat, np.sin(r), np.cos(r)])


def ajustar_qr(X, y, alpha):
    lo = QuantileRegressor(quantile=alpha / 2, alpha=QR_L1, solver="highs").fit(X, y)
    hi = QuantileRegressor(quantile=1 - alpha / 2, alpha=QR_L1, solver="highs").fit(X, y)
    return lo, hi


def cqr(mv, mt, alpha):
    X_val, X_test = features(mv["y_pred"].values, mv["fecha"]), features(mt["y_pred"].values, mt["fecha"])
    y_val = mv["y_real"].values
    scores = []
    for tr, ca in KFold(n_splits=N_FOLDS, shuffle=False).split(X_val):
        lo, hi = ajustar_qr(X_val[tr], y_val[tr], alpha)
        scores.append(np.maximum(lo.predict(X_val[ca]) - y_val[ca], y_val[ca] - hi.predict(X_val[ca])))
    s = np.concatenate(scores)
    q = float(np.quantile(s, np.ceil((1 - alpha) * (len(s) + 1)) / (len(s) + 1), method="higher"))
    lo, hi = ajustar_qr(X_val, y_val, alpha)
    lower, upper = np.maximum(lo.predict(X_test) - q, 0.0), hi.predict(X_test) + q
    swap = lower > upper
    lower[swap], upper[swap] = upper[swap], lower[swap]
    return lower, upper


def qr_simple(mv, mt, alpha):
    lo, hi = ajustar_qr(mv["y_pred"].values.reshape(-1, 1), mv["y_real"].values, alpha)
    X_test = mt["y_pred"].values.reshape(-1, 1)
    lower, upper = np.maximum(lo.predict(X_test), 0.0), hi.predict(X_test)
    malo = upper < lower
    upper[malo] = lower[malo] + 1.0
    return lower, upper



def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    df_crudo = pd.read_csv("Datos/h/2020-2026.csv", sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])

    filas = []
    for barra in BARRAS:
        serie = df_crudo.loc[df_crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        mv, mt = marco(datos, barra, "val", serie), marco(datos, barra, "test", serie)
        y = mt["y_real"].values
        for alpha, nivel in NIVELES.items():
            for metodo, fn in (("CQR", cqr), ("QR sin conformalizar", qr_simple)):
                lo, up = fn(mv, mt, alpha)
                cov = float(np.mean((y >= lo) & (y <= up)))
                winkler_ = winkler(y, lo, up, alpha)
                pb = (pinball(y, lo, alpha / 2) + pinball(y, up, 1 - alpha / 2)) / 2
                filas.append({"Barra": barra, "Nivel": nivel, "Metodo": metodo, "Cobertura": round(cov, 4),
                              "Desv": abs(cov - (1 - alpha)), "Ancho": round(float(np.mean(up - lo)), 2),
                              "Winkler": round(winkler_, 2), "Pinball": round(pb, 3),
                              "Pinball_inf": round(pinball(y, lo, alpha / 2), 3),
                              "Pinball_sup": round(pinball(y, up, 1 - alpha / 2), 3)})
        print(f"[{barra}] ok")

    df = pd.DataFrame(filas)
    pd.set_option("display.width", 200)
    print(df.groupby(["Metodo", "Nivel"])[["Cobertura", "Desv", "Ancho", "Winkler", "Pinball", "Pinball_inf", "Pinball_sup"]].mean().round(4).to_string())
    print()
    print(df[df.Metodo == "CQR"].pivot(index="Barra", columns="Nivel", values=["Cobertura", "Ancho"]).to_string())
    df.to_csv("Scripts/resultado_cqr_niveles.csv", index=False)
    print("\nGuardado: Scripts/resultado_cqr_niveles.csv")


if __name__ == "__main__":
    main()
