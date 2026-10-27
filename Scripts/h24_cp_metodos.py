#!/usr/bin/env python3
"""
h24_cp_metodos.py - Compara los cuatro metodos de Conformal Prediction de la tesis
(Split CP, ACI, LW Split CP y LW-ACP, gamma=0.05) sobre el modelo day-ahead S+C-LN,
calibrando cada metodo por separado en cada paso del horizonte (h=1..24): calibracion
en validacion y evaluacion en prueba de ese mismo paso, contra la serie cruda,
excluyendo VENTANA_PRECIO_FALLA. 90% de cobertura nominal, el nivel principal.

Adaptacion online sin informacion futura. A 24 horas, el pronostico del paso h para la hora
t se emite en t - h, de modo que al emitirlo solo se conocen los resultados de las horas
anteriores a t - h. Actualizar alpha_t con el resultado de la hora t - 1, como a una hora,
usaria resultados que aun no ocurren. ACI y LW-ACP se evaluan por eso como en Zaffran et al.
(2022) para precios day-ahead: 24 series, una por hora del dia, cada una con su propio
alpha_t, que se actualiza con el resultado de la misma hora del dia anterior, ya observado al
emitir cualquier paso h <= 24 ("por hora"). Como verificacion se calcula tambien un unico
alpha_t global actualizado con el ultimo resultado ya observado al emitir, h horas antes
("retraso h"). Split CP y LW Split CP no tienen adaptacion online y no cambian.

Lee el cache de Scripts/extraer_predicciones_h24.py.

Uso:
    python Scripts/h24_cp_metodos.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA
from Modulos.Conformal_Prediction_Wrapper import _estimar_sigma_h
from cache_h24 import cargar_preds_sc
from cp_metricas_completas import evaluar_metodo, pinball, winkler, GAMMA

RUTA_CRUDA = "Datos/h/2020-2026.csv"
ALPHA = 0.10
METODOS = ["Split CP", "ACI", "LW Split CP", "LW-ACP"]
VERIFICACION = ["ACI (retraso h)", "LW-ACP (retraso h)"]


def marco(d, serie):
    m = pd.DataFrame({"fecha": pd.to_datetime(d["fechas"]), "y_pred": np.asarray(d["y_pred"])})
    m = m.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
    m["y_real"] = serie.reindex(m["fecha"]).values
    m = m.dropna(subset=["y_real"])
    ini, fin = VENTANA_PRECIO_FALLA
    return m.loc[~((m["fecha"] >= ini) & (m["fecha"] <= fin))].reset_index(drop=True)


def adaptativo(mv, mt, alpha, ponderado, esquema, paso, sigma_h=None):
    """ACI (ponderado=False) o LW-ACP (ponderado=True) sin informacion futura.
    esquema="por hora": un alpha_t por hora del dia, actualizado con el resultado de esa
    misma hora (Zaffran et al., 2022). esquema="retraso h": un alpha_t global que en el
    instante t incorpora el resultado de t - paso. sigma_h permite pasar otra escala horaria
    (por defecto, la de la version principal de LW-ACP)."""
    e = np.abs(mv["y_real"].values - mv["y_pred"].values)
    hc, ht = mv["fecha"].dt.hour.values, mt["fecha"].dt.hour.values
    if sigma_h is None:
        sigma_h = _estimar_sigma_h(e, hc) if ponderado else np.ones(24)
    scores = e / sigma_h[hc]
    n = len(scores)
    y, yhat = mt["y_real"].values, mt["y_pred"].values
    lower, upper, err = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
    alpha_t = np.full(24, alpha)
    for t in range(len(y)):
        if esquema == "por hora":
            k = ht[t]
        else:
            k = 0
            if t - paso >= 0:
                alpha_t[0] = np.clip(alpha_t[0] + GAMMA * (alpha - err[t - paso]), 0.001, 0.999)
        q = np.quantile(scores, min((1 - alpha_t[k]) * (n + 1) / n, 1.0), method="higher")
        lower[t] = max(yhat[t] - q * sigma_h[ht[t]], 0.0)
        upper[t] = yhat[t] + q * sigma_h[ht[t]]
        err[t] = 1 - int(lower[t] <= y[t] <= upper[t])
        if esquema == "por hora":
            alpha_t[k] = np.clip(alpha_t[k] + GAMMA * (alpha - err[t]), 0.001, 0.999)
    return {"y_test": y, "lower": lower, "upper": upper}


def main():
    preds = cargar_preds_sc("LN")
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])

    filas = []
    for barra, por_paso in preds.items():
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        for h, d in por_paso.items():
            mv, mt = marco(d["val"], serie), marco(d["test"], serie)
            res = {
                "Split CP": evaluar_metodo("Split CP", mv, mt, ALPHA),
                "LW Split CP": evaluar_metodo("LW Split CP", mv, mt, ALPHA),
                "ACI": adaptativo(mv, mt, ALPHA, False, "por hora", h),
                "LW-ACP": adaptativo(mv, mt, ALPHA, True, "por hora", h),
                "ACI (retraso h)": adaptativo(mv, mt, ALPHA, False, "retraso h", h),
                "LW-ACP (retraso h)": adaptativo(mv, mt, ALPHA, True, "retraso h", h),
            }
            for metodo, r in res.items():
                y, lo, up = r["y_test"], r["lower"], r["upper"]
                cov = float(np.mean((y >= lo) & (y <= up)))
                filas.append({"Barra": barra, "h": h, "Metodo": metodo, "Cobertura": cov,
                              "Desv": abs(cov - (1 - ALPHA)), "Ancho": float(np.mean(up - lo)),
                              "Winkler": winkler(y, lo, up, ALPHA),
                              "Pinball": (pinball(y, lo, ALPHA / 2) + pinball(y, up, 1 - ALPHA / 2)) / 2,
                              "Pinball_inf": pinball(y, lo, ALPHA / 2), "Pinball_sup": pinball(y, up, 1 - ALPHA / 2)})
        print(f"[{barra}] listo", flush=True)

    df = pd.DataFrame(filas)
    pd.set_option("display.width", 200)
    print("\n=== Promedio de las 8 barras y los 24 pasos, 90% nominal ===")
    print(df.groupby("Metodo")[["Cobertura", "Desv", "Ancho", "Winkler", "Pinball", "Pinball_inf", "Pinball_sup"]].mean()
          .loc[METODOS + VERIFICACION].round(4).to_string())

    principal = df[df.Metodo.isin(METODOS)]
    print("\n=== Por paso (promedio de 8 barras): Winkler por metodo ===")
    print(principal.pivot_table(index="h", columns="Metodo", values="Winkler").loc[:, METODOS].round(2).to_string())

    ganador = principal.loc[principal.groupby(["Barra", "h"])["Winkler"].idxmin()]
    print("\n=== Metodo con menor Winkler en cada combinacion barra-paso (de 192) ===")
    print(ganador["Metodo"].value_counts().to_string())
    print("\nPor barra:")
    print(ganador.groupby("Barra")["Metodo"].value_counts().unstack(fill_value=0).to_string())

    print("\n=== Promedio por barra (24 pasos) ===")
    print(principal.pivot_table(index="Barra", columns="Metodo", values=["Winkler", "Desv"]).round(3).to_string())

    df.to_csv("Scripts/resultado_h24_cp_metodos.csv", index=False)
    print("\nGuardado: Scripts/resultado_h24_cp_metodos.csv")


if __name__ == "__main__":
    main()
