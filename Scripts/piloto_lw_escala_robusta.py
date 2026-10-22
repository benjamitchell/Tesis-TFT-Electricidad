#!/usr/bin/env python3
"""
piloto_lw_escala_robusta.py - Variante de LW-ACP con una escala horaria robusta a colas
pesadas, en vez de la desviacion estandar con piso de la version principal.

En la version principal (Modulos/Conformal_Prediction_Wrapper._estimar_sigma_h), la escala
de cada hora es la desviacion estandar del error de calibracion de esa hora, con un piso
igual a la desviacion global. Con colas pesadas, unos pocos errores grandes inflan la
desviacion estandar (Atacama a las 15 h: P90 del error 0.05 USD/MWh, desviacion 16.3), y el
piso impide que las horas tranquilas tengan una escala menor que la global: al 90%, el
intervalo sobrecubre al mediodia y subcubre en las transiciones solares.

La variante usa como escala de cada hora el cuantil (1 - alpha) del error absoluto de
calibracion de esa hora, con un piso absoluto. Con esa escala, en cada hora una fraccion
alpha de los scores normalizados supera 1, de modo que el metodo se acerca a calibrar cada
hora por separado (Mondrian conformal prediction), mas la adaptacion online de alpha_t de
LW-ACP (gamma=0.05).

El piso se elige SOLO en validacion: se calibra con la primera mitad de la validacion, se
evalua el Winkler promedio de las 8 barras en la segunda mitad para cada piso de la grilla y
se elige el menor. Luego se calibra con toda la validacion y se evalua en prueba, contra
Split CP, ACI y LW-ACP (version principal), con las metricas de la tesis mas la cobertura
por hora (desviacion estandar entre horas y cobertura en las horas de transicion solar).

Salida: Scripts/resultado_piloto_lw_escala_robusta.csv (por nivel, barra y metodo) y
        Scripts/resultado_piloto_lw_escala_robusta_piso.csv (Winkler de validacion por piso).
Uso:
    python Scripts/piloto_lw_escala_robusta.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Conformal_Prediction_Wrapper import (split_cp_from_calibration, aci_from_calibration,
                                                  _estimar_sigma_h)
from cp_metricas_completas import marco, pinball, winkler, RUTA_EVAL, BARRAS, GAMMA

RUTA_CRUDA = "Datos/h/2020-2026.csv"
HORAS_TRANSICION = [7, 8, 19, 20, 21]
PISOS = [0.5, 1, 2, 5, 10, 15, 20, 30, 50, 100]  # USD/MWh
METODOS = ["Split CP", "ACI", "LW-ACP", "LW-ACP robusto"]


def escala_horaria(errores_cal, horas_cal, alpha, piso):
    """Escala por hora: cuantil (1 - alpha) del error absoluto de la hora (piso=None:
    desviacion estandar con piso global, la version principal de LW-ACP)."""
    if piso is None:
        return _estimar_sigma_h(errores_cal, horas_cal)
    s = np.array([np.quantile(errores_cal[horas_cal == h], 1 - alpha) for h in range(24)])
    return np.maximum(s, piso)


def lw_acp(mv, mt, alpha, piso=None):
    """LW-ACP con la escala horaria indicada; mismo bucle online que
    Modulos/Conformal_Prediction_Wrapper.lw_acp_online_from_calibration."""
    e_cal = np.abs(mv["y_real"].values - mv["y_pred"].values)
    h_cal = mv["fecha"].dt.hour.values
    h_test = mt["fecha"].dt.hour.values
    sigma_h = escala_horaria(e_cal, h_cal, alpha, piso)
    scores = e_cal / sigma_h[h_cal]
    n_cal = len(scores)
    y, yhat = mt["y_real"].values, mt["y_pred"].values
    lower, upper = np.empty(len(y)), np.empty(len(y))
    alpha_t = alpha
    for t in range(len(y)):
        q_t = np.quantile(scores, min((1 - alpha_t) * (n_cal + 1) / n_cal, 1.0), method="higher")
        lower[t] = max(yhat[t] - q_t * sigma_h[h_test[t]], 0.0)
        upper[t] = yhat[t] + q_t * sigma_h[h_test[t]]
        err_t = 1 - int(lower[t] <= y[t] <= upper[t])
        alpha_t = np.clip(alpha_t + GAMMA * (alpha - err_t), 0.001, 0.999)
    return {"y_test": y, "lower": lower, "upper": upper}


def elegir_piso(marcos, alpha):
    """Winkler promedio en la segunda mitad de validacion, calibrando con la primera."""
    w = {}
    for piso in PISOS:
        ws = []
        for mv, _ in marcos.values():
            mitad = len(mv) // 2
            r = lw_acp(mv.iloc[:mitad].reset_index(drop=True), mv.iloc[mitad:].reset_index(drop=True), alpha, piso)
            ws.append(winkler(r["y_test"], r["lower"], r["upper"], alpha))
        w[piso] = float(np.mean(ws))
    return min(w, key=w.get), w


def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    marcos = {}
    for barra in BARRAS:
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        marcos[barra] = (marco(datos, barra, "val", serie), marco(datos, barra, "test", serie))

    filas, filas_piso, filas_hora = [], [], []
    for alpha in (0.10, 0.05):
        nivel = f"{int(round(100 * (1 - alpha)))}%"
        piso, w_val = elegir_piso(marcos, alpha)
        filas_piso += [{"Nivel": nivel, "Piso": p, "Winkler_val": v, "Elegido": p == piso} for p, v in w_val.items()]
        print(f"[{nivel}] piso elegido en validacion: {piso} USD/MWh", flush=True)
        for barra, (mv, mt) in marcos.items():
            horas = mt["fecha"].dt.hour.values
            res = {
                "Split CP": split_cp_from_calibration(mv.y_real.values, mv.y_pred.values,
                                                      mt.y_real.values, mt.y_pred.values, alpha=alpha),
                "ACI": aci_from_calibration(mv.y_real.values, mv.y_pred.values, mt.y_real.values,
                                            mt.y_pred.values, alpha=alpha, gamma=GAMMA),
                "LW-ACP": lw_acp(mv, mt, alpha),
                "LW-ACP robusto": lw_acp(mv, mt, alpha, piso),
            }
            for nombre, r in res.items():
                y, lo, up = r["y_test"], r["lower"], r["upper"]
                dentro = (y >= lo) & (y <= up)
                for h in range(24):
                    filas_hora.append({"Nivel": nivel, "Metodo": nombre, "Hora": h,
                                       "dentro": int(dentro[horas == h].sum()), "n": int((horas == h).sum())})
                filas.append({
                    "Nivel": nivel, "Barra": barra, "Metodo": nombre, "Piso": piso if nombre == "LW-ACP robusto" else np.nan,
                    "Cobertura": dentro.mean(), "Desv": abs(dentro.mean() - (1 - alpha)),
                    "Ancho": np.mean(up - lo), "Winkler": winkler(y, lo, up, alpha),
                    "Pinball": (pinball(y, lo, alpha / 2) + pinball(y, up, 1 - alpha / 2)) / 2,
                    "SD_cob_hora": pd.Series(dentro).groupby(horas).mean().std(),
                    "Cob_transicion": dentro[np.isin(horas, HORAS_TRANSICION)].mean(),
                    "Ancho_10_16h": np.mean((up - lo)[(horas >= 10) & (horas <= 16)]),
                })

    df = pd.DataFrame(filas)
    pd.set_option("display.width", 220)
    # Cobertura por hora agregando las 8 barras, con la misma definicion que
    # Scripts/cp_hora_mes_90.py (desviacion estandar poblacional entre las 24 horas)
    hh = pd.DataFrame(filas_hora).groupby(["Nivel", "Metodo", "Hora"])[["dentro", "n"]].sum()
    hh["cob"] = hh["dentro"] / hh["n"]
    print("\n=== Cobertura por hora agregando las 8 barras ===")
    for (nivel, metodo), g in hh.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        trans = g.loc[HORAS_TRANSICION, "dentro"].sum() / g.loc[HORAS_TRANSICION, "n"].sum()
        print(f"{nivel} {metodo:15s} SD_hora={np.std(g['cob'].values):.4f}  cob_transicion={trans:.4f}")
    for nivel in ("90%", "95%"):
        d = df[df.Nivel == nivel]
        print(f"\n=== {nivel}: promedio de las 8 barras ===")
        print(d.groupby("Metodo")[["Cobertura", "Desv", "Ancho", "Winkler", "Pinball", "SD_cob_hora",
                                   "Cob_transicion", "Ancho_10_16h"]].mean().loc[METODOS].round(4).to_string())
        w = d.pivot(index="Barra", columns="Metodo", values="Winkler")[METODOS]
        print(f"\nWinkler por barra ({nivel}); robusto mejor que LW-ACP en "
              f"{int((w['LW-ACP robusto'] < w['LW-ACP']).sum())} de 8:")
        print(w.round(2).to_string())
    df.to_csv("Scripts/resultado_piloto_lw_escala_robusta.csv", index=False)
    pd.DataFrame(filas_piso).to_csv("Scripts/resultado_piloto_lw_escala_robusta_piso.csv", index=False)
    print("\nGuardado: Scripts/resultado_piloto_lw_escala_robusta.csv y _piso.csv")


if __name__ == "__main__":
    main()
