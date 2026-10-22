#!/usr/bin/env python3
"""
piloto_lw_robusto_mejoras.py - Prueba dos extensiones de LW-ACP robusto
(Scripts/piloto_lw_escala_robusta.py) a 1 hora, 8 barras:

  1. Calibracion movil: la escala por hora y los scores de calibracion se recalculan una
     vez al dia con los errores de los ultimos W dias ya observados (al inicio, los de la
     validacion; luego se van sumando los de la prueba a medida que ocurren). "fija" es la
     version actual, calibrada una sola vez con la validacion; "acum" usa todos los errores
     observados hasta el momento.
  2. Intervalo asimetrico: el limite superior y el inferior se escalan por separado con los
     cuantiles alpha/2 y 1 - alpha/2 del residuo con signo (y - yhat) de cada hora, con
     score de no conformidad max(r / s_sup, -r / s_inf). El intervalo resultante es
     [yhat - q * s_inf, yhat + q * s_sup].

En ambos casos se mantiene la adaptacion online de alpha_t (gamma = 0.05) y el piso
absoluto c de la escala. W y c se eligen SOLO en validacion: se calibra con la primera mitad
de la validacion, se corre el metodo sobre la segunda mitad (actualizandose con sus errores
si la calibracion es movil) y se elige la combinacion de menor Winkler promedio en las 8
barras. Luego se evalua en prueba calibrando con toda la validacion.

Salida: Scripts/resultado_piloto_lw_robusto_mejoras.csv (prueba) y _seleccion.csv (validacion)
Uso:
    python Scripts/piloto_lw_robusto_mejoras.py
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
from cp_metricas_completas import marco, pinball, winkler, RUTA_EVAL, BARRAS, GAMMA
from piloto_lw_escala_robusta import HORAS_TRANSICION

RUTA_CRUDA = "Datos/h/2020-2026.csv"
PISOS = [5, 10, 15, 20, 30]
VENTANAS = ["fija", 30, 90, 180, "acum"]  # dias


def escalas(r, h, alpha, piso, asim):
    """Escala superior e inferior por hora a partir de residuos con signo r = y - yhat."""
    s_sup, s_inf = np.empty(24), np.empty(24)
    for k in range(24):
        rk = r[h == k]
        if asim:
            s_sup[k] = np.quantile(rk, 1 - alpha / 2)
            s_inf[k] = -np.quantile(rk, alpha / 2)
        else:
            s_sup[k] = s_inf[k] = np.quantile(np.abs(rk), 1 - alpha)
    return np.maximum(s_sup, piso), np.maximum(s_inf, piso)


def lw_acp_ext(mv, mt, alpha, piso, ventana="fija", asim=False):
    r_buf = list(mv["y_real"].values - mv["y_pred"].values)
    h_buf = list(mv["fecha"].dt.hour.values)
    y, yhat = mt["y_real"].values, mt["y_pred"].values
    h_test = mt["fecha"].dt.hour.values
    dia = mt["fecha"].dt.normalize().values
    lower, upper = np.empty(len(y)), np.empty(len(y))
    alpha_t, dia_actual = alpha, None
    for t in range(len(y)):
        if dia_actual is None or (ventana != "fija" and dia[t] != dia_actual):
            dia_actual = dia[t]
            r_w, h_w = np.asarray(r_buf), np.asarray(h_buf)
            if isinstance(ventana, int):
                r_w, h_w = r_w[-24 * ventana:], h_w[-24 * ventana:]
            s_sup, s_inf = escalas(r_w, h_w, alpha, piso, asim)
            scores = np.maximum(r_w / s_sup[h_w], -r_w / s_inf[h_w])
            n = len(scores)
        q_t = np.quantile(scores, min((1 - alpha_t) * (n + 1) / n, 1.0), method="higher")
        k = h_test[t]
        lower[t] = max(yhat[t] - q_t * s_inf[k], 0.0)
        upper[t] = yhat[t] + q_t * s_sup[k]
        err = 1 - int(lower[t] <= y[t] <= upper[t])
        alpha_t = np.clip(alpha_t + GAMMA * (alpha - err), 0.001, 0.999)
        r_buf.append(y[t] - yhat[t])
        h_buf.append(k)
    return {"y_test": y, "lower": lower, "upper": upper}


def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    marcos = {}
    for barra in BARRAS:
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        marcos[barra] = (marco(datos, barra, "val", serie), marco(datos, barra, "test", serie))

    filas, filas_sel = [], []
    for alpha in (0.10, 0.05):
        nivel = f"{int(round(100 * (1 - alpha)))}%"
        elegidos = {}
        for asim in (False, True):
            for ventana in VENTANAS:
                for piso in PISOS:
                    ws = []
                    for mv, _ in marcos.values():
                        mitad = len(mv) // 2
                        r = lw_acp_ext(mv.iloc[:mitad].reset_index(drop=True), mv.iloc[mitad:].reset_index(drop=True),
                                       alpha, piso, ventana, asim)
                        ws.append(winkler(r["y_test"], r["lower"], r["upper"], alpha))
                    filas_sel.append({"Nivel": nivel, "Asimetrico": asim, "Ventana": str(ventana), "Piso": piso,
                                      "Winkler_val": float(np.mean(ws))})
                    print(f"[{nivel}] asim={asim} ventana={ventana} piso={piso}: {np.mean(ws):.2f}", flush=True)
        sel = pd.DataFrame(filas_sel)
        sel = sel[sel.Nivel == nivel]
        for nombre, filtro in (("Robusto", (~sel.Asimetrico) & (sel.Ventana == "fija")),
                               ("+ ventana movil", (~sel.Asimetrico) & (sel.Ventana != "fija")),
                               ("+ asimetrico", sel.Asimetrico & (sel.Ventana == "fija")),
                               ("+ ambas", sel.Asimetrico & (sel.Ventana != "fija"))):
            mejor = sel[filtro].sort_values("Winkler_val").iloc[0]
            v = mejor.Ventana if mejor.Ventana in ("fija", "acum") else int(mejor.Ventana)
            elegidos[nombre] = (v, float(mejor.Piso), bool(mejor.Asimetrico))
            print(f"[{nivel}] {nombre}: ventana={v} piso={mejor.Piso} (Winkler val {mejor.Winkler_val:.2f})", flush=True)

        for barra, (mv, mt) in marcos.items():
            horas, meses = mt["fecha"].dt.hour.values, mt["fecha"].dt.month.values
            for nombre, (v, piso, asim) in elegidos.items():
                r = lw_acp_ext(mv, mt, alpha, piso, v, asim)
                y, lo, up = r["y_test"], r["lower"], r["upper"]
                dentro = (y >= lo) & (y <= up)
                filas.append({"Nivel": nivel, "Barra": barra, "Metodo": nombre, "Ventana": str(v), "Piso": piso,
                              "Cobertura": dentro.mean(), "Desv": abs(dentro.mean() - (1 - alpha)),
                              "Ancho": np.mean(up - lo), "Winkler": winkler(y, lo, up, alpha),
                              "Pinball": (pinball(y, lo, alpha / 2) + pinball(y, up, 1 - alpha / 2)) / 2,
                              **{f"dentro_h{k}": int(dentro[horas == k].sum()) for k in range(24)},
                              **{f"n_h{k}": int((horas == k).sum()) for k in range(24)},
                              **{f"dentro_m{m}": int(dentro[meses == m].sum()) for m in range(1, 13)},
                              **{f"n_m{m}": int((meses == m).sum()) for m in range(1, 13)}})
            print(f"[{nivel}] {barra} evaluada en prueba", flush=True)

    df = pd.DataFrame(filas)
    orden = ["Robusto", "+ ventana movil", "+ asimetrico", "+ ambas"]
    pd.set_option("display.width", 220)
    for nivel in ("90%", "95%"):
        d = df[df.Nivel == nivel]
        res = d.groupby("Metodo")[["Cobertura", "Desv", "Ancho", "Winkler", "Pinball"]].mean().loc[orden]
        agg = d.groupby("Metodo").sum(numeric_only=True)
        res["SD_hora"] = [np.std([agg.loc[m, f"dentro_h{k}"] / agg.loc[m, f"n_h{k}"] for k in range(24)]) for m in orden]
        res["Cob_trans"] = [sum(agg.loc[m, f"dentro_h{k}"] for k in HORAS_TRANSICION)
                            / sum(agg.loc[m, f"n_h{k}"] for k in HORAS_TRANSICION) for m in orden]
        res["SD_mes"] = [np.std([agg.loc[m, f"dentro_m{k}"] / agg.loc[m, f"n_m{k}"] for k in range(1, 13)]) for m in orden]
        print(f"\n=== {nivel}: prueba, promedio de las 8 barras ===")
        print(res.round(4).to_string())
        w = d.pivot(index="Barra", columns="Metodo", values="Winkler")[orden]
        print(w.round(2).to_string())
        for m in orden[1:]:
            print(f"{m}: mejor que Robusto en {int((w[m] < w['Robusto']).sum())} de 8 barras")
    df.to_csv("Scripts/resultado_piloto_lw_robusto_mejoras.csv", index=False)
    pd.DataFrame(filas_sel).to_csv("Scripts/resultado_piloto_lw_robusto_mejoras_seleccion.csv", index=False)
    print("\nGuardado: Scripts/resultado_piloto_lw_robusto_mejoras.csv y _seleccion.csv")


if __name__ == "__main__":
    main()
