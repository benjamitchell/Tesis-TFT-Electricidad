#!/usr/bin/env python3
"""
stacking_validacion_tablas.py - Recalcula, con los pesos del Stacking ajustados en validacion,
todas las cifras de P+C a una hora que dependen de esos pesos (Capitulo 4): mejor estrategia por
barra y arquitectura, pesos, MAE de prueba, mejora respecto a Prophet y al TFT de precios,
comparacion entre metodos de optimizacion, y P+C contra S+C con Diebold-Mariano.

Protocolo: los pesos se ajustan sobre la PRIMERA mitad de la validacion del TFT (donde sus
predicciones son fuera de muestra) y la mejor estrategia se elige por MAE en la SEGUNDA mitad; la
prueba se usa solo para evaluar, contra la serie cruda y excluyendo VENTANA_PRECIO_FALLA. Las
predicciones de Prophet y de los TFT son las ya guardadas por el pipeline (sin reentrenar).

DeltaProphet se calcula respecto a Prophet truncado en 0 (el mismo de la Tabla de Prophet).

Uso:
    python Scripts/stacking_validacion_tablas.py
"""
import os
import sys
import pickle
import time
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Stacking_Optimization_TFT import (grid_search_modelos, scipy_minimize_nelder_mead, scipy_minimize_slsqp,
                                               scipy_minimize_lbfgsb, differential_evolution_opt, ridge_regression,
                                               random_search)
from Modulos.Comparativa import diebold_mariano_desde_errores
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA

RUTA_CRUDA = "Datos/h/2020-2026.csv"
ARQS = {"LN": ("Multi-Modelos_TFT/h/LN/pred_prophet_clima/Resultados/estrategias_prophet_LN_raw.pkl",
               "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"),
        "DyT": ("Multi-Modelos_TFT/h/DyT/pred_prophet_clima/Resultados/estrategias_prophet_DyT_raw.pkl",
                "Multi-Modelos_TFT/h/DyT/pred_sol_clima/Resultados/eval_TFT_DyT.pkl")}
ESTRATEGIAS = ["Prophet solo", "TFT precios solo", "Prophet + TFT precios", "Prophet + residuos (directo)",
               "Prophet + residuos (opt)", "(Prophet + residuos) + TFT precios"]
NOMBRES = {"ATACAMA": "Atacama", "CARDONES": "Cardones", "CHARRUA": "Charrúa", "CRUCERO": "Crucero",
           "P.AZUCAR": "Pan de Azúcar", "P.MONTT": "Puerto Montt", "QUILLOTA": "Quillota", "TARAPACA": "Tarapacá"}
ORDEN = ["ATACAMA", "CARDONES", "CHARRUA", "CRUCERO", "P.AZUCAR", "P.MONTT", "QUILLOTA", "TARAPACA"]
OPTIMIZADORES = {
    "Grid_21": lambda y, a, b: grid_search_modelos(y, a, b, resolution=21),
    "Grid_41": lambda y, a, b: grid_search_modelos(y, a, b, resolution=41),
    "Grid_101": lambda y, a, b: grid_search_modelos(y, a, b, resolution=101),
    "Nelder_Mead": scipy_minimize_nelder_mead, "SLSQP": scipy_minimize_slsqp, "L_BFGS_B": scipy_minimize_lbfgsb,
    "Diff_Evolution": differential_evolution_opt, "Random_1000": lambda y, a, b: random_search(y, a, b, n_iter=1000),
}


def mae(y, p):
    return float(np.nanmean(np.abs(y - p)))


def predicciones(P, T, R, w):
    w_pt, w_pr, w_prt = w
    return {"Prophet solo": P, "TFT precios solo": T,
            "Prophet + TFT precios": w_pt * P + (1 - w_pt) * T,
            "Prophet + residuos (directo)": P + R,
            "Prophet + residuos (opt)": w_pr * P + (1 - w_pr) * R,
            "(Prophet + residuos) + TFT precios": w_prt * (P + R) + (1 - w_prt) * T}


def main():
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    ini, fin = VENTANA_PRECIO_FALLA
    filas, filas_opt = [], []
    for arq, (ruta_pc, ruta_sc) in ARQS.items():
        pc = pickle.load(open(ruta_pc, "rb"))
        sc, _ = pickle.load(open(ruta_sc, "rb"))
        for barra in ORDEN:
            d = pc[barra]
            split = d["_datos_split"]
            g = lambda est, c: np.asarray(d[est][f"prediccion_{c}"], dtype=float)
            P = {c: g("Prophet_Solo", c) for c in ("val", "test")}
            T = {c: g("TFT_Precios_Solo", c) for c in ("val", "test")}
            R = {c: g("TFT_Residuos_Solo", c) for c in ("val", "test")}
            yv = np.asarray(split["y_real_val"], dtype=float)
            fechas_test = pd.to_datetime(split["fechas_test"])
            serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
            yt = serie.reindex(fechas_test).values
            ok = ~np.isnan(yt) & ~((fechas_test >= ini) & (fechas_test <= fin))
            ds = sc[barra]["test"]["Precios"]["datos"]
            s_sc = pd.Series(np.asarray(ds["y_pred"]), index=pd.to_datetime(ds["fechas"]))
            s_sc = s_sc[~s_sc.index.duplicated()].reindex(fechas_test).values

            mitad = len(yv) // 2
            A = slice(0, mitad)
            combos = {"pt": (P["val"][A], T["val"][A]), "pr": (P["val"][A], R["val"][A]),
                      "prt": (P["val"][A] + R["val"][A], T["val"][A])}
            # comparacion de optimizadores (sobre la primera mitad de validacion)
            w = {}
            for nombre_c, (a, b) in combos.items():
                res_opt = {}
                for nombre_o, f in OPTIMIZADORES.items():
                    t0 = time.perf_counter()
                    r = f(yv[A], a, b)
                    res_opt[nombre_o] = (r["w1"], mae(yv[A], r["w1"] * a + (1 - r["w1"]) * b), time.perf_counter() - t0)
                t0 = time.perf_counter()
                rr = ridge_regression(yv[A], a, b)
                t_ridge = time.perf_counter() - t0
                mae_ridge = mae(yv[A], rr["w1"] * a + rr.get("w2", 1 - rr["w1"]) * b)
                maes = [v[1] for v in res_opt.values()]
                filas_opt.append({"Arq": arq, "Barra": barra, "Combo": nombre_c,
                                  "dif_rel_max_%": 100 * (max(maes) - min(maes)) / min(maes),
                                  "dif_ridge_%": 100 * (mae_ridge - min(maes)) / min(maes),
                                  "t_random": res_opt["Random_1000"][2], "t_slsqp": res_opt["SLSQP"][2], "t_ridge": t_ridge})
                w[nombre_c] = res_opt["Grid_41"][0]
            wv = (w["pt"], w["pr"], w["prt"])
            pv = predicciones(P["val"], T["val"], R["val"], wv)
            pt = predicciones(P["test"], T["test"], R["test"], wv)
            mae_sel = {e: mae(yv[mitad:], pv[e][mitad:]) for e in ESTRATEGIAS}
            mejor = min(mae_sel, key=mae_sel.get)
            e_pc = np.abs(yt - pt[mejor])
            e_sc = np.abs(yt - s_sc)
            m = ok & ~np.isnan(e_pc) & ~np.isnan(e_sc)
            dm = diebold_mariano_desde_errores(e_pc[m], e_sc[m], h=1)
            mae_pc = float(e_pc[m].mean())
            mae_prophet_trunc = float(np.abs(yt - np.maximum(pt["Prophet solo"], 0))[m].mean())
            mae_tft = float(np.abs(yt - pt["TFT precios solo"])[m].mean())
            mae_opt = float(np.abs(yt - pt["Prophet + residuos (opt)"])[m].mean())
            filas.append({"Arq": arq, "Barra": barra, "Estrategia": mejor, "w_pt": w["pt"], "w_pr": w["pr"], "w_prt": w["prt"],
                          "MAE_PC": mae_pc, "MAE_SC": float(e_sc[m].mean()), "MAE_Prophet_trunc": mae_prophet_trunc,
                          "MAE_TFT": mae_tft, "MAE_estrategia5": mae_opt,
                          "dProphet_%": 100 * (1 - mae_pc / mae_prophet_trunc), "dTFT_%": 100 * (1 - mae_pc / mae_tft),
                          "DM": dm["DM"], "p": dm["p_value"]})
    df, dfo = pd.DataFrame(filas), pd.DataFrame(filas_opt)
    df.to_csv("Scripts/resultado_stacking_validacion_tablas.csv", index=False)
    dfo.to_csv("Scripts/resultado_stacking_validacion_optimizadores.csv", index=False)
    pd.set_option("display.width", 240)
    print(df.round(3).to_string(index=False))
    print("\nOptimizadores: dif. relativa max entre los 8 metodos = %.4f%%, Ridge max = %.3f%%; t Random %.2f-%.2f s, "
          "SLSQP max %.3f s, Ridge max %.3f s" % (dfo["dif_rel_max_%"].max(), dfo["dif_ridge_%"].abs().max(),
                                                  dfo.t_random.min(), dfo.t_random.max(), dfo.t_slsqp.max(), dfo.t_ridge.max()))


if __name__ == "__main__":
    main()
