#!/usr/bin/env python3
"""
dm_sensibilidad_rezagos.py - Test de Diebold-Mariano a una hora para las dos tablas de DM del
Capitulo 4: LN contra DyT (16 combinaciones de barra y rama) y P+C contra S+C (16 combinaciones de
barra y arquitectura), en dos versiones:

  - Horaria: d_t = |e_a| - |e_b| por hora, varianza de Newey-West (Bartlett) con 24 rezagos, porque
    la diferencia de perdidas esta autocorrelacionada en el rezago diario (24 h) aunque casi no en
    los rezagos 1 a 3.
  - Diaria (Lago et al., 2021): d_dia = promedio de d_t en las horas de cada dia, varianza de
    Newey-West con truncamiento automatico de Andrews (1991). En la serie diaria la autocorrelacion
    decae en 2 o 3 dias, sin picos estacionales, que es el caso para el que esta pensado el plug-in AR(1).

Se guarda tambien la version con lag = h + 1 = 2 de las tablas anteriores, como referencia.

Las predicciones son las ya guardadas por el pipeline (sin reentrenar). Para P+C se reproduce la
seleccion de stacking_validacion_tablas.py (pesos en la primera mitad de validacion, estrategia
elegida en la segunda). Los errores se calculan contra la serie cruda, excluyendo VENTANA_PRECIO_FALLA.

P+C contra S+C se repite ademas en la segunda mitad de la validacion (tabla PC_vs_SC_val), para
verificar que la eleccion de S+C no depende de haber mirado la prueba. Esa comparacion favorece a
P+C, cuya estrategia se elige en esas mismas horas.

Uso:
    python Scripts/dm_sensibilidad_rezagos.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Comparativa import diebold_mariano_desde_errores
from Modulos.Stacking_Optimization_TFT import grid_search_modelos
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA
from stacking_validacion_tablas import ARQS, ESTRATEGIAS, NOMBRES, ORDEN, RUTA_CRUDA, mae, predicciones

LAG_HORARIO = 24


def lag_andrews(d):
    """Truncamiento de Bartlett segun Andrews (1991), plug-in AR(1): S = 1.1447 (a1 T)^(1/3)."""
    d = d - d.mean()
    rho = float(np.sum(d[1:] * d[:-1]) / np.sum(d[:-1] ** 2))
    a1 = 4 * rho ** 2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return int(np.floor(1.1447 * (a1 * len(d)) ** (1 / 3)))


def evaluar(e_a, e_b, fechas):
    """DM horario (lag 2 y lag 24) y DM diario (Andrews) sobre d = e_a - e_b."""
    m = np.isfinite(e_a) & np.isfinite(e_b)
    e_a, e_b, f = e_a[m], e_b[m], fechas[m]
    out = {"MAE_a": e_a.mean(), "MAE_b": e_b.mean(), "n": len(e_a)}
    for lag in (2, LAG_HORARIO):
        r = diebold_mariano_desde_errores(e_a, e_b, h=1, max_lag=lag)
        out[f"DM_{lag}"], out[f"p_{lag}"] = r["DM"], r["p_value"]
    g = pd.DataFrame({"a": e_a, "b": e_b}, index=f).groupby(f.normalize()).mean()
    lag_d = lag_andrews(g["a"].values - g["b"].values)
    r = diebold_mariano_desde_errores(g["a"].values, g["b"].values, h=1, max_lag=lag_d)
    out.update({"n_dias": len(g), "lag_diario": lag_d, "DM_diario": r["DM"], "p_diario": r["p_value"]})
    return out


def holm(p, alfa=0.05):
    """Mascara de rechazo de Holm-Bonferroni."""
    p = np.asarray(p)
    orden = np.argsort(p)
    rechaza = np.zeros(len(p), dtype=bool)
    for i, j in enumerate(orden):
        if p[j] > alfa / (len(p) - i):
            break
        rechaza[j] = True
    return rechaza


def cargar():
    """Errores absolutos de prueba (y de la segunda mitad de validacion para P+C y S+C) alineados
    por fecha, por arquitectura y barra."""
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    ini, fin = VENTANA_PRECIO_FALLA
    err = {}
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
            fechas = pd.to_datetime(split["fechas_test"])
            serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
            yt = serie.reindex(fechas).values
            yt[(fechas >= ini) & (fechas <= fin)] = np.nan
            ds = sc[barra]["test"]["Precios"]["datos"]
            s_sc = pd.Series(np.asarray(ds["y_pred"]), index=pd.to_datetime(ds["fechas"]))
            s_sc = s_sc[~s_sc.index.duplicated()].reindex(fechas).values

            mitad = len(yv) // 2
            A = slice(0, mitad)
            w = [grid_search_modelos(yv[A], a[A], b[A], resolution=41)["w1"]
                 for a, b in ((P["val"], T["val"]), (P["val"], R["val"]), (P["val"] + R["val"], T["val"]))]
            pv = predicciones(P["val"], T["val"], R["val"], w)
            pt = predicciones(P["test"], T["test"], R["test"], w)
            mejor = min(ESTRATEGIAS, key=lambda e: mae(yv[mitad:], pv[e][mitad:]))
            fv = pd.to_datetime(split["fechas_val"])[mitad:]
            yv2 = serie.reindex(fv).values
            yv2[(fv >= ini) & (fv <= fin)] = np.nan
            dv = sc[barra]["val"]["Precios"]["datos"]
            s_scv = pd.Series(np.asarray(dv["y_pred"]), index=pd.to_datetime(dv["fechas"]))
            s_scv = s_scv[~s_scv.index.duplicated()].reindex(fv).values
            err[(arq, barra)] = {"fechas": fechas, "SC": np.abs(yt - s_sc), "PC": np.abs(yt - pt[mejor]),
                                 "TFT_PC": np.abs(yt - T["test"]),
                                 "val": {"fechas": fv, "SC": np.abs(yv2 - s_scv), "PC": np.abs(yv2 - pv[mejor][mitad:])}}
    return err


def main():
    err = cargar()
    filas = []
    for barra in ORDEN:
        for rama, clave in (("S+C", "SC"), ("P+C", "TFT_PC")):
            a, b = err[("LN", barra)], err[("DyT", barra)]
            f = pd.Series(a[clave], index=a["fechas"])
            g = pd.Series(b[clave], index=b["fechas"])
            idx = f.index.intersection(g.index)
            r = evaluar(f[idx].values, g[idx].values, idx)
            filas.append({"Tabla": "LN_vs_DyT", "Barra": NOMBRES[barra], "Grupo": rama, **r})
    for barra in ORDEN:
        for arq in ("LN", "DyT"):
            e = err[(arq, barra)]
            r = evaluar(e["PC"], e["SC"], e["fechas"])
            filas.append({"Tabla": "PC_vs_SC", "Barra": NOMBRES[barra], "Grupo": arq, **r})
            v = e["val"]
            r = evaluar(v["PC"], v["SC"], v["fechas"])
            filas.append({"Tabla": "PC_vs_SC_val", "Barra": NOMBRES[barra], "Grupo": arq, **r})
    df = pd.DataFrame(filas)
    cols_p = ["p_2", f"p_{LAG_HORARIO}", "p_diario"]
    for tabla in df["Tabla"].unique():
        m = df["Tabla"] == tabla
        for c in cols_p:
            df.loc[m, f"holm_{c[2:]}"] = holm(df.loc[m, c].values)
    df.to_csv("Scripts/resultado_dm_sensibilidad_rezagos.csv", index=False)

    pd.set_option("display.width", 250)
    for tabla, signo in (("LN_vs_DyT", "DM < 0 favorece a LN"), ("PC_vs_SC", "DM > 0 favorece a S+C"),
                         ("PC_vs_SC_val", "DM > 0 favorece a S+C")):
        t = df[df["Tabla"] == tabla]
        print(f"\n=== {tabla} ({signo}) ===")
        print(t[["Barra", "Grupo", "MAE_a", "MAE_b", "DM_2", "p_2", f"DM_{LAG_HORARIO}", f"p_{LAG_HORARIO}",
                 f"holm_{LAG_HORARIO}", "lag_diario", "DM_diario", "p_diario", "holm_diario"]]
              .round(3).to_string(index=False))
        for c in cols_p:
            s = t[t["holm_" + c[2:]].astype(bool)]
            dm = s["DM_" + c[2:]]
            print(f"  {c:10s} sin corregir {int((t[c] < 0.05).sum()):2d}, Holm {len(s):2d} "
                  f"(DM>0: {int((dm > 0).sum())}, DM<0: {int((dm < 0).sum())})")


if __name__ == "__main__":
    main()
