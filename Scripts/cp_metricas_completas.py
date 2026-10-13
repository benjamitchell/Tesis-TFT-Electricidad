#!/usr/bin/env python3
"""
cp_metricas_completas.py - Amplia la comparacion de metodos de CP (Split CP, ACI,
LW Split CP, LW-ACP) con las metricas que usa Kath & Ziel (2021) ademas de las que ya
se reportan (ancho, cobertura, Winkler): perdida pinball promedio y el test de
Christoffersen (UC: cobertura incondicional; CC: cobertura condicional) aplicado hora
por hora sobre el conjunto de prueba, igual que su Figura 8 (tasa de aprobacion sobre
las 24 horas del dia, nivel de significancia 5%). Se agrega ademas alpha=0.10 (90%) y
alpha=0.50 (50%) junto al alpha=0.05 (95%) ya reportado, para que la comparacion de
ancho entre niveles sea evidencia cuantitativa de colas pesadas.

CQR queda fuera de esta ampliacion (su calibracion via regresion cuantilica con
features solares es una pieza aparte, Modulos/Conformal_Prediction_Wrapper.py); se
mantiene solo al 95% ya reportado, como comparacion externa a la familia de CP.

Uso:
    python Scripts/cp_metricas_completas.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Conformal_Prediction_Wrapper import (split_cp_from_calibration, aci_from_calibration,
                                                  lw_split_cp_from_calibration, lw_acp_online_from_calibration)
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA

RUTA_EVAL = "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"
BARRAS = ["ATACAMA", "CARDONES", "CHARRUA", "CRUCERO", "P.AZUCAR", "P.MONTT", "QUILLOTA", "TARAPACA"]
GAMMA = 0.05
NIVELES = {0.05: "95%", 0.10: "90%", 0.50: "50%"}


def pinball(y, q, tau):
    e = y - q
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def _ll(n0, n1, p):
    """log-verosimilitud binomial con piso numerico para p en {0,1}."""
    out = 0.0
    if n0 > 0:
        out += n0 * np.log(max(1 - p, 1e-12))
    if n1 > 0:
        out += n1 * np.log(max(p, 1e-12))
    return out


def christoffersen_test(hits, alpha):
    """Test de Christoffersen (1998). hits: secuencia binaria, 1 = el valor real
    quedo fuera del intervalo, en orden cronologico. Devuelve LR_uc, p_uc, LR_cc, p_cc."""
    hits = np.asarray(hits).astype(int)
    n = len(hits)
    if n < 2:
        return dict(LR_uc=np.nan, p_uc=np.nan, LR_cc=np.nan, p_cc=np.nan)
    n1 = int(hits.sum())
    n0 = n - n1
    pi_hat = n1 / n

    ll_r = _ll(n0, n1, alpha)
    ll_u = _ll(n0, n1, pi_hat) if 0 < pi_hat < 1 else 0.0
    lr_uc = -2 * (ll_r - ll_u)

    trans = np.zeros((2, 2))
    for t in range(1, n):
        trans[hits[t - 1], hits[t]] += 1
    n00, n01, n10, n11 = trans[0, 0], trans[0, 1], trans[1, 0], trans[1, 1]
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi2 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_ind = (_ll(n00, n01, pi01) if (n00 + n01) > 0 else 0.0) + \
             (_ll(n10, n11, pi11) if (n10 + n11) > 0 else 0.0)
    ll_restricted = _ll(n00 + n10, n01 + n11, pi2)
    lr_ind = -2 * (ll_restricted - ll_ind)
    lr_cc = lr_uc + lr_ind

    p_uc = 1 - chi2.cdf(lr_uc, df=1) if np.isfinite(lr_uc) else np.nan
    p_cc = 1 - chi2.cdf(lr_cc, df=2) if np.isfinite(lr_cc) else np.nan
    return dict(LR_uc=lr_uc, p_uc=p_uc, LR_cc=lr_cc, p_cc=p_cc)


def marco(datos, barra, conjunto, serie):
    d = datos[barra][conjunto]["Precios"]["datos"]
    m = pd.DataFrame({"fecha": pd.to_datetime(d["fechas"]), "y_pred": np.asarray(d["y_pred"])})
    m = m.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
    m["y_real"] = serie.reindex(m["fecha"]).values
    m = m.dropna(subset=["y_real"])
    ini, fin = VENTANA_PRECIO_FALLA
    return m.loc[~((m["fecha"] >= ini) & (m["fecha"] <= fin))].reset_index(drop=True)


def evaluar_metodo(nombre, mv, mt, alpha):
    if nombre == "Split CP":
        r = split_cp_from_calibration(mv["y_real"].values, mv["y_pred"].values,
                                      mt["y_real"].values, mt["y_pred"].values, alpha=alpha)
    elif nombre == "ACI":
        r = aci_from_calibration(mv["y_real"].values, mv["y_pred"].values,
                                 mt["y_real"].values, mt["y_pred"].values, alpha=alpha, gamma=GAMMA)
    elif nombre == "LW Split CP":
        r = lw_split_cp_from_calibration(mv["y_real"].values, mv["y_pred"].values, mv["fecha"].values,
                                         mt["y_real"].values, mt["y_pred"].values, mt["fecha"].values, alpha=alpha)
    elif nombre == "LW-ACP":
        r = lw_acp_online_from_calibration(mv["y_real"].values, mv["y_pred"].values, mv["fecha"].values,
                                           mt["y_real"].values, mt["y_pred"].values, mt["fecha"].values,
                                           alpha=alpha, gamma=GAMMA)
    return r


def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    df_crudo = pd.read_csv("Datos/h/2020-2026.csv", sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])

    METODOS = ["Split CP", "ACI", "LW Split CP", "LW-ACP"]
    filas = []

    for barra in BARRAS:
        serie = df_crudo.loc[df_crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        mv, mt = marco(datos, barra, "val", serie), marco(datos, barra, "test", serie)
        horas_test = mt["fecha"].dt.hour.values

        for alpha, nivel in NIVELES.items():
            tau_lo, tau_hi = alpha / 2, 1 - alpha / 2
            for metodo in METODOS:
                r = evaluar_metodo(metodo, mv, mt, alpha)
                y, lo, up = r["y_test"], r["lower"], r["upper"]
                dentro = (y >= lo) & (y <= up)
                hits = (~dentro).astype(int)

                pb = (pinball(y, lo, tau_lo) + pinball(y, up, tau_hi)) / 2
                winkler = float(np.mean((up - lo) + (2 / alpha) * np.maximum(lo - y, 0)
                                        + (2 / alpha) * np.maximum(y - up, 0)))

                pass_uc, pass_cc = 0, 0
                for h in range(24):
                    mask_h = horas_test == h
                    if mask_h.sum() < 10:
                        continue
                    t = christoffersen_test(hits[mask_h], alpha)
                    if np.isfinite(t["p_uc"]) and t["p_uc"] >= 0.05:
                        pass_uc += 1
                    if np.isfinite(t["p_cc"]) and t["p_cc"] >= 0.05:
                        pass_cc += 1

                filas.append({
                    "Barra": barra, "Nivel": nivel, "Metodo": metodo,
                    "Cobertura": round(float(dentro.mean()), 4), "Ancho": round(float(np.mean(up - lo)), 2),
                    "Winkler": round(winkler, 2), "Pinball": round(pb, 3),
                    "UC_pasa_%": round(100 * pass_uc / 24, 1), "CC_pasa_%": round(100 * pass_cc / 24, 1),
                })
        print(f"[{barra}] ok")

    df = pd.DataFrame(filas)
    pd.set_option("display.width", 200)
    print("\n=== Promedio de las 8 barras, por nivel y metodo ===")
    print(df.groupby(["Nivel", "Metodo"])[["Cobertura", "Ancho", "Winkler", "Pinball", "UC_pasa_%", "CC_pasa_%"]]
          .mean().round(3).to_string())

    df.to_csv("Scripts/resultado_cp_metricas_completas.csv", index=False)
    print("\nGuardado: Scripts/resultado_cp_metricas_completas.csv")


if __name__ == "__main__":
    main()
