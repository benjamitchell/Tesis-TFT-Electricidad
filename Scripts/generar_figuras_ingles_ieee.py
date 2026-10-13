#!/usr/bin/env python3
"""
generar_figuras_ingles_ieee.py - Genera versiones en ingles de 3 figuras usadas en el
paper IEEE (hoy reusa las figuras en espanol de Escrito/imagenes/ porque el codigo
original que las genero no quedo guardado en el repo). Reconstruye desde los datos:

  1. fig_phase_shift_panel_en.pdf - 4 casos de error extremo en validacion, Atacama,
     ventanas de +-48h alrededor de cada instante. Los casos elegidos aqui son los de
     mayor error absoluto en la serie cruda; no necesariamente las mismas 4 horas de
     la figura en espanol (ese codigo no esta guardado), pero misma metodologia.
  2. fig_cp_cobertura_ancho_hora_en.pdf - cobertura y ancho medio por hora del dia,
     Split CP vs ACI vs LW-ACP, promediado sobre las 8 barras (alpha=0.05, gamma=0.05).
  3. fig_cp_ejemplo_intervalo_en.pdf - intervalo LW-ACP de ejemplo, Atacama, una semana
     de test (misma ventana que la tesis: 4-11 marzo 2026).

Guarda en Escrito/imagenes/ con sufijo _en, sin pisar las versiones en espanol.

Uso:
    python Scripts/generar_figuras_ingles_ieee.py
"""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Conformal_Prediction_Wrapper import (split_cp_from_calibration,
                                                  aci_from_calibration,
                                                  lw_acp_online_from_calibration)
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA

RUTA_EVAL = "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"
RUTA_CRUDA = "Datos/h/2020-2026.csv"
CARPETA_SALIDA = "Escrito/imagenes"
BARRAS = ["ATACAMA", "CARDONES", "CHARRUA", "CRUCERO", "P.AZUCAR", "P.MONTT", "QUILLOTA", "TARAPACA"]
ALPHA, GAMMA = 0.05, 0.05

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def cargar_crudo():
    df = pd.read_csv(RUTA_CRUDA, sep=";")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def serie_barra(df_crudo, barra):
    return df_crudo.loc[df_crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()


def marco(datos, barra, conjunto, serie):
    d = datos[barra][conjunto]["Precios"]["datos"]
    m = pd.DataFrame({"fecha": pd.to_datetime(d["fechas"]), "y_pred": np.asarray(d["y_pred"])})
    m = m.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
    m["y_real"] = serie.reindex(m["fecha"]).values
    return m.dropna(subset=["y_real"])


def fig_phase_shift(datos, df_crudo):
    barra = "ATACAMA"
    serie = serie_barra(df_crudo, barra)
    mv = marco(datos, barra, "val", serie)
    ini_falla, fin_falla = VENTANA_PRECIO_FALLA
    margen = pd.Timedelta(hours=12)
    mv = mv.loc[~((mv["fecha"] >= ini_falla - margen) & (mv["fecha"] <= fin_falla + margen))]
    mv["error"] = (mv["y_real"] - mv["y_pred"]).abs()

    # 4 mayores errores, exigiendo que las ventanas +-48h no se traslapen demasiado
    candidatos = mv.sort_values("error", ascending=False)
    elegidos = []
    for _, row in candidatos.iterrows():
        if all(abs((row["fecha"] - f).total_seconds()) > 4 * 24 * 3600 for f in elegidos):
            elegidos.append(row["fecha"])
        if len(elegidos) == 4:
            break

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, centro in zip(axes.flat, elegidos):
        ini, fin = centro - pd.Timedelta(hours=48), centro + pd.Timedelta(hours=48)
        ventana = mv[(mv["fecha"] >= ini) & (mv["fecha"] <= fin)]
        ax.plot(ventana["fecha"], ventana["y_real"], color="tab:blue", label="Real")
        ax.plot(ventana["fecha"], ventana["y_pred"], color="tab:red", linestyle="--", label="Predicted")
        ax.axvline(centro, color="darkorange", linestyle=":", linewidth=1.5)
        err = mv.loc[mv["fecha"] == centro, "error"].values[0]
        ax.set_title(f"{centro:%Y-%m-%d %H:%M}, error={err:.1f} USD/MWh", fontsize=9)
        ax.set_ylabel("USD/MWh")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        ax.tick_params(axis="x", rotation=30)
    axes.flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Extreme error cases, S+C-LN TFT, Atacama (validation set)")
    fig.tight_layout()
    ruta = os.path.join(CARPETA_SALIDA, "fig_phase_shift_panel_en.pdf")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {ruta}  (casos: {[str(c) for c in elegidos]})")


def fig_cp_por_hora(datos, df_crudo):
    acumulado = {m: {"cov": np.zeros(24), "width": np.zeros(24), "n": np.zeros(24)}
                 for m in ("Split CP", "ACI", "LW-ACP")}

    for barra in BARRAS:
        serie = serie_barra(df_crudo, barra)
        mv = marco(datos, barra, "val", serie)
        mt = marco(datos, barra, "test", serie)
        horas_t = mt["fecha"].dt.hour.values

        r_scp = split_cp_from_calibration(mv["y_real"].values, mv["y_pred"].values,
                                          mt["y_real"].values, mt["y_pred"].values, alpha=ALPHA)
        r_aci = aci_from_calibration(mv["y_real"].values, mv["y_pred"].values,
                                     mt["y_real"].values, mt["y_pred"].values, alpha=ALPHA, gamma=GAMMA)
        r_lwacp = lw_acp_online_from_calibration(mv["y_real"].values, mv["y_pred"].values, mv["fecha"].values,
                                                 mt["y_real"].values, mt["y_pred"].values, mt["fecha"].values,
                                                 alpha=ALPHA, gamma=GAMMA)

        for nombre, r in (("Split CP", r_scp), ("ACI", r_aci), ("LW-ACP", r_lwacp)):
            y, lo, up = r["y_test"], r["lower"], r["upper"]
            dentro = (y >= lo) & (y <= up)
            for h in range(24):
                mask = horas_t == h
                if mask.sum() == 0:
                    continue
                acumulado[nombre]["cov"][h] += dentro[mask].sum()
                acumulado[nombre]["width"][h] += (up[mask] - lo[mask]).sum()
                acumulado[nombre]["n"][h] += mask.sum()
        print(f"  [{barra}] ok")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colores = {"Split CP": "tab:gray", "ACI": "tab:orange", "LW-ACP": "tab:blue"}
    for nombre in ("Split CP", "ACI", "LW-ACP"):
        n = acumulado[nombre]["n"]
        cov_h = acumulado[nombre]["cov"] / n
        width_h = acumulado[nombre]["width"] / n
        ax1.plot(range(24), cov_h, marker="o", markersize=3, label=nombre, color=colores[nombre])
        ax2.plot(range(24), width_h, marker="o", markersize=3, label=nombre, color=colores[nombre])

    ax1.axhline(0.95, color="black", linestyle=":", linewidth=1)
    ax1.set_xlabel("Hour of day"); ax1.set_ylabel("Empirical coverage"); ax1.set_title("Coverage by hour")
    ax2.set_xlabel("Hour of day"); ax2.set_ylabel("Mean interval width (USD/MWh)"); ax2.set_title("Width by hour")
    ax1.legend(fontsize=8); ax1.set_xticks(range(0, 24, 3)); ax2.set_xticks(range(0, 24, 3))
    fig.tight_layout()
    ruta = os.path.join(CARPETA_SALIDA, "fig_cp_cobertura_ancho_hora_en.pdf")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {ruta}")


def fig_cp_ejemplo(datos, df_crudo):
    barra = "ATACAMA"
    serie = serie_barra(df_crudo, barra)
    mv = marco(datos, barra, "val", serie)
    mt = marco(datos, barra, "test", serie)

    r = lw_acp_online_from_calibration(mv["y_real"].values, mv["y_pred"].values, mv["fecha"].values,
                                       mt["y_real"].values, mt["y_pred"].values, mt["fecha"].values,
                                       alpha=ALPHA, gamma=GAMMA)
    df_r = pd.DataFrame({"fecha": mt["fecha"].values, "y_real": r["y_test"],
                         "y_pred": mt["y_pred"].values, "lower": r["lower"], "upper": r["upper"]})

    ini, fin = pd.Timestamp("2026-03-04"), pd.Timestamp("2026-03-11")
    v = df_r[(df_r["fecha"] >= ini) & (df_r["fecha"] <= fin)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(v["fecha"], v["lower"], v["upper"], color="tab:blue", alpha=0.2, label="LW-ACP interval (95%)")
    ax.plot(v["fecha"], v["y_real"], color="black", linewidth=1.2, label="Real")
    ax.plot(v["fecha"], v["y_pred"], color="tab:red", linestyle="--", linewidth=1, label="S+C-LN prediction")
    ax.set_ylabel("USD/MWh")
    ax.set_title(f"Example LW-ACP prediction interval, Atacama, {ini:%d %b}--{fin:%d %b %Y}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    ruta = os.path.join(CARPETA_SALIDA, "fig_cp_ejemplo_intervalo_en.pdf")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {ruta}")


def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    df_crudo = cargar_crudo()

    print("1/3 Phase-shift panel (Atacama)...")
    fig_phase_shift(datos, df_crudo)
    print("2/3 CP coverage/width by hour (8 barras)...")
    fig_cp_por_hora(datos, df_crudo)
    print("3/3 CP example interval (Atacama)...")
    fig_cp_ejemplo(datos, df_crudo)


if __name__ == "__main__":
    main()
