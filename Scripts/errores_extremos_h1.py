#!/usr/bin/env python3
"""
errores_extremos_h1.py - Analisis de errores extremos de S+C-LN a h=1 en validacion
(Seccion de errores extremos de la tesis), sobre la serie cruda.

Para cada barra se toman los 20 errores absolutos mas grandes del conjunto de validacion, se
recalcula el error tras desplazar la prediccion entre -3 y +3 horas (pred(t+s) frente a y(t)) y
se retiene el desplazamiento que minimiza el error. Si la reduccion supera el 60%, el caso es
"desfase de fase"; si no, "evento estructural". Un desplazamiento positivo indica que el modelo
predice la transicion con retraso. Como control, el mismo criterio se aplica a la persistencia
estacional (y(t-24)), y se reporta la sensibilidad al umbral (40% y 80%).

Se usan las predicciones con entradas crudas (Scripts/inferencia_entradas_crudas.py) y la serie
cruda, sin la ventana de precio de falla: el recorte de las entradas de validacion reemplaza
los spikes por interpolacion lineal y los saca del analisis.

Genera:
    docs/errores_extremos_top20_por_barra.csv
    Escrito/imagenes/fig_error_dist_atacama.pdf
    Escrito/imagenes/fig_errores_extremos_casos.pdf

Uso:
    python Scripts/errores_extremos_h1.py
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
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA
from inferencia_entradas_crudas import RUTA_BARRA

RUTA_CRUDA = "Datos/h/2020-2026.csv"
CARPETA_FIG = "Escrito/imagenes"
TOP_N = 20
DESPLAZAMIENTOS = range(-3, 4)
UMBRAL = 0.60
HORAS_TRANSICION = {7, 8, 19, 20, 21}
NOMBRES = {"ATACAMA": "Atacama", "CARDONES": "Cardones", "CHARRUA": "Charrúa", "CRUCERO": "Crucero",
           "P.AZUCAR": "Pan de Azúcar", "P.MONTT": "Puerto Montt", "QUILLOTA": "Quillota", "TARAPACA": "Tarapacá"}
# (barra, instante del error, titulo) de la figura de casos: dos desfases y dos eventos estructurales
CASOS = [("CARDONES", "2025-03-25 20:00", "Desfase de fase, Cardones"),
         ("ATACAMA", "2024-09-01 22:00", "Desfase de fase, Atacama"),
         ("ATACAMA", "2025-02-26 09:00", "Evento estructural, Atacama"),
         ("CARDONES", "2024-08-09 08:00", "Evento estructural, Cardones")]

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def clasificar(fechas, pred, y):
    ps = pd.Series(pred, index=fechas)
    e = np.abs(pred - y)
    filas = []
    for i in np.argsort(e)[::-1][:TOP_N]:
        t = fechas[i]
        mejor_red, mejor_s = -np.inf, 0
        for s in DESPLAZAMIENTOS:
            v = ps.get(t + pd.Timedelta(hours=s), np.nan)
            if np.isnan(v):
                continue
            red = 1 - abs(v - y[i]) / e[i]
            if red > mejor_red:
                mejor_red, mejor_s = red, s
        filas.append({"fecha": t, "error": e[i], "desplazamiento": mejor_s, "reduccion": mejor_red})
    return pd.DataFrame(filas), e


def fig_distribucion(e, ruta):
    mae, p90, p95, mx = e.mean(), np.percentile(e, 90), np.percentile(e, 95), e.max()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(e, bins=np.arange(0, mx + 5, 5), color="tab:blue", alpha=0.7)
    ax.set_yscale("log")
    for v, c, ls, nombre in ((mae, "red", "-", "MAE"), (p90, "#f4a259", "-", "P90"),
                             (p95, "#d9480f", "-", "P95"), (mx, "#800000", ":", "Máximo")):
        ax.axvline(v, color=c, linestyle=ls, linewidth=1.5, label=f"{nombre} = {v:.1f}")
    ax.annotate("", xy=(mx, 3), xytext=(mx - 60, 30), arrowprops=dict(arrowstyle="->", color="#800000"))
    ax.set_xlabel("Error absoluto (USD/MWh)")
    ax.set_ylabel("Frecuencia (escala log)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def fig_casos(series, preds, ruta):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (barra, instante, titulo) in zip(axes.flat, CASOS):
        t = pd.Timestamp(instante)
        sl = slice(t - pd.Timedelta(hours=36), t + pd.Timedelta(hours=36))
        y, p = series[barra][sl], preds[barra].sort_index().asfreq("h")[sl]
        ax.plot(y.index, y.values, color="tab:blue", label="Real")
        ax.plot(p.index, p.values, color="tab:red", linestyle="--", label="Predicción")
        ax.axvline(t, color="darkorange", linestyle=":", linewidth=1.5)
        err = abs(p.get(t, np.nan) - y.get(t, np.nan))
        ax.set_title(f"{titulo}\n{t:%Y-%m-%d %H:%M}, error = {err:.1f} USD/MWh", fontsize=9)
        ax.set_ylabel("USD/MWh")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def main():
    with open(RUTA_BARRA, "rb") as f:
        crudas = pickle.load(f)
    df = pd.read_csv(RUTA_CRUDA, sep=";")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    ini, fin = VENTANA_PRECIO_FALLA

    resumen, detalle, series, preds = [], [], {}, {}
    for barra, d in crudas.items():
        serie = df.loc[df["Barra"] == barra].groupby("Fecha")["Valor"].mean().asfreq("h")
        # la figura muestra la serie y la prediccion completas, incluida la ventana de precio de falla
        series[barra], preds[barra] = serie.copy(), pd.Series(d["val"][1], index=d["val"][0])
        serie.loc[(serie.index >= ini) & (serie.index <= fin)] = np.nan
        fechas, pred = d["val"]
        y = serie.reindex(fechas).values
        ok = ~np.isnan(y)
        fechas, pred, y = fechas[ok], pred[ok], y[ok]

        top, e = clasificar(fechas, pred, y)
        y24 = serie.reindex(fechas - pd.Timedelta(hours=24)).values
        ok24 = ~np.isnan(y24)
        top24, _ = clasificar(fechas[ok24], y24[ok24], y[ok24])

        desfase = top["reduccion"] > UMBRAL
        resumen.append({
            "Barra": NOMBRES[barra], "MAE": e.mean(), "P95": np.percentile(e, 95), "Maximo": e.max(),
            "Desfase": int(desfase.sum()),
            "Transicion": int(top["fecha"].dt.hour.isin(HORAS_TRANSICION).sum()),
            "Naive_t24": int((top24["reduccion"] > UMBRAL).sum()),
            "Desfase_40": int((top["reduccion"] > 0.4).sum()), "Desfase_80": int((top["reduccion"] > 0.8).sum()),
            "Red_mediana_%": 100 * top.loc[desfase, "reduccion"].median(),
            "Desplaz_positivo_%": 100 * (top.loc[desfase, "desplazamiento"] > 0).mean(),
        })
        for rank, fila in enumerate(top.itertuples(), 1):
            detalle.append({"Barra": NOMBRES[barra], "Rank": rank, "Fecha": f"{fila.fecha:%Y-%m-%d}",
                            "Hora": f"{fila.fecha:%H:%M}", "Error_abs_USD_MWh": round(fila.error, 2),
                            "Desplazamiento_h": fila.desplazamiento, "Reduccion": round(fila.reduccion, 3),
                            "Tipo": "Desfase de fase" if fila.reduccion > UMBRAL else "Evento estructural"})
        if barra == "ATACAMA":
            fig_distribucion(e, os.path.join(CARPETA_FIG, "fig_error_dist_atacama.pdf"))
            print(f"Atacama: n={len(e)}, P90={np.percentile(e, 90):.2f}")

    pd.DataFrame(detalle).to_csv("docs/errores_extremos_top20_por_barra.csv", index=False)
    fig_casos(series, preds, os.path.join(CARPETA_FIG, "fig_errores_extremos_casos.pdf"))
    res = pd.DataFrame(resumen)
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))
    print(res[["Desfase", "Desfase_40", "Desfase_80", "Naive_t24", "Transicion"]].sum().to_string())


if __name__ == "__main__":
    main()
