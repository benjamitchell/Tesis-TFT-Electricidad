#!/usr/bin/env python3
"""
piloto_lw_robusto_h24.py - Evalua la variante de LW-ACP con escala horaria robusta
(Scripts/piloto_lw_escala_robusta.py) sobre el modelo S+C-LN a 24 horas, en cada paso del
horizonte y en las 8 barras, junto a la version principal de LW-ACP y a ACI, igual que
Scripts/h24_cp_metodos.py. Como ahi, la adaptacion online usa un alpha_t por hora del dia
actualizado con el resultado de la misma hora del dia anterior (Zaffran et al., 2022), para
no usar resultados que aun no ocurren al emitir el pronostico.

El piso de la escala se elige solo en validacion, como a una hora: para cada piso de la
grilla se calibra con la primera mitad de la validacion de cada barra y paso y se mide el
Winkler en la segunda mitad; se elige un unico piso por nivel, el de menor Winkler promedio
sobre las 8 barras y los 24 pasos.

Lee las predicciones por paso de Scripts/extraer_predicciones_h24.py (Scripts/cache_h24.py).

Salida: Scripts/resultado_piloto_lw_robusto_h24.csv
Uso:
    python Scripts/piloto_lw_robusto_h24.py
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
from cache_h24 import cargar_preds_sc
from cp_metricas_completas import winkler, pinball
from h24_cp_metodos import marco, adaptativo
from piloto_lw_escala_robusta import lw_acp, PISOS, HORAS_TRANSICION

RUTA_CRUDA = "Datos/h/2020-2026.csv"
ALPHAS = (0.10, 0.05)
GAMMA = 0.05


def main():
    preds = cargar_preds_sc("LN")
    crudo = pd.read_csv(RUTA_CRUDA, sep=";")
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"])
    marcos = {}
    for barra, por_paso in preds.items():
        serie = crudo.loc[crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        for h, d in por_paso.items():
            marcos[(barra, h)] = (marco(d["val"], serie), marco(d["test"], serie))
    print(f"{len(marcos)} combinaciones barra-paso", flush=True)

    filas = []
    for alpha in ALPHAS:
        nivel = f"{int(round(100 * (1 - alpha)))}%"
        w_val = {}
        for piso in PISOS:
            ws = []
            for mv, _ in marcos.values():
                mitad = len(mv) // 2
                r = lw_acp(mv.iloc[:mitad].reset_index(drop=True), mv.iloc[mitad:].reset_index(drop=True), alpha, piso,
                           por_hora=True)
                ws.append(winkler(r["y_test"], r["lower"], r["upper"], alpha))
            w_val[piso] = float(np.mean(ws))
            print(f"[{nivel}] piso {piso}: Winkler validacion {w_val[piso]:.2f}", flush=True)
        piso = min(w_val, key=w_val.get)
        print(f"[{nivel}] piso elegido en validacion: {piso}", flush=True)

        for (barra, h), (mv, mt) in marcos.items():
            horas = mt["fecha"].dt.hour.values
            res = {"ACI": adaptativo(mv, mt, alpha, False, "por hora", h),
                   "LW-ACP": adaptativo(mv, mt, alpha, True, "por hora", h),
                   "LW-ACP robusto": lw_acp(mv, mt, alpha, piso, por_hora=True)}
            for nombre, r in res.items():
                y, lo, up = r["y_test"], r["lower"], r["upper"]
                dentro = (y >= lo) & (y <= up)
                filas.append({"Nivel": nivel, "Barra": barra, "h": h, "Metodo": nombre, "Piso": piso,
                              "Cobertura": dentro.mean(), "Desv": abs(dentro.mean() - (1 - alpha)),
                              "Ancho": np.mean(up - lo), "Winkler": winkler(y, lo, up, alpha),
                              "Pinball": (pinball(y, lo, alpha / 2) + pinball(y, up, 1 - alpha / 2)) / 2,
                              "Pinball_inf": pinball(y, lo, alpha / 2), "Pinball_sup": pinball(y, up, 1 - alpha / 2),
                              "SD_cob_hora": pd.Series(dentro).groupby(horas).mean().std(),
                              "Cob_transicion": dentro[np.isin(horas, HORAS_TRANSICION)].mean()})

    df = pd.DataFrame(filas)
    pd.set_option("display.width", 220)
    for nivel in ("90%", "95%"):
        d = df[df.Nivel == nivel]
        print(f"\n=== {nivel}: promedio de las 8 barras y los 24 pasos ===")
        print(d.groupby("Metodo")[["Cobertura", "Desv", "Ancho", "Winkler", "Pinball", "Pinball_inf", "Pinball_sup",
                                   "SD_cob_hora", "Cob_transicion"]].mean().round(4).to_string())
        w = d.pivot_table(index=["Barra", "h"], columns="Metodo", values="Winkler")
        print(f"Robusto mejor que LW-ACP en {int((w['LW-ACP robusto'] < w['LW-ACP']).sum())} de {len(w)} combinaciones barra-paso")
        print("Winkler promedio por paso:")
        print(d.pivot_table(index="h", columns="Metodo", values="Winkler").round(2).T.to_string())
    df.to_csv("Scripts/resultado_piloto_lw_robusto_h24.csv", index=False)
    print("\nGuardado: Scripts/resultado_piloto_lw_robusto_h24.csv")


if __name__ == "__main__":
    main()
