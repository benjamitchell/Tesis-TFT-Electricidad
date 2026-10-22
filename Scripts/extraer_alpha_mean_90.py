#!/usr/bin/env python3
"""Extrae alpha_mean de ACI y LW-ACP al 90% (alpha=0.10), reusando la misma carga
de datos que cp_metricas_completas.py, para completar las tablas de Resultados
que ahora reportan 90% como nivel principal."""
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Conformal_Prediction_Wrapper import aci_from_calibration, lw_acp_online_from_calibration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cp_metricas_completas import marco

RUTA_EVAL = "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"
BARRAS = ["ATACAMA", "CARDONES", "CHARRUA", "CRUCERO", "P.AZUCAR", "P.MONTT", "QUILLOTA", "TARAPACA"]
GAMMA = 0.05
ALPHA = 0.10



def main():
    with open(RUTA_EVAL, "rb") as f:
        datos, _ = pickle.load(f)
    df_crudo = pd.read_csv("Datos/h/2020-2026.csv", sep=";")
    df_crudo["Fecha"] = pd.to_datetime(df_crudo["Fecha"])

    for barra in BARRAS:
        serie = df_crudo.loc[df_crudo["Barra"] == barra].groupby("Fecha")["Valor"].mean()
        mv = marco(datos, barra, "val", serie)
        mt = marco(datos, barra, "test", serie)

        r_aci = aci_from_calibration(mv["y_real"].values, mv["y_pred"].values,
                                      mt["y_real"].values, mt["y_pred"].values, alpha=ALPHA, gamma=GAMMA)
        r_lwacp = lw_acp_online_from_calibration(mv["y_real"].values, mv["y_pred"].values, mv["fecha"].values,
                                                  mt["y_real"].values, mt["y_pred"].values, mt["fecha"].values,
                                                  alpha=ALPHA, gamma=GAMMA)
        print(f"{barra}: ACI alpha_mean={r_aci['alpha_mean']:.4f}  LW-ACP alpha_mean={r_lwacp['alpha_mean']:.4f}")


if __name__ == "__main__":
    main()
