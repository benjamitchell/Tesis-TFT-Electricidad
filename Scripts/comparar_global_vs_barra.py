#!/usr/bin/env python3
"""
comparar_global_vs_barra.py - Test de Diebold-Mariano (Newey-West) comparando el
modelo GLOBAL (Scripts/evaluar_modelo_global.py) contra el modelo S+C-LN por-barra
existente (Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl),
sobre el conjunto de prueba, errores absolutos horarios pareados por fecha.

Uso:
    python Scripts/comparar_global_vs_barra.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import os
import pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modulos.Comparativa import diebold_mariano
from Modulos.Evaluacion_TFT import VENTANA_PRECIO_FALLA

RUTA_GLOBAL    = "Multi-Modelos_TFT/h/LN/pred_sol_clima_global/Resultados/eval_TFT_GLOBAL.pkl"
RUTA_POR_BARRA = "Multi-Modelos_TFT/h/LN/pred_sol_clima/Resultados/eval_TFT_LN.pkl"
RUTA_SERIE_CRUDA = "Datos/h/2020-2026.csv"

with open(RUTA_GLOBAL, "rb") as f:
    datos_global, _ = pickle.load(f)
with open(RUTA_POR_BARRA, "rb") as f:
    datos_barra, _ = pickle.load(f)

# Serie cruda real (Datos/h/2020-2026.csv), la misma referencia que usa la tesis
# (recalcular_metricas_serie_cruda) -- el y_real interno de cada pkl (via scaler)
# no coincide exactamente con esta serie, asi que se descarta y se usa esta.
df_crudo = pd.read_csv(RUTA_SERIE_CRUDA, sep=';')
df_crudo['Fecha'] = pd.to_datetime(df_crudo['Fecha'])

barras = sorted(datos_global.keys())
filas = []

for barra in barras:
    dg = datos_global[barra]['test']['Precios']['datos']
    db = datos_barra[barra]['test']['Precios']['datos']

    serie_barra = df_crudo.loc[df_crudo['Barra'] == barra].groupby('Fecha')['Valor'].mean()

    df_g = pd.DataFrame({'fecha': pd.to_datetime(dg['fechas']), 'y_pred_g': dg['y_pred']})
    df_b = pd.DataFrame({'fecha': pd.to_datetime(db['fechas']), 'y_pred_b': db['y_pred']})
    m = df_g.merge(df_b, on='fecha', how='inner')
    m['y_real'] = serie_barra.reindex(m['fecha']).values

    ini, fin = VENTANA_PRECIO_FALLA
    m = m.dropna(subset=['y_real'])
    m = m.loc[~((m['fecha'] >= ini) & (m['fecha'] <= fin))]

    res = diebold_mariano(m['y_real'].values, m['y_pred_g'].values, m['y_pred_b'].values, h=1, loss='mae')

    mae_g = np.mean(np.abs(m['y_real'].values - m['y_pred_g'].values))
    mae_b = np.mean(np.abs(m['y_real'].values - m['y_pred_b'].values))

    filas.append({
        'Barra': barra,
        'n_pareado': len(m),
        'MAE_global': round(mae_g, 3),
        'MAE_por_barra': round(mae_b, 3),
        'DM': round(res['DM'], 3),
        'p_value': res['p_value'],
        'Gana_global': res['DM'] < 0,
        'Signif_5pct': res['p_value'] < 0.05,
    })

df = pd.DataFrame(filas)
pd.set_option('display.width', 140)
print(df.to_string(index=False))

n_signif_global = int(((df['Gana_global']) & (df['Signif_5pct'])).sum())
n_signif_barra  = int(((~df['Gana_global']) & (df['Signif_5pct'])).sum())
print(f"\nGlobal gana de forma significativa en {n_signif_global}/8 barras.")
print(f"Por-barra gana de forma significativa en {n_signif_barra}/8 barras.")
print(f"Sin diferencia significativa en {8 - n_signif_global - n_signif_barra}/8 barras.")

# Test de signo simple (binomial), complementario: bajo H0 de que no hay ventaja
# sistematica, la probabilidad de que el modelo global gane en las 8/8 barras por
# puro azar es 0.5**8 -- un chequeo rapido, no reemplaza al DM (que ya es el test
# per-barra riguroso, con autocorrelacion serial considerada).
n_gana_global = int(df['Gana_global'].sum())
from scipy.stats import binomtest
p_signo = binomtest(n_gana_global, n=8, p=0.5, alternative='two-sided').pvalue
print(f"\nTest de signo (8 barras, H0: p=0.5 de que cualquiera gane): "
      f"global gana en {n_gana_global}/8, p={p_signo:.4f}")

ruta_out = "Multi-Modelos_TFT/h/LN/pred_sol_clima_global/Resultados/dm_global_vs_barra.csv"
df.to_csv(ruta_out, index=False)
print(f"\nGuardado: {ruta_out}")
