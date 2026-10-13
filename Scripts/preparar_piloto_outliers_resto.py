#!/usr/bin/env python3
"""
preparar_piloto_outliers_resto.py - Extiende el piloto de entrenamiento sin
recorte de outliers (Metodologia Seccion 3.8.6) a las 4 barras restantes del
SEN (Cardones, Crucero, P.Montt, Quillota), que no estaban en el piloto
original de 4 barras (Atacama, Charrua, P.Azucar, Tarapaca).

Replica exactamente el pipeline del piloto original (preparar_piloto_outliers.py,
recuperado del scratchpad de la sesion que lo genero): Prophet sin recortar
outliers en train (recortar_outliers_train=False), clima + features solares,
split + normalizacion, dataloaders con la misma config que pred_sol_clima.

A diferencia del original, NO sobreescribe las 4 barras ya existentes en
Multi-Modelos_TFT/h/LN/pred_sol_clima_piloto_sin_outliers/ (ya entrenadas y
evaluadas) -- carga los pkl existentes y les AGREGA las 4 barras nuevas,
dejando las 4 originales intactas.

Uso:
    python Scripts/preparar_piloto_outliers_resto.py
"""
import sys, io, os, json, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('.')
import pandas as pd
import holidays

from Modulos.Preprocesamiento_Prophet import preprocesamiento_prophet
from Modulos.Prophet_Modular import multi_prophet, optimizacion_prophet
from Modulos.Preprocesamiento_Transformer import (
    generar_clima, predicciones_con_clima, holiday_binary, COORDENADAS_BARRAS,
    agregar_features_solares, dividir_serie_temporal, normalizar_datos, limpiar_nans,
    crear_dataloaders,
)

BARRAS_RESTO = ['CARDONES', 'CRUCERO', 'P.MONTT', 'QUILLOTA']
FREQ = 'h'
ANO_INICIO, ANO_FIN = 2020, 2026
PROPORCIONES = [0.7, 0.15, 0.15]
UMBRAL_OUTLIERS = 4

CARPETA_PROPHET_PILOTO = 'Modelos_Prophet/h/modelos_piloto_sin_outliers'
CARPETA_TFT_PILOTO = 'Multi-Modelos_TFT/h/LN/pred_sol_clima_piloto_sin_outliers'

# ---------- 1) Cargar datos crudos ----------
df = pd.read_csv('Datos/h/2020-2026.csv', sep=';', decimal=',')
df['Barra'] = df['Barra'].astype(str)
df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce')
df['Valor_CLP'] = pd.to_numeric(df['Valor_CLP'], errors='coerce')

df_por_barras = {barra: df[df['Barra'] == barra].copy() for barra in BARRAS_RESTO}

# ---------- 2) Preprocesamiento Prophet (train SIN recorte automatico) ----------
# Ninguna de estas 4 barras tiene una exclusion manual conocida (a diferencia de
# P.AZUCAR en el piloto original, que si tenia un error de medicion verificado).
años = list(range(ANO_INICIO, ANO_FIN + 1))
ch_holidays = holidays.Chile(years=años)
df_feriados_comun = pd.DataFrame(ch_holidays.items(), columns=['ds', 'holiday'])
df_feriados_comun['ds'] = pd.to_datetime(df_feriados_comun['ds'])
df_feriados_comun = df_feriados_comun.sort_values(by='ds', ascending=True).reset_index(drop=True)

prepro_por_barras = {}
for barra in BARRAS_RESTO:
    df_train, df_val, df_test, df_fer = preprocesamiento_prophet(
        df_por_barras, barra, FREQ, ANO_INICIO, ANO_FIN, PROPORCIONES, UMBRAL_OUTLIERS,
        graficos=None, df_feriados=df_feriados_comun,
        recortar_outliers_train=False, fechas_excluir_manual_train=None)
    prepro_por_barras[barra] = {'df_train': df_train, 'df_val': df_val, 'df_test': df_test, 'df_feriados': df_fer}
    print(f"{barra}: train={len(df_train)} val={len(df_val)} test={len(df_test)}  max_train={df_train['y'].max():.2f}")

with open('prepro_por_barras_piloto_resto.pkl', 'wb') as f:
    pickle.dump(prepro_por_barras, f)

# ---------- 3) Prophet: reusar hiperparametros ya optimizados, reentrenar (train nuevo) ----------
ruta_params = f"Modelos_Prophet/{FREQ}/params_prophet_{FREQ}_{ANO_INICIO}-{ANO_FIN}.json"
parametros_full = {'grid': None, 'daily_seasonality': True, 'weekly_seasonality': True,
                    'yearly_seasonality': True, 'techo': 1.5}
mejores_params_todas = optimizacion_prophet(prepro_por_barras, ruta_params, parametros_full,
                                             modo='usar', metrica_objetivo='MAE', usar_cv=True, n_splits=5, hmap=False)
mejores_params_resto = {b: mejores_params_todas[b] for b in BARRAS_RESTO}

params_base = {'daily_seasonality': True, 'weekly_seasonality': True, 'yearly_seasonality': True}
modelos, df_predicciones, df_metricas_prophet = multi_prophet(
    prepro_por_barras=prepro_por_barras, parametros=mejores_params_resto, params_base=params_base,
    modo='entrenar', mostrar_params=False, mostrar_graficos=False, mostrar_componentes=False,
    carpeta_modelos=CARPETA_PROPHET_PILOTO)

print(df_metricas_prophet)

# ---------- 4) Clima + feriados + features solares ----------
Coordenadas = {b: COORDENADAS_BARRAS[b] for b in BARRAS_RESTO}
fecha_inicio = df_predicciones['ds'].min().strftime('%Y-%m-%d')
fecha_fin = df_predicciones['ds'].max().strftime('%Y-%m-%d')
carpeta_clima = f"Datos/{FREQ}/Climaticos"
datos_clima = generar_clima(fecha_inicio, fecha_fin, Coordenadas, carpeta_salida=carpeta_clima, freq=FREQ)
dict_barras_clima = predicciones_con_clima(df_predicciones, datos_clima)

for barra in dict_barras_clima.keys():
    df_bin = holiday_binary(dict_barras_clima[barra], prepro_por_barras[barra]['df_feriados'])
    dict_barras_clima[barra] = df_bin

datos_para_transformer = {}
for barra, df_b in dict_barras_clima.items():
    df_b = df_b.sort_values('ds').reset_index(drop=True)
    nan_mask = df_b['residuo'].isna()
    df_b.loc[nan_mask, 'residuo'] = df_b.loc[nan_mask, 'y_real'] - df_b.loc[nan_mask, 'yhat']
    df_b['time_idx'] = range(len(df_b))
    df_b['serie_id'] = barra
    datos_para_transformer[barra] = df_b

datos_para_transformer = agregar_features_solares(datos_para_transformer, Coordenadas)

# ---------- 5) Split + normalizacion ----------
datasets_transformer = {}
for barra, df_b in datos_para_transformer.items():
    df_train, df_val, df_test = dividir_serie_temporal(df_b, PROPORCIONES)
    datasets_transformer[barra] = {'train': df_train, 'val': df_val, 'test': df_test}

feature_cols = []
targets = ['y_real', 'residuo']
datasets_norm_nuevo, scalers_nuevo = normalizar_datos(datasets_transformer, feature_cols, targets)
datasets_norm_nuevo = limpiar_nans(BARRAS_RESTO, datasets_norm_nuevo)

# ---------- 6) Dataloaders (config identica a pred_sol_clima / al piloto original) ----------
var_clima = ['temperatura', 'humedad', 'velocidad_viento', 'precipitacion', 'nubosidad']
var_solar = ['es_horario_verano', 'elevacion_solar', 'cos_elevacion']
known_reals = var_solar
unknown_reals_price = ['y_real'] + var_clima
unknown_reals_resid = []

max_encoder_length = 168
max_prediction_length = 1
batch_size = 128

dataloaders_precios_nuevo, dataloaders_residuos_nuevo, _, _ = crear_dataloaders(
    BARRAS_RESTO, datasets_norm_nuevo, known_reals, unknown_reals_price, unknown_reals_resid,
    max_encoder_length, max_prediction_length, batch_size)

# ---------- 7) Cargar lo existente (4 barras del piloto original) y fusionar ----------
with open(f'{CARPETA_TFT_PILOTO}/dataloaders_precios.pkl', 'rb') as f:
    dataloaders_precios_total = pickle.load(f)
with open(f'{CARPETA_TFT_PILOTO}/dataloaders_residuos.pkl', 'rb') as f:
    dataloaders_residuos_total = pickle.load(f)
with open(f'{CARPETA_TFT_PILOTO}/scalers.pkl', 'rb') as f:
    scalers_total = pickle.load(f)
with open(f'{CARPETA_TFT_PILOTO}/datasets_norm.pkl', 'rb') as f:
    datasets_norm_total = pickle.load(f)
with open(f'{CARPETA_TFT_PILOTO}/feature_config.json', encoding='utf-8') as f:
    feature_config_total = json.load(f)

barras_ya_existentes = set(dataloaders_precios_total.keys())
interseccion = barras_ya_existentes & set(BARRAS_RESTO)
assert not interseccion, f"Estas barras ya existian en el piloto, abortando para no pisarlas: {interseccion}"

dataloaders_precios_total.update({b: dataloaders_precios_nuevo[b] for b in BARRAS_RESTO})
dataloaders_residuos_total.update({b: dataloaders_residuos_nuevo[b] for b in BARRAS_RESTO})
scalers_total.update({b: scalers_nuevo[b] for b in BARRAS_RESTO})
datasets_norm_total.update({b: datasets_norm_nuevo[b] for b in BARRAS_RESTO})
feature_config_total['lista_barras'] = sorted(set(feature_config_total['lista_barras']) | set(BARRAS_RESTO))

with open(f'{CARPETA_TFT_PILOTO}/dataloaders_precios.pkl', 'wb') as f:
    pickle.dump(dataloaders_precios_total, f)
with open(f'{CARPETA_TFT_PILOTO}/dataloaders_residuos.pkl', 'wb') as f:
    pickle.dump(dataloaders_residuos_total, f)
with open(f'{CARPETA_TFT_PILOTO}/scalers.pkl', 'wb') as f:
    pickle.dump(scalers_total, f)
with open(f'{CARPETA_TFT_PILOTO}/datasets_norm.pkl', 'wb') as f:
    pickle.dump(datasets_norm_total, f)
with open(f'{CARPETA_TFT_PILOTO}/feature_config.json', 'w', encoding='utf-8') as f:
    json.dump(feature_config_total, f, indent=2, ensure_ascii=False)

print("\nLISTO. Piloto sin recorte de outliers ahora cubre:", sorted(dataloaders_precios_total.keys()))
for b in BARRAS_RESTO:
    batch = next(iter(dataloaders_precios_nuevo[b]['train']))
    print(f"{b}: encoder_cont shape = {batch[0]['encoder_cont'].shape}, target shape = {batch[1][0].shape}")
