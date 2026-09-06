import gc
import os
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Extraer predicciones del modelo considerando desafase
def extraer_predicciones_modelo(barra, modelo, dataloader, datasets_norm,
                                diccionario_scalers, conjunto='test',
                                target_key='y_real', scaler_key=None,
                                horizonte=1, umbral_desfase=0.001,
                                df_completo=None, n_train=None, n_val=None):

    if scaler_key is None:
        scaler_key = target_key

    # Preparar datasets
    if df_completo is None:
        df_train = datasets_norm[barra]['train']
        df_val   = datasets_norm[barra]['val']
        df_test  = datasets_norm[barra]['test']
        df_completo = pd.concat([df_train, df_val, df_test], ignore_index=True)
        n_train = len(df_train)
        n_val   = len(df_val)

    fechas_completo = pd.to_datetime(df_completo['ds'].values)
    
    # Validaciones
    if target_key not in df_completo.columns:
        raise KeyError(f"Columna '{target_key}' no existe en datasets_norm")
    
    try:
        scaler = diccionario_scalers[barra]['scalers_target'][scaler_key]
    except KeyError:
        raise KeyError(f"Scaler '{scaler_key}' no encontrado para {barra}")
    
    # Determinar límites del conjunto
    if conjunto == 'train':
        inicio_conjunto = 0
        fin_conjunto = n_train
    elif conjunto == 'val':
        inicio_conjunto = n_train
        fin_conjunto = n_train + n_val
    else:  # test
        inicio_conjunto = n_train + n_val
        fin_conjunto = len(df_completo)
    
    n_conjunto = fin_conjunto - inicio_conjunto
    
    # Obtener valores reales
    y_original_norm = df_completo[target_key].values
    y_original_completo = scaler.inverse_transform(y_original_norm.reshape(-1, 1)).flatten()
    y_real_conjunto = y_original_completo[inicio_conjunto:fin_conjunto]
    fechas_conjunto = fechas_completo[inicio_conjunto:fin_conjunto]
    
    # Extraer predicciones del modelo
    with torch.inference_mode():
        raw = modelo.predict(dataloader, mode="raw", return_y=True, return_index=True)

    predictions = raw.output.prediction if hasattr(raw.output, 'prediction') else raw.output[0]

    # Si tiene quantiles (7 columnas), usar mediana (índice 3)
    if len(predictions.shape) == 3 and predictions.shape[2] == 7:
        predictions = predictions[:, :, 3]

    # Extraer horizonte específico
    if len(predictions.shape) == 2:
        y_pred_norm = predictions[:, horizonte-1].cpu().numpy()
    else:
        y_pred_norm = predictions.cpu().numpy()

    y_pred_completo = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

    # Obtener valores reales del modelo
    actuals = raw.y
    if isinstance(actuals, (list, tuple)):
        y_tft_reales_norm = actuals[0].cpu().numpy()
    else:
        y_tft_reales_norm = actuals.cpu().numpy()

    if len(y_tft_reales_norm.shape) > 1:
        y_tft_reales_norm = y_tft_reales_norm.flatten()

    y_tft_reales_completo = scaler.inverse_transform(y_tft_reales_norm.reshape(-1, 1)).flatten()

    # Alinear por time_idx
    try:
        time_idx = raw.index['time_idx'].cpu().numpy()
    except AttributeError:
        time_idx = raw.index['time_idx'].values

    del raw
    
    # Mapear a posiciones locales (vectorizado)
    y_pred_mapeado = np.full(n_conjunto, np.nan)
    y_tft_reales_mapeado = np.full(n_conjunto, np.nan)

    mask_conjunto = (time_idx >= inicio_conjunto) & (time_idx < fin_conjunto)
    idx_locales = time_idx[mask_conjunto] - inicio_conjunto
    y_pred_mapeado[idx_locales] = y_pred_completo[mask_conjunto]
    y_tft_reales_mapeado[idx_locales] = y_tft_reales_completo[mask_conjunto]
    
    # Filtrar solo valores válidos
    mask_validos = ~np.isnan(y_pred_mapeado)
    
    y_pred_validos = y_pred_mapeado[mask_validos]
    y_tft_reales_validos = y_tft_reales_mapeado[mask_validos]
    y_real_validos = y_real_conjunto[mask_validos]
    fechas_validas = fechas_conjunto[mask_validos]
    
    # Calcular desfase (diff_reales)
    if len(y_real_validos) > 0:
        diff_reales = np.mean(np.abs(y_real_validos - y_tft_reales_validos))
        tiene_desfase = diff_reales >= umbral_desfase
    else:
        diff_reales = np.nan
        tiene_desfase = True
    
    return {
        'fechas': fechas_validas,
        'y_real': y_real_validos,
        'y_pred': y_pred_validos,
        'y_tft_reales': y_tft_reales_validos,
        'n_validos': len(y_pred_validos),
        'n_total': n_conjunto,
        'diff_reales': diff_reales,
        'tiene_desfase': tiene_desfase,
        'indices_validos': np.where(mask_validos)[0],
    }

# Ejecuta UNA sola inferencia sobre el dataloader del conjunto test (que cubre
# train+val+test) y devuelve las predicciones divididas por conjunto.
def inferir_raw_modelo(modelo, dataloader_test):
    """
    Corre la inferencia completa del modelo sobre `dataloader_test` una sola vez.
    Reusar el resultado (parametro `raw` de extraer_todas_predicciones_modelo) para
    extraer varios horizontes del mismo modelo sin re-inferir sobre todo el set de
    test en cada paso -- evitar_todas_predicciones_modelo llamado en un loop de 24
    horizontes sin esto re-hace la inferencia completa 24 veces por nada.
    """
    with torch.inference_mode():
        return modelo.predict(dataloader_test, mode="raw", return_y=True, return_index=True)


def extraer_todas_predicciones_modelo(barra, modelo, dataloader_test, datasets_norm,
                                      diccionario_scalers, target_key='y_real',
                                      scaler_key=None, horizonte=1, umbral_desfase=0.001,
                                      raw=None):

    if scaler_key is None:
        scaler_key = target_key

    df_train    = datasets_norm[barra]['train']
    df_val      = datasets_norm[barra]['val']
    df_test_df  = datasets_norm[barra]['test']
    df_completo = pd.concat([df_train, df_val, df_test_df], ignore_index=True)
    n_train     = len(df_train)
    n_val       = len(df_val)
    n_total     = len(df_completo)

    fechas_completo = pd.to_datetime(df_completo['ds'].values)

    if target_key not in df_completo.columns:
        raise KeyError(f"Columna '{target_key}' no existe en datasets_norm")
    try:
        scaler = diccionario_scalers[barra]['scalers_target'][scaler_key]
    except KeyError:
        raise KeyError(f"Scaler '{scaler_key}' no encontrado para {barra}")

    y_original_completo = scaler.inverse_transform(
        df_completo[target_key].values.reshape(-1, 1)).flatten()

    # Una sola inferencia (o reusar una ya calculada, ver inferir_raw_modelo)
    if raw is None:
        raw = inferir_raw_modelo(modelo, dataloader_test)

    predictions = raw.output.prediction if hasattr(raw.output, 'prediction') else raw.output[0]
    if len(predictions.shape) == 3 and predictions.shape[2] == 7:
        predictions = predictions[:, :, 3]  # perdida cuantilica: mediana de las 7 cuantiles
    elif len(predictions.shape) == 3 and predictions.shape[2] == 1:
        # perdida puntual (MAE) con max_prediction_length>1: (n_muestras, n_pasos, 1),
        # sin dimension de cuantiles -- se aplana la ultima dimension antes de indexar
        # por paso de horizonte, si no y_pred_norm queda con n_muestras*n_pasos valores.
        predictions = predictions.squeeze(-1)
    if len(predictions.shape) == 2:
        y_pred_norm = predictions[:, horizonte - 1].cpu().numpy()
    else:
        y_pred_norm = predictions.cpu().numpy()
    y_pred_completo = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

    actuals = raw.y
    if isinstance(actuals, (list, tuple)):
        y_tft_reales_norm = actuals[0].cpu().numpy()
    else:
        y_tft_reales_norm = actuals.cpu().numpy()
    if len(y_tft_reales_norm.shape) > 1:
        # multi-step (max_prediction_length > 1): una columna por paso de horizonte,
        # igual que predictions -- seleccionar la misma columna, no aplanar todas
        y_tft_reales_norm = y_tft_reales_norm[:, horizonte - 1]
    y_tft_reales_completo = scaler.inverse_transform(
        y_tft_reales_norm.reshape(-1, 1)).flatten()

    try:
        time_idx = raw.index['time_idx'].cpu().numpy()
    except AttributeError:
        time_idx = raw.index['time_idx'].values
    # raw.index['time_idx'] es el time_idx del PRIMER paso de predicción
    # (pytorch_forecasting: TimeSeriesDataSet.x_to_index -> decoder_time_idx[:, 0]);
    # para horizonte > 1 el paso horizonte cae en time_idx + (horizonte - 1).
    if horizonte > 1:
        time_idx = time_idx + (horizonte - 1)

    del raw

    limites = {
        'train': (0, n_train),
        'val':   (n_train, n_train + n_val),
        'test':  (n_train + n_val, n_total),
    }

    resultados = {}
    for conjunto, (inicio, fin) in limites.items():
        n_conj = fin - inicio

        y_pred_mapeado       = np.full(n_conj, np.nan)
        y_tft_reales_mapeado = np.full(n_conj, np.nan)

        mask = (time_idx >= inicio) & (time_idx < fin)
        idx_locales = time_idx[mask] - inicio
        y_pred_mapeado[idx_locales]       = y_pred_completo[mask]
        y_tft_reales_mapeado[idx_locales] = y_tft_reales_completo[mask]

        mask_validos         = ~np.isnan(y_pred_mapeado)
        y_real_validos       = y_original_completo[inicio:fin][mask_validos]
        y_pred_validos       = y_pred_mapeado[mask_validos]
        y_tft_reales_validos = y_tft_reales_mapeado[mask_validos]
        fechas_validas       = fechas_completo[inicio:fin][mask_validos]

        diff_reales   = (np.mean(np.abs(y_real_validos - y_tft_reales_validos))
                         if len(y_real_validos) > 0 else np.nan)
        tiene_desfase = (diff_reales >= umbral_desfase
                         if not np.isnan(diff_reales) else True)

        resultados[conjunto] = {
            'fechas':          fechas_validas,
            'y_real':          y_real_validos,
            'y_pred':          y_pred_validos,
            'y_tft_reales':    y_tft_reales_validos,
            'n_validos':       len(y_pred_validos),
            'n_total':         n_conj,
            'diff_reales':     diff_reales,
            'tiene_desfase':   tiene_desfase,
            'indices_validos': np.where(mask_validos)[0],
        }

    return resultados


# Calcular métricas
def calcular_metricas_predicciones(y_real, y_pred, y_tft_reales=None,
                                    umbral_desfase=0.001, verbose=True):
    
    if len(y_real) == 0:
        return {
            'MAE': np.nan,
            'RMSE': np.nan,
            'MSE': np.nan,
            'R2': np.nan,
            'diff_reales': np.nan,
            'tiene_desfase': True
        }
    
    # Métricas estándar
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    
    # Calcular desfase
    if y_tft_reales is not None and len(y_tft_reales) > 0:
        diff_reales = np.mean(np.abs(y_real - y_tft_reales))
        tiene_desfase = diff_reales >= umbral_desfase
    else:
        diff_reales = np.nan
        tiene_desfase = False
    
    # Status con desfase
    status = "Desfase" if tiene_desfase else "Alineado"
    
    if verbose:
        print(f"    MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
        print(f"    Desfase: {diff_reales:.6f} (umbral: {umbral_desfase}) → {status}")
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2,
        'diff_reales': diff_reales,
        'tiene_desfase': tiene_desfase
    }

# Recalcula métricas contra la serie cruda (sin recorte de outliers), reusando predicciones ya guardadas
# Ventana de precio de falla/administrativo (17h del 25-feb-2025 a 03h del 26-feb-2025):
# valor idéntico (576.82) en las 8 barras simultáneamente, verificado contra la serie
# cruda -- no es un precio de mercado (no depende de clima/solar/historial), es un
# techo regulatorio aplicado a todo el sistema. Se excluye por defecto de las métricas
# de test, igual que cualquier otro dato administrativo no representativo del fenómeno
# que el modelo intenta predecir.
VENTANA_PRECIO_FALLA = (pd.Timestamp('2025-02-25 17:00:00'), pd.Timestamp('2025-02-26 03:00:00'))


def recalcular_metricas_serie_cruda(barra, fechas, y_pred, ruta_serie_cruda='Datos/h/2020-2026.csv',
                                     percentil_regimen=95, excluir_precio_falla=True, verbose=True):
    """
    Recalcula MAE/RMSE/R2 usando la serie cruda (Datos/h/2020-2026.csv, sin recorte de
    outliers por Z-score) como y_real, en vez de la serie recortada con la que se
    evaluó originalmente. Reusa las predicciones (y_pred) ya calculadas por el modelo
    entrenado — no requiere reentrenar ni re-inferir.

    Además calcula métricas condicionales al régimen sobre esa misma serie cruda:
    horas de precio muy alto (>= percentil `percentil_regimen` del propio conjunto
    evaluado) y horas de precio cero (curtailment/sobreoferta).

    Si `excluir_precio_falla=True` (default), se excluye del recálculo la ventana de
    precio administrativo verificada en VENTANA_PRECIO_FALLA.
    """
    df_crudo = pd.read_csv(ruta_serie_cruda, sep=';')
    df_crudo['Fecha'] = pd.to_datetime(df_crudo['Fecha'])
    sub_crudo = df_crudo.loc[df_crudo['Barra'] == barra]
    # Timestamps duplicados (hora 01:00 del cambio de horario en abril) se promedian
    serie_barra = sub_crudo.groupby('Fecha')['Valor'].mean()

    fechas = pd.to_datetime(fechas)
    y_real_crudo = serie_barra.reindex(fechas).values
    y_pred = np.asarray(y_pred)

    faltantes = np.isnan(y_real_crudo)
    if faltantes.any() and verbose:
        print(f"[{barra}] Advertencia: {faltantes.sum()} de {len(fechas)} fechas sin dato crudo "
              f"(no encontradas en {ruta_serie_cruda}), se excluyen del recálculo.")
    mask = ~faltantes

    if excluir_precio_falla:
        ini, fin = VENTANA_PRECIO_FALLA
        mask_falla = (fechas >= ini) & (fechas <= fin)
        if mask_falla.any() and verbose:
            print(f"[{barra}] Excluyendo {int(mask_falla.sum())} horas de precio de falla administrativo "
                  f"({ini} a {fin}).")
        mask = mask & ~np.asarray(mask_falla)

    metricas_general = calcular_metricas_predicciones(y_real_crudo[mask], y_pred[mask], verbose=False)

    umbral_alto = float(np.percentile(y_real_crudo[mask], percentil_regimen))
    mask_alto = mask & (y_real_crudo >= umbral_alto)
    mask_cero = mask & (y_real_crudo == 0)

    metricas_alto = (calcular_metricas_predicciones(y_real_crudo[mask_alto], y_pred[mask_alto], verbose=False)
                      if mask_alto.sum() > 0 else None)
    metricas_cero = (calcular_metricas_predicciones(y_real_crudo[mask_cero], y_pred[mask_cero], verbose=False)
                      if mask_cero.sum() > 0 else None)

    if verbose:
        print(f"[{barra}] General (serie cruda, n={int(mask.sum())}): "
              f"MAE={metricas_general['MAE']:.2f} | R²={metricas_general['R2']:.3f}")
        if metricas_alto is not None:
            print(f"[{barra}] Régimen alto (y>=p{percentil_regimen}={umbral_alto:.1f}, "
                  f"n={int(mask_alto.sum())}): MAE={metricas_alto['MAE']:.2f}")
        if metricas_cero is not None:
            print(f"[{barra}] Régimen cero (y=0, n={int(mask_cero.sum())}): MAE={metricas_cero['MAE']:.2f}")

    return {
        'general': metricas_general,
        'n_total': int(mask.sum()),
        f'p{percentil_regimen}_umbral': umbral_alto,
        f'p{percentil_regimen}_n': int(mask_alto.sum()),
        f'p{percentil_regimen}_metricas': metricas_alto,
        'cero_n': int(mask_cero.sum()),
        'cero_metricas': metricas_cero,
    }

# Líneas base ingenuas (persistencia, naive estacional, AR vía OLS) para MASE/skill score
def calcular_baselines_naive(barra, fechas_train, fechas_test, mae_modelo=None,
                              ruta_serie_cruda='Datos/h/2020-2026.csv',
                              lags_naive=(1, 24, 168), lags_ar=(24, 168), verbose=True):
    """
    Calcula líneas base ingenuas sobre la serie cruda (sin recorte de outliers, ver
    recalcular_metricas_serie_cruda): persistencia y naive estacional en `lags_naive`
    (ej. lag 1 = predecir el valor de la hora anterior, lag 24 = mismo momento ayer,
    lag 168 = mismo momento la semana pasada), más AR(lag) vía regresión lineal (OLS)
    ajustada en train para cada lag en `lags_ar`. Todas se evalúan en el mismo conjunto
    de test (mismas fechas) que el modelo, para que las comparaciones sean directas.

    Si se entrega `mae_modelo` (el MAE del modelo en ese mismo test), calcula además
    MASE = mae_modelo / MAE_persistencia_lag1 y el skill score = 1 - MASE, la
    definición estándar (Hyndman y Koehler, 2006) usando la persistencia de un paso
    como referencia.
    """
    df_crudo = pd.read_csv(ruta_serie_cruda, sep=';')
    df_crudo['Fecha'] = pd.to_datetime(df_crudo['Fecha'])
    serie = df_crudo.loc[df_crudo['Barra'] == barra].groupby('Fecha')['Valor'].mean()

    fechas_train = pd.to_datetime(fechas_train)
    fechas_test = pd.to_datetime(fechas_test)
    y_real_test = serie.reindex(fechas_test).values

    resultados = {}

    for lag in lags_naive:
        y_pred = serie.reindex(fechas_test - pd.Timedelta(hours=lag)).values
        mask = np.isfinite(y_real_test) & np.isfinite(y_pred)
        mae = mean_absolute_error(y_real_test[mask], y_pred[mask])
        nombre = 'persistencia' if lag == 1 else f'naive_lag{lag}'
        resultados[nombre] = {'MAE': mae, 'n': int(mask.sum())}
        if verbose:
            print(f"[{barra}] {nombre} (lag={lag}h): MAE={mae:.2f} (n={int(mask.sum())})")

    for lag in lags_ar:
        y_train = serie.reindex(fechas_train).values
        x_train = serie.reindex(fechas_train - pd.Timedelta(hours=lag)).values
        mask_train = np.isfinite(y_train) & np.isfinite(x_train)

        modelo = LinearRegression()
        modelo.fit(x_train[mask_train].reshape(-1, 1), y_train[mask_train])

        x_test = serie.reindex(fechas_test - pd.Timedelta(hours=lag)).values
        mask_test = np.isfinite(y_real_test) & np.isfinite(x_test)
        y_pred = modelo.predict(x_test[mask_test].reshape(-1, 1))
        mae = mean_absolute_error(y_real_test[mask_test], y_pred)

        nombre = f'AR{lag}'
        resultados[nombre] = {'MAE': mae, 'n': int(mask_test.sum()),
                               'coef': float(modelo.coef_[0]), 'intercept': float(modelo.intercept_)}
        if verbose:
            print(f"[{barra}] {nombre} (OLS en train): MAE={mae:.2f} "
                  f"(y = {modelo.coef_[0]:.3f}*y[t-{lag}] + {modelo.intercept_:.2f})")

    if mae_modelo is not None:
        mae_persistencia = resultados['persistencia']['MAE']
        mase = mae_modelo / mae_persistencia
        skill_score = 1 - mase
        resultados['MASE'] = mase
        resultados['skill_score'] = skill_score
        if verbose:
            print(f"[{barra}] MASE={mase:.3f} | Skill score={skill_score:.3f} "
                  f"(modelo MAE={mae_modelo:.2f} vs. persistencia MAE={mae_persistencia:.2f})")

    return resultados

# Función para crear gráfico de evaluación de precios y residuos
def crear_grafico_evaluacion(barra, datos_precios, datos_residuos,
                            metricas_precios, metricas_residuos,
                            nombre_modelo='TFT', horizonte=1,
                            carpeta=None, mostrar=True, verbose=True):

    N_INSET = 14 * 24  # 2 semanas en datos horarios

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── SUBPLOT 1: PRECIOS ─────────────────────────────────────────────────────
    ax = axes[0]

    ax.plot(datos_precios['fechas'], datos_precios['y_real'],
            label='Real', color='black', linewidth=2.5, alpha=0.9, zorder=3)
    ax.plot(datos_precios['fechas'], datos_precios['y_pred'],
            label=f'{nombre_modelo} (t+{horizonte})',
            color='red', linewidth=1.5, alpha=0.8, zorder=2)

    m_p = metricas_precios
    ax.set_title(
        f'{nombre_modelo} - {barra} - PRECIOS TEST (t+{horizonte})\n'
        f'MAE: {m_p["MAE"]:.2f} | RMSE: {m_p["RMSE"]:.2f} | '
        f'R²: {m_p["R2"]:.3f} | N: {datos_precios["n_validos"]}/{datos_precios["n_total"]}',
        fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Fecha', fontsize=11)
    ax.set_ylabel('Precio (USD/MWh)', fontsize=11)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # Inset precios: últimas 2 semanas
    n_in = min(N_INSET, len(datos_precios['fechas']))
    ax_in = ax.inset_axes([0.62, 0.55, 0.36, 0.38])
    ax_in.plot(datos_precios['fechas'][-n_in:], datos_precios['y_real'][-n_in:],
               color='black', linewidth=1.5, alpha=0.9)
    ax_in.plot(datos_precios['fechas'][-n_in:], datos_precios['y_pred'][-n_in:],
               color='red', linewidth=1.2, alpha=0.85)
    ax_in.set_title('Últimas 2 sem.', fontsize=8, fontweight='bold')
    ax_in.tick_params(labelsize=7)
    ax_in.tick_params(axis='x', rotation=30)
    ax_in.grid(True, alpha=0.3, linewidth=0.5)
    ax.indicate_inset_zoom(ax_in, edgecolor='0.4', linewidth=1.2)

    # ── SUBPLOT 2: RESIDUOS ────────────────────────────────────────────────────
    ax = axes[1]

    ax.plot(datos_residuos['fechas'], datos_residuos['y_real'],
            label='Residuos Reales', color='black', linewidth=2.5, alpha=0.9, zorder=3)
    ax.plot(datos_residuos['fechas'], datos_residuos['y_pred'],
            label=f'Residuos {nombre_modelo} (t+{horizonte})',
            color='red', linewidth=1.5, alpha=0.8, zorder=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    m_r = metricas_residuos
    ax.set_title(
        f'{nombre_modelo} - {barra} - RESIDUOS TEST (t+{horizonte})\n'
        f'MAE: {m_r["MAE"]:.2f} | RMSE: {m_r["RMSE"]:.2f} | '
        f'R²: {m_r["R2"]:.3f} | N: {datos_residuos["n_validos"]}/{datos_residuos["n_total"]}',
        fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Fecha', fontsize=11)
    ax.set_ylabel('Residuo (USD/MWh)', fontsize=11)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # Inset residuos: últimas 2 semanas
    n_in = min(N_INSET, len(datos_residuos['fechas']))
    ax_in = ax.inset_axes([0.62, 0.55, 0.36, 0.38])
    ax_in.plot(datos_residuos['fechas'][-n_in:], datos_residuos['y_real'][-n_in:],
               color='black', linewidth=1.5, alpha=0.9)
    ax_in.plot(datos_residuos['fechas'][-n_in:], datos_residuos['y_pred'][-n_in:],
               color='red', linewidth=1.2, alpha=0.85)
    ax_in.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_in.set_title('Últimas 2 sem.', fontsize=8, fontweight='bold')
    ax_in.tick_params(labelsize=7)
    ax_in.tick_params(axis='x', rotation=30)
    ax_in.grid(True, alpha=0.3, linewidth=0.5)
    ax.indicate_inset_zoom(ax_in, edgecolor='0.4', linewidth=1.2)

    # Advertencia si R² negativo
    if m_r['R2'] < 0:
        ax.text(0.02, 0.98,
                f'R² negativo ({m_r["R2"]:.3f})\nModelo peor que promedio',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                fontsize=9)

    plt.tight_layout()

    if carpeta is not None:
        os.makedirs(carpeta, exist_ok=True)
        nombre_archivo = f'evaluacion_{nombre_modelo}_{barra}.png'
        ruta = os.path.join(carpeta, nombre_archivo)
        plt.savefig(ruta, dpi=300, bbox_inches='tight')
        if verbose:
            tqdm.write(f"  Gráfico guardado: {ruta}")

    if mostrar:
        plt.show()
    else:
        plt.close()

    return fig, axes

# Función para crear tabla resumen de evaluación
def crear_tabla_resumen(datos_evaluacion):

    datos_tabla = []
    
    # Loop
    for barra in sorted(datos_evaluacion.keys()):
        for tipo_modelo in ['Precios', 'Residuos']:
            for conjunto in ['train', 'val', 'test']:
                
                try:
                    datos = datos_evaluacion[barra][conjunto][tipo_modelo]
                    
                    status = "Alineado" if not datos['metricas']['tiene_desfase'] else "⚠️ Desfase"
                    
                    datos_tabla.append({
                        'Barra': barra,
                        'Modelo': tipo_modelo.upper(),
                        'Set': conjunto.capitalize(),
                        'N': datos['n_validos'],
                        'MAE': round(datos['metricas']['MAE'], 2),
                        'RMSE': round(datos['metricas']['RMSE'], 2),
                        'R²': round(datos['metricas']['R2'], 3),
                        'Status': status
                    })
                
                except KeyError:
                    pass
    
    # Crear DataFrame
    df = pd.DataFrame(datos_tabla)

    # Ocultar nombres repetidos (solo para mostrar)
    df_display = df.copy()
    df_display['Barra'] = df_display['Barra'].mask(
        df_display['Barra'].duplicated(keep='first'), '')

    barra_modelo = df['Barra'] + '_' + df['Modelo']
    df_display['Modelo'] = df_display['Modelo'].mask(
        barra_modelo.duplicated(keep='first'),'')


    try:
        from IPython.display import display
        display(df_display)
    except Exception:
        print(df_display.to_string(index=False))

    return df

# Función principal
def evaluar_modelos_tft(lista_barras, 
                           modelos_precios, 
                           modelos_residuos,
                           dataloaders_precios, 
                           dataloaders_residuos, 
                           datasets_norm,
                           diccionario_scalers,
                           horizonte=1,
                           nombre_modelo='TFT',
                           umbral_desfase=0.001,
                           carpeta_graficos=None,
                           mostrar_graficos=True,
                           verbose=True):
    
    datos_evaluacion = {}
    
    # Loop principal
    for barra in tqdm(lista_barras, desc="Evaluando barras", unit="barra"):
        if verbose:
            tqdm.write(f"\nEvaluando: {barra}")

        datos_evaluacion[barra] = {}

        # Inferencia secuencial para no saturar la RAM
        todas_precios = extraer_todas_predicciones_modelo(
            barra,
            modelos_precios[barra],
            dataloaders_precios[barra]['test'],
            datasets_norm,
            diccionario_scalers,
            'y_real',
            None,
            horizonte,
            umbral_desfase,
        )
        gc.collect()

        todas_residuos = extraer_todas_predicciones_modelo(
            barra,
            modelos_residuos[barra],
            dataloaders_residuos[barra]['test'],
            datasets_norm,
            diccionario_scalers,
            'residuo',
            None,
            horizonte,
            umbral_desfase,
        )
        gc.collect()

        for conjunto in ['train', 'val', 'test']:

            datos_evaluacion[barra][conjunto] = {}

            # PRECIOS
            datos_precios = todas_precios[conjunto]
            metricas_precios = calcular_metricas_predicciones(
                datos_precios['y_real'],
                datos_precios['y_pred'],
                y_tft_reales=datos_precios['y_tft_reales'],
                umbral_desfase=umbral_desfase,
                verbose=verbose,
            )
            datos_evaluacion[barra][conjunto]['Precios'] = {
                'datos': datos_precios,
                'metricas': metricas_precios,
                'n_validos': datos_precios['n_validos'],
                'n_total': datos_precios['n_total'],
            }

            # RESIDUOS
            datos_residuos = todas_residuos[conjunto]
            metricas_residuos = calcular_metricas_predicciones(
                datos_residuos['y_real'],
                datos_residuos['y_pred'],
                y_tft_reales=datos_residuos['y_tft_reales'],
                umbral_desfase=umbral_desfase,
                verbose=verbose,
            )
            datos_evaluacion[barra][conjunto]['Residuos'] = {
                'datos': datos_residuos,
                'metricas': metricas_residuos,
                'n_validos': datos_residuos['n_validos'],
                'n_total': datos_residuos['n_total'],
            }
        
        # Grafico de test        
        datos_test_precios = datos_evaluacion[barra]['test']['Precios']['datos']
        datos_test_residuos = datos_evaluacion[barra]['test']['Residuos']['datos']
        metricas_test_precios = datos_evaluacion[barra]['test']['Precios']['metricas']
        metricas_test_residuos = datos_evaluacion[barra]['test']['Residuos']['metricas']
        
        crear_grafico_evaluacion(
            barra,
            datos_test_precios,
            datos_test_residuos,
            metricas_test_precios,
            metricas_test_residuos,
            nombre_modelo=nombre_modelo,
            horizonte=horizonte,
            carpeta=carpeta_graficos,
            mostrar=mostrar_graficos,
            verbose=verbose)

        gc.collect()
        torch.cuda.empty_cache()

    # Tabla
    df_metricas = crear_tabla_resumen(datos_evaluacion)
    
    return datos_evaluacion, df_metricas