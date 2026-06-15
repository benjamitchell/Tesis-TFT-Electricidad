import gc
import os
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
def extraer_todas_predicciones_modelo(barra, modelo, dataloader_test, datasets_norm,
                                      diccionario_scalers, target_key='y_real',
                                      scaler_key=None, horizonte=1, umbral_desfase=0.001):

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

    # Una sola inferencia
    with torch.inference_mode():
        raw = modelo.predict(dataloader_test, mode="raw", return_y=True, return_index=True)

    predictions = raw.output.prediction if hasattr(raw.output, 'prediction') else raw.output[0]
    if len(predictions.shape) == 3 and predictions.shape[2] == 7:
        predictions = predictions[:, :, 3]
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
        y_tft_reales_norm = y_tft_reales_norm.flatten()
    y_tft_reales_completo = scaler.inverse_transform(
        y_tft_reales_norm.reshape(-1, 1)).flatten()

    try:
        time_idx = raw.index['time_idx'].cpu().numpy()
    except AttributeError:
        time_idx = raw.index['time_idx'].values

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