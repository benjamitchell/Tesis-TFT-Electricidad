import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import json, os, pickle, warnings, logging
from sklearn.model_selection import ParameterGrid
warnings.filterwarnings('ignore')

# Silenciar logs de Prophet
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

def crear_prophet(parametros, params, growth_type, df_feriados):
    return Prophet(
        daily_seasonality=parametros.get('daily_seasonality', True),
        weekly_seasonality=parametros.get('weekly_seasonality', True),
        yearly_seasonality=parametros.get('yearly_seasonality', True),
        seasonality_mode=params.get('seasonality_mode', 'additive'),
        seasonality_prior_scale=params.get('seasonality_prior_scale', 1.0),
        changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
        growth=growth_type,
        holidays=df_feriados,
        interval_width=0.95
    )

def calcular_score(metrica_objetivo, y_true, y_pred):
    if metrica_objetivo == 'RMSE':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metrica_objetivo == 'MAE':
        return mean_absolute_error(y_true, y_pred)
    elif metrica_objetivo == 'MSE':
        return mean_squared_error(y_true, y_pred)
    else:
        raise ValueError(f"Métrica {metrica_objetivo} no disponible.")

# Función para optimizar hiperparámetros de Prophet
def optimizar_hiperparametros(df_dict, barra_nombre, parametros, metrica_objetivo,
                              usar_cv=False, n_splits=5, hmap=True):
    
    if 'grid' not in parametros:
        raise ValueError("parametros debe contener 'grid' en el diccionario de búsqueda")
    
    # Extraemos los datos
    data_barra = df_dict[barra_nombre]
    df_train = data_barra['df_train'].copy()
    df_val = data_barra['df_val'].copy()
    df_feriados = data_barra['df_feriados'].copy()
    
    # Eliminar columnas innecesarias (solo mantener 'ds' y 'y')
    df_train = df_train[['ds', 'y']]
    df_val = df_val[['ds', 'y']]
    
    grid = ParameterGrid(parametros['grid'])
    resultados = []
    
    print(f"\n  Optimizando {barra_nombre}")
    print(f"  {'─'*60}")
    print(f"  Combinaciones: {len(grid)}")
    print(f"  Métrica: {metrica_objetivo}")
    print(f"  Método: {'Validación Cruzada' if usar_cv else 'Train/Val Simple'}")
    
    # Configuración de growth
    growth_type = parametros.get('growth', 'linear')
    
    # NO hacer logistic por ahora
    if growth_type == 'logistic':
        print(f"  Advertencia: growth='logistic' puede causar problemas")
        print(f"  Usando 'linear' en su lugar")
        growth_type = 'linear'
    
    # Entrenamiento
    for i, params in enumerate(grid, 1):
        print(f"  Procesando {i}/{len(grid)}", end='\r')
        
        try:
            scores = []

            if usar_cv:
                # Concatenamos train + val para hacer CV temporal
                df_completo = pd.concat([df_train, df_val], ignore_index=True)
                n_total = len(df_completo)
                
                # Empezamos con el 60% de los datos para entrenar
                initial_train_size = int(n_total * 0.6)
                step = (n_total - initial_train_size) // n_splits
                
                for split_i in range(n_splits):
                    train_end = initial_train_size + split_i * step
                    val_end = train_end + step
                    
                    # Evitar overflow de índices
                    if val_end > n_total: 
                        val_end = n_total
                    
                    df_train_cv = df_completo.iloc[:train_end].copy()
                    df_val_cv = df_completo.iloc[train_end:val_end].copy()
                    
                    if len(df_val_cv) < 2: 
                        continue

                    # Instanciar Prophet
                    try:
                        m = crear_prophet(parametros, params, growth_type, df_feriados)

                        # Entrenar
                        m.fit(df_train_cv)

                        # Predecir
                        forecast = m.predict(df_val_cv[['ds']])

                        # Cálculo métrica
                        y_true = df_val_cv['y'].values
                        y_pred = forecast['yhat'].values
                        scores.append(calcular_score(metrica_objetivo, y_true, y_pred))
                    
                    except Exception as e:
                        continue
                
                final_score = np.mean(scores) if scores else np.inf

            # Si no usamos CV, entrenamos directamente con train/val
            else:
                try:
                    m = crear_prophet(parametros, params, growth_type, df_feriados)

                    # Entrenar
                    m.fit(df_train)

                    # Predecir
                    forecast = m.predict(df_val[['ds']])

                    y_true = df_val['y'].values
                    y_pred = forecast['yhat'].values
                    final_score = calcular_score(metrica_objetivo, y_true, y_pred)

                except Exception as e:
                    final_score = np.inf

            # Guardamos
            res_dict = {
                'changepoint_prior_scale': params['changepoint_prior_scale'],
                'seasonality_prior_scale': params['seasonality_prior_scale'],
                'seasonality_mode': params.get('seasonality_mode', 'additive'),
                'growth': growth_type,
                metrica_objetivo: final_score
            }
            
            resultados.append(res_dict)

        except Exception as e:
            continue

    print(f"Optimización completada")
    
    # Resultados y selección
    df_resultados = pd.DataFrame(resultados)
    
    if df_resultados.empty:
        raise ValueError(f"No se obtuvieron resultados para {barra_nombre}")
    
    # Mejor resultado (Minimizando error)
    mejor_idx = df_resultados[metrica_objetivo].idxmin()
    mejor_resultado = df_resultados.loc[mejor_idx]
    
    print(f"\n  Mejor configuración:")
    print(f"  ├─ CPS: {mejor_resultado['changepoint_prior_scale']}")
    print(f"  ├─ SPS: {mejor_resultado['seasonality_prior_scale']}")
    print(f"  ├─ Mode: {mejor_resultado['seasonality_mode']}")
    print(f"  └─ {metrica_objetivo}: {mejor_resultado[metrica_objetivo]:.4f}")

    # Mapa de calor
    if hmap and len(df_resultados) > 1:
        try:
            # Filtrar por modo ganador
            modo_ganador = mejor_resultado['seasonality_mode']
            df_plot = df_resultados[df_resultados['seasonality_mode'] == modo_ganador]
            
            pivot_table = df_plot.pivot_table(
                values=metrica_objetivo,
                index='changepoint_prior_scale',
                columns='seasonality_prior_scale',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 7))
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis_r', 
                       cbar_kws={'label': metrica_objetivo}, linewidths=0.5)
            plt.title(f'Optimización de Hiperparámetros - {barra_nombre}\n({metrica_objetivo})', 
                     fontweight='bold', fontsize=12)
            plt.xlabel('Seasonality Prior Scale (Flexibilidad Estacional)', fontsize=10)
            plt.ylabel('Changepoint Prior Scale (Flexibilidad Tendencia)', fontsize=10)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"No se pudo generar heatmap: {e}")

    return df_resultados, mejor_resultado


# Función para gestionar optimización
def optimizacion_prophet(resultados_dict, ruta_archivo, parametros, modo, metrica_objetivo, 
                        usar_cv=False, n_splits=5, hmap=False):
    
    # Validar modo
    if modo not in ['usar', 'optimizar']:
        raise ValueError("El modo debe ser 'usar' o 'optimizar'")

    # Asegurar directorio
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)

    # Lógica de control
    archivo_existe = os.path.exists(ruta_archivo)
    
    # Modo 'usar' y el archivo existe -> Cargar y salir
    if modo == 'usar' and archivo_existe:
        print(f"\n{'='*80}")
        print(f"[Modo: USAR] Cargando parámetros preexistentes")
        print(f"{'='*80}")
        print(f"Ruta: {ruta_archivo}\n")

        try:
            with open(ruta_archivo, 'r') as f:
                mejores_params = json.load(f)
            
            print(f"Parámetros cargados para {len(mejores_params)} barras")
            return mejores_params
        
        except json.JSONDecodeError:
            print(f"Archivo JSON corrupto. Se procederá a re-optimizar.\n")

    # Modo 'optimizar' o modo 'usar' pero el archivo no existe
    if modo == 'optimizar':
        print(f"\n{'='*80}")
        print(f"[Modo: OPTIMIZAR] Forzando búsqueda de nuevos hiperparámetros")
        print(f"{'='*80}\n")
        
    elif modo == 'usar' and not archivo_existe:
        print(f"\n{'='*80}")
        print(f"[Modo: USAR] No se encontró {ruta_archivo}")
        print(f"Se iniciará la optimización automática")
        print(f"{'='*80}\n")

    if usar_cv:
        print(f"CV Activado: Validación cruzada temporal con {n_splits} splits\n")
    
    # Optimización
    print(f"Iniciando optimización para {len(resultados_dict)} barras\n")

    mejores_params = {}
    barras = list(resultados_dict.keys())

    for idx, barra in enumerate(barras, 1):
        print(f"[{idx}/{len(barras)}] {barra}")
        
        try:
            df_res, mejor_res = optimizar_hiperparametros(
                resultados_dict, 
                barra_nombre=barra, 
                parametros=parametros, 
                metrica_objetivo=metrica_objetivo,
                usar_cv=usar_cv, 
                n_splits=n_splits, 
                hmap=hmap
            )

            mejores_params[barra] = {
                'changepoint_prior_scale': float(mejor_res['changepoint_prior_scale']),
                'seasonality_prior_scale': float(mejor_res['seasonality_prior_scale']),
                'seasonality_mode': mejor_res['seasonality_mode'],
                'growth': mejor_res.get('growth', 'linear'),
                metrica_objetivo: float(mejor_res[metrica_objetivo])
            }
        
        except Exception as e:
            print(f"Error optimizando {barra}: {e}\n")
            continue

    # Guardamos los resultados
    print(f"\n{'='*80}")
    print(f"GUARDANDO RESULTADOS")
    print(f"{'='*80}\n")
    
    with open(ruta_archivo, 'w') as f:
        json.dump(mejores_params, f, indent=4)
    
    print(f"Resultados guardados en: {ruta_archivo}")
    print(f"Barras optimizadas: {len(mejores_params)}/{len(barras)}\n")
    
    return mejores_params

# Función para entrenar modelo Prophet para una barra
def entrenar_prophet_barra(df_train, df_feriados, parametros, 
                           growth='linear', techo=1.5, verbose=True):
    
    df_train_copy = df_train.copy()
    
    if growth == 'logistic':
        max_y = float(df_train_copy['y'].max())
        min_y = float(df_train_copy['y'].min())
        cap_val = float(max_y * techo) if max_y > 0 else 100.0
        floor_val = float(min(0.0, min_y * 1.1))
        
        # Asignación limpia y normal
        df_train_copy['cap'] = cap_val
        df_train_copy['floor'] = floor_val

    # Instanciar Prophet
    modelo = Prophet(growth=growth,
                     daily_seasonality=parametros.get('daily_seasonality', True),
                     weekly_seasonality=parametros.get('weekly_seasonality', True),
                     yearly_seasonality=parametros.get('yearly_seasonality', True),
                     seasonality_mode=parametros.get('seasonality_mode', 'additive'),
                     seasonality_prior_scale=parametros.get('seasonality_prior_scale', 10.0),
                     changepoint_prior_scale=parametros.get('changepoint_prior_scale', 0.05),
                     holidays=df_feriados,
                     interval_width=0.95) 

    # Entrenar
    modelo.fit(df_train_copy)
    
    # Parche anti-bug para 'logistic'
    if growth == 'logistic':
        for param in ['k', 'm', 'sigma_obs']:
            if param in modelo.params and modelo.params[param].ndim == 2:
                # Aplastamos la matriz de (1, 1) a (1,)
                # Al indexar esto después, Prophet obtendrá un número puro y no explotará.
                modelo.params[param] = modelo.params[param].ravel()

    return modelo


# Función para predecir con Prophet
def predecir_prophet_barra(modelo, df_train, df_val, df_test,
                           growth='linear', techo=1.5, verbose=True):
    
    predicciones = {}

    if growth == 'logistic':
        max_y = float(df_train['y'].max())
        min_y = float(df_train['y'].min())
        cap_val = float(max_y * techo) if max_y > 0 else 100.0
        floor_val = float(min(0.0, min_y * 1.1))

    for nombre, df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        
        df_copy = df.copy()
        
        if growth == 'logistic':
            df_copy['cap'] = cap_val
            df_copy['floor'] = floor_val
            cols_pred = ['ds', 'cap', 'floor']
        else:
            cols_pred = ['ds']
        
        forecast = modelo.predict(df_copy[cols_pred])
        predicciones[nombre] = forecast
        
        if verbose:
            print(f"  Predicción en {nombre}: {len(forecast)} filas")
    
    return predicciones

# Métricas
def calcular_metricas(y_real, y_pred, conjunto=''):
    
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mape = round(mean_absolute_percentage_error(y_real, y_pred), 6)
    r2 = r2_score(y_real, y_pred)
    
    return {'MAE': mae,
            'RMSE': rmse,
            'MSE': mse,
            'MAPE': mape,
            'R2': r2}

# Generara datafrane final para una barra
def generar_dataframe_final(predicciones_dict, df_dict, barra, verbose=True):

    dfs_por_conjunto = []    
    for conjunto in ['train', 'val', 'test']:

        # Obtener datos
        df_real = df_dict[f'df_{conjunto}'].copy()
        forecast = predicciones_dict[conjunto].copy()
        
        # Validar alineación
        if len(df_real) != len(forecast):
            raise ValueError(f"Tamaño mismatch en {barra} {conjunto}: "
                             f"df_real={len(df_real)}, forecast={len(forecast)}")
        
        # Crear dataframe intermedio
        temp = pd.DataFrame({
            'ds': df_real['ds'].values,
            'y_real': df_real['y'].values,
            'yhat': forecast['yhat'].values,
            'yhat_lower': forecast['yhat_lower'].values,
            'yhat_upper': forecast['yhat_upper'].values,
            'trend': forecast['trend'].values,
            'Barra': barra,
            'Conjunto': conjunto.capitalize()
        })
        
        # Agregar estacionalidades si existen
        if 'yearly' in forecast.columns:
            temp['yearly'] = forecast['yearly'].values
        if 'weekly' in forecast.columns:
            temp['weekly'] = forecast['weekly'].values
        if 'daily' in forecast.columns:
            temp['daily'] = forecast['daily'].values
        
        # Calcular residuo (SOLO en train/val, NO en test)
        if conjunto in ['train', 'val']:
            temp['residuo'] = temp['y_real'] - temp['yhat']
        else:
            # En test, no podemos calcular residuo (no sabemos y_real futuro)
            temp['residuo'] = np.nan
        
        dfs_por_conjunto.append(temp)
        
        if verbose:
            print(f"  Dataframe {conjunto}: {len(temp)} filas")
    
    # Concatenar
    df_final = pd.concat(dfs_por_conjunto, ignore_index=True)
    
    return df_final

# Mostrar parámetros de entrenamiento
def mostrar_parametros_barra(barra, params_barra, growth='linear'):
    
    df_params = pd.DataFrame({
        'Barra': [barra],
        'Growth': [growth],
        'Changepoint Prior Scale': [params_barra.get('changepoint_prior_scale', 'N/A')],
        'Seasonality Prior Scale': [params_barra.get('seasonality_prior_scale', 'N/A')],
        'Seasonality Mode': [params_barra.get('seasonality_mode', 'N/A')],
        'Daily Seasonality': [params_barra.get('daily_seasonality', True)],
        'Weekly Seasonality': [params_barra.get('weekly_seasonality', True)],
        'Yearly Seasonality': [params_barra.get('yearly_seasonality', True)]
    })
    
    display(df_params)
    
    return df_params

# Visualizar resultados
def visualizar_resultados(df_barra, modelo, barra, mostrar_componentes=True):
    
    # Definir colores
    color_real = 'gray'
    color_train = '#1f77b4'      # Azul
    color_val = '#ff7f0e'        # Naranja
    color_test = '#2ca02c'       # Verde
    color_intervalo = "#62d8f6"  # Rojo claro #ff6b9d
    
    # GRÁFICA 1: Serie completa y zooms
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    
    # Subplot 1: SERIE COMPLETA
    ax = axes[0, 0]
    
    # Línea de valores reales (Negro, siempre visible)
    ax.plot(df_barra['ds'], df_barra['y_real'], '-', 
            color=color_real, alpha=0.9, label='Real', 
            markersize=3, linewidth=1.5)
    
    # Predicciones por conjunto (colores distintos)
    for conjunto, color in [('Train', color_train), ('Val', color_val), ('Test', color_test)]:
        df_conj = df_barra[df_barra['Conjunto'] == conjunto]
        if not df_conj.empty:
            ax.plot(df_conj['ds'], df_conj['yhat'], 
                   color=color, linewidth=2.2, alpha=1,
                   label=f'Pred {conjunto}')
    
    # Intervalo de confianza (Rojo claro, solo para test)
    ax.fill_between(df_barra['ds'], 
                    df_barra['yhat_lower'], df_barra['yhat_upper'],
                    color=color_intervalo, alpha=0.4, 
                    label='Intervalo (95%)')
    
    ax.set_title(f'Serie Completa', fontweight='bold', fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylabel('Precio (USD/MWh)', fontsize=10)
    
    # Subplot 2: ZOOM TRAIN
    ax = axes[0, 1]
    df_train = df_barra[df_barra['Conjunto'] == 'Train']
    if not df_train.empty:
        # Real
        ax.plot(df_train['ds'], df_train['y_real'], '-', 
               color=color_real, alpha=0.9, label='Real',
               markersize=4, linewidth=1.5)
        
        # Predicción Train
        ax.plot(df_train['ds'], df_train['yhat'], 
               color=color_train, alpha=1, label='Predicción',
               linewidth=2.5)
        
        # Intervalo
        ax.fill_between(df_train['ds'], 
                       df_train['yhat_lower'], df_train['yhat_upper'],
                       color=color_intervalo, alpha=0.4, label='Intervalo (95%)')
    
    ax.set_title('TRAIN', fontweight='bold', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylabel('Precio (USD/MWh)', fontsize=10)
    
    # Subplot 3: ZOOM VALIDACIÓN
    ax = axes[1, 0]
    df_val = df_barra[df_barra['Conjunto'] == 'Val']
    if not df_val.empty:
        # Real
        ax.plot(df_val['ds'], df_val['y_real'], '-', 
               color=color_real, alpha=0.9, label='Real',
               markersize=4, linewidth=1.5)
        
        # Predicción Val
        ax.plot(df_val['ds'], df_val['yhat'], 
               color=color_val, linewidth=2.5, alpha=1,
               label='Predicción')
        
        # Intervalo
        ax.fill_between(df_val['ds'], 
                       df_val['yhat_lower'], df_val['yhat_upper'],
                       color=color_intervalo, alpha=0.4, label='Intervalo (95%)')
    
    ax.set_title('VALIDACIÓN', fontweight='bold', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylabel('Precio (USD/MWh)', fontsize=10)
    
    # Subplot 4: ZOOM TEST
    ax = axes[1, 1]
    df_test = df_barra[df_barra['Conjunto'] == 'Test']
    
    if not df_test.empty:
        # Real
        ax.plot(df_test['ds'], df_test['y_real'], '-', 
               color=color_real, alpha=1, label='Real',
               markersize=4, linewidth=1.5)
        
        # Predicción Test
        ax.plot(df_test['ds'], df_test['yhat'], 
               color=color_test, linewidth=2.5, alpha=1,
               label='Predicción')
        
        # Intervalo
        ax.fill_between(df_test['ds'], 
                       df_test['yhat_lower'], df_test['yhat_upper'],
                       color=color_intervalo, alpha=0.4, label='Intervalo (95%)')
    
    ax.set_title('TEST', fontweight='bold', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylabel('Precio (USD/MWh)', fontsize=10)
    
    plt.suptitle(f'{barra} - Predicciones de Prophet', 
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    # GRÁFICA 2: COMPONENTES
    if mostrar_componentes and modelo is not None:
        try:
            # Crear dataframe mínimo para plot_components
            df_for_plot = df_barra[['ds']].drop_duplicates().reset_index(drop=True)
            
            # SOLUCIÓN: Si es logístico, rescatamos el cap y floor de la memoria del modelo
            if getattr(modelo, 'growth', 'linear') == 'logistic':
                df_for_plot['cap'] = modelo.history['cap'].iloc[0]
                df_for_plot['floor'] = modelo.history['floor'].iloc[0]
            
            # Hacer predicción (para obtener componentes)
            forecast_for_plot = modelo.predict(df_for_plot)
            
            # Graficar componentes
            fig = modelo.plot_components(forecast_for_plot)
            plt.suptitle(f'{barra} - Componentes de Prophet', 
                         fontsize=15, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"  No se pudieron graficar componentes: {e}")

# Función para cargar una barra especifica de un modelo Prophet guardado 
def cargar_modelo_prophet(barra, carpeta_modelos='Modelos_Prophet/h/modelos'):

    ruta_modelo = os.path.join(carpeta_modelos, f'modelo_{barra}.pkl')
    
    if os.path.exists(ruta_modelo):
        with open(ruta_modelo, 'rb') as f:
            modelo = pickle.load(f)
        print(f"  Modelo cargado: {barra}")
        return modelo
    else:
        print(f"  Modelo no encontrado: {barra}")
        return None

# Guarda un modelo Prophet
def guardar_modelo_prophet(modelo, barra, carpeta_modelos='Modelos_Prophet/h/modelos'):

    os.makedirs(carpeta_modelos, exist_ok=True)
    ruta_modelo = os.path.join(carpeta_modelos, f'modelo_{barra}.pkl')
    
    with open(ruta_modelo, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"  Modelo guardado: {barra}")

# Función para cargar todos los modelos de una lista de barras
def cargar_todos_modelos(barras, carpeta_modelos='Modelos_Prophet/h/modelos'):

    modelos_cargados = {}
    
    for barra in barras:
        modelo = cargar_modelo_prophet(barra, carpeta_modelos)
        if modelo is not None:
            modelos_cargados[barra] = modelo
    
    return modelos_cargados

# Función principal modificada
def multi_prophet(prepro_por_barras, parametros, params_base,
                  modo='entrenar', mostrar_params=True, 
                  mostrar_graficos=True, mostrar_componentes=True, 
                  growth='linear', techo=1.5, 
                  carpeta_modelos='Modelos_Prophet/h/modelos'):

    barras = list(prepro_por_barras.keys())
    n_barras = len(barras)
    
    print(f"\n{'='*80}")
    print(f"MULTI PROPHET MODULAR - {n_barras} BARRAS")
    print(f"Modo: {modo.upper()}")
    print(f"{'='*80}\n")
    
    modelos_entrenados = {}
    dfs_resultados = []
    metricas_lista = []
    
    # Modo Cargar: Cargar modelos existentes
    if modo == 'cargar':
        print("Cargando modelos entrenados desde disco...")
        modelos_entrenados = cargar_todos_modelos(barras, carpeta_modelos)
        
        if len(modelos_entrenados) != n_barras:
            print(f"  Se cargaron {len(modelos_entrenados)}/{n_barras} modelos")
            print("   Los modelos faltantes se entrenarán...")
            modo = 'híbrido'  # Modo para entrenar los faltantes
    
    # Loop
    for idx, barra in enumerate(barras, 1):
        print(f"[{idx}/{n_barras}] Procesando: {barra}")
        
        try:
            # Obtener datos
            data_barra = prepro_por_barras[barra]
            df_train = data_barra['df_train'].copy()
            df_val = data_barra['df_val'].copy()
            df_test = data_barra['df_test'].copy()
            df_feriados = data_barra['df_feriados'].copy()
            
            # Obtener parámetros
            params_barra = parametros.get(barra, params_base)

            # OBTENER MODELO (cargado o entrenado)
            modelo = None
            
            # Intentar cargar si está en modo cargar o híbrido
            if modo in ['cargar', 'híbrido'] and barra in modelos_entrenados:
                modelo = modelos_entrenados[barra]
            
            # Entrenar si no se pudo cargar o modo es entrenar
            if modelo is None:
                if mostrar_params:
                    print(f"\n  Parámetros de entrenamiento:")
                    mostrar_parametros_barra(
                        barra=barra,
                        params_barra=params_barra,
                        growth=growth)
                
                print(f"  Entrenando modelo...")
                modelo = entrenar_prophet_barra(
                    df_train=df_train,
                    df_feriados=df_feriados,
                    parametros=params_barra,
                    growth=growth,
                    techo=techo,
                    verbose=False
                )
                modelos_entrenados[barra] = modelo
                
                # Guardar modelo entrenado
                guardar_modelo_prophet(modelo, barra, carpeta_modelos)
            
            # Predecir
            predicciones = predecir_prophet_barra(
                modelo=modelo,
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                growth=growth,
                techo=techo,
                verbose=False
            )
            
            # Generar dataframe final
            df_barra = generar_dataframe_final(
                predicciones_dict=predicciones,
                df_dict={
                    'df_train': df_train,
                    'df_val': df_val,
                    'df_test': df_test
                },
                barra=barra,
                verbose=False
            )
            dfs_resultados.append(df_barra)
            
            # Calcular métricas
            for conjunto in ['train', 'val', 'test']:
                df_conj = df_barra[df_barra['Conjunto'] == conjunto.capitalize()]
                
                if not df_conj.empty:
                    metricas = calcular_metricas(
                        y_real=df_conj['y_real'].values,
                        y_pred=df_conj['yhat'].values,
                        conjunto=conjunto
                    )
                    
                    metricas_lista.append({
                        'Barra': barra,
                        'Conjunto': conjunto.capitalize(),
                        **metricas
                    })
            
            # Visualizar resultados
            if mostrar_graficos:
                visualizar_resultados(
                    df_barra=df_barra,
                    modelo=modelo,
                    barra=barra,
                    mostrar_componentes=mostrar_componentes)
            
        except Exception as e:
            print(f"  ERROR en {barra}: {e}\n")
            continue
    
    # Consolidar resultados
    df_resultados_total = pd.concat(dfs_resultados, ignore_index=True) if dfs_resultados else pd.DataFrame()
    df_metricas_total = pd.DataFrame(metricas_lista) if metricas_lista else pd.DataFrame()
    
    print(f"\n{'='*80}")
    print(f"RESUMEN FINAL")
    print(f"{'='*80}")
    print(f"Total de filas: {len(df_resultados_total)}")
    print(f"Barras procesadas: {len(modelos_entrenados)}/{n_barras}")
    
    if not df_metricas_total.empty:
        df_display = df_metricas_total.copy()
        df_display['Barra'] = df_display['Barra'].mask(df_display['Barra'].duplicated(), '')

        # Redondear métricas y formatear
        df_display['MAE'] = df_display['MAE'].round(4)
        df_display['RMSE'] = df_display['RMSE'].round(4)
        df_display['MSE'] = df_display['MSE'].round(4)
        if 'MAPE' in df_display.columns:
            df_display['MAPE'] = (df_display['MAPE'] * 100).round(4).astype(str) + '%'
        df_display['R2'] = df_display['R2'].round(4)

        display(df_display)
    
    return modelos_entrenados, df_resultados_total, df_metricas_total