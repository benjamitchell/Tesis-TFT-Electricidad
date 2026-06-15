import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import Ridge
import time
import torch
from IPython.display import display

# ======================== FUNCIONES PARA STACKING OPTIMIZATION ==============================

# Función para calcular métricas de evaluación
def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    return {'MAE': mae, 'RMSE': np.sqrt(mse), 'MSE': mse, 'R2': r2}

# Función para extraer predicciones con  desfase y posibles índices faltantes
def extraer_predicciones_con_desfase(barra, modelo, dataloader,
                                     datasets_norm, diccionario_scalers,
                                     conjunto,
                                     target_key='y_real',
                                     scaler_key=None):
    
    if scaler_key is None:
        scaler_key = target_key
    
    # Dataset completo
    df_train = datasets_norm[barra]['train']
    df_val = datasets_norm[barra]['val']
    df_test = datasets_norm[barra]['test']
    df_completo = pd.concat([df_train, df_val, df_test], ignore_index=True)
    fechas_completo = pd.to_datetime(df_completo['ds'].values)
    
    # Scalers
    scaler = diccionario_scalers[barra]['scalers_target'][scaler_key]
    scaler_reales = diccionario_scalers[barra]['scalers_target'][target_key]
    
    # Determinar conjunto
    if conjunto == 'train':
        n_conjunto = len(df_train)
        inicio_conjunto = 0
        fin_conjunto = n_conjunto
    elif conjunto == 'val':
        n_conjunto = len(df_val)
        inicio_conjunto = len(df_train)
        fin_conjunto = len(df_train) + len(df_val)
    else:
        n_conjunto = len(df_test)
        inicio_conjunto = len(df_train) + len(df_val)
        fin_conjunto = len(df_completo)
    
    # Valores reales del conjunto
    y_real_norm = df_completo[target_key].values
    y_real_completo = scaler_reales.inverse_transform(y_real_norm.reshape(-1, 1)).flatten()
    
    if conjunto == 'test':
        y_real_conjunto = y_real_completo[-n_conjunto:]
        fechas_conjunto = fechas_completo[-n_conjunto:]
    else:
        y_real_conjunto = y_real_completo[inicio_conjunto:fin_conjunto]
        fechas_conjunto = fechas_completo[inicio_conjunto:fin_conjunto]
    
    # Obtener todas las predicciones
    with torch.no_grad():
        raw = modelo.predict(dataloader, mode="raw", return_y=True, return_index=True)
    
    predictions = raw.output.prediction if hasattr(raw.output, 'prediction') else raw.output[0]
    
    if len(predictions.shape) == 3 and predictions.shape[2] == 7:
        predictions = predictions[:, :, 3]
    
    y_pred_norm = predictions[:, 0].cpu().numpy()
    y_pred_completo = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    
    # Extraer time_idx
    try:
        time_idx = raw.index['time_idx'].cpu().numpy()
    except AttributeError:
        time_idx = raw.index['time_idx'].values
    
    # Mapear posiciones locales del conjunto (vectorizado)
    y_pred_mapeado = np.full(n_conjunto, np.nan)

    mask_conjunto = (time_idx >= inicio_conjunto) & (time_idx < fin_conjunto)
    idx_locales = time_idx[mask_conjunto] - inicio_conjunto
    y_pred_mapeado[idx_locales] = y_pred_completo[mask_conjunto]
    
   # Filtrar solo valores válidos
    mask_validos = ~np.isnan(y_pred_mapeado)
    
    y_pred = y_pred_mapeado[mask_validos]
    y_real = y_real_conjunto[mask_validos]
    fechas = fechas_conjunto[mask_validos]
    
    # Obtener índices válidos
    indices_validos = np.where(mask_validos)[0]
    
    return {'y_pred': y_pred,
            'y_real': y_real,
            'fechas': fechas,
            'n_puntos': len(y_pred),
            'indices_validos': indices_validos}

# Función para extraer predicciones de Prophet
def extraer_predicciones_prophet(barra, datasets_norm, diccionario_scalers, conjunto, indices_validos=None):
    
    # Obtener dataset del conjunto
    df_conjunto = datasets_norm[barra][conjunto]
    
    # Obtener scaler
    scaler_yhat = diccionario_scalers[barra]['scalers_target'].get('yhat')
    if scaler_yhat is None:
        scaler_yhat = diccionario_scalers[barra]['scalers_target']['y_real']
    
    scaler_y_real = diccionario_scalers[barra]['scalers_target']['y_real']
    
    # Desnormalizar predicciones de Prophet
    yhat_norm = df_conjunto['yhat'].values
    y_pred = scaler_yhat.inverse_transform(yhat_norm.reshape(-1, 1)).flatten()
    
    # Desnormalizar valores reales
    y_real_norm = df_conjunto['y_real'].values
    y_real = scaler_y_real.inverse_transform(y_real_norm.reshape(-1, 1)).flatten()
    
    # Fechas
    fechas = pd.to_datetime(df_conjunto['ds'].values)
    
    # Filtrata por indices válidos
    if indices_validos is not None:
        y_pred = y_pred[indices_validos]
        y_real = y_real[indices_validos]
        fechas = fechas[indices_validos]
    
    return {'y_pred': y_pred,
            'y_real': y_real,
            'fechas': fechas,
            'n_puntos': len(y_pred)}

# Función para realizar grid search de pesos
def grid_search_modelos(y_real, pred1, pred2, resolution, metrica='MAE'):
    
     # Grid de w1
    w1_grid = np.linspace(0, 1, resolution)     # Restricción w1, w2 >= 0 

    best_score = np.inf
    best_w1 = 0.5
    best_w2 = 0.5
    
    start_time = time.time()
    for w1 in w1_grid:
        w2 = 1 - w1                             # Restrricción w1 + w2 = 1
        y_pred = w1 * pred1 + w2 * pred2        # Predicción
        
        # Métrica a optimizar
        score = calcular_metricas(y_real, y_pred)[metrica]
        
        if score < best_score:
            best_score = score
            best_w1 = w1
            best_w2 = w2
            best_y_pred = y_pred
    
    end_time = time.time()
    elapsed = end_time - start_time
    metricas = calcular_metricas(y_real, best_y_pred)
    
    return {'w1': best_w1, 'w2': best_w2,
            'MAE': metricas['MAE'], 'RMSE': metricas['RMSE'], 'MSE': metricas['MSE'], 'R2': metricas['R2'],
            'time': elapsed}

# Función para hacer grafico de lineas
def grafico_lineas(df_resultados, modo='estrategias', metricas=None, 
                   nombre_modelo='TFT', conjunto='Test',
                   linestyles=['-'], colormap='tab10',
                   colores_personalizados=None, figsize=None):

    # Validar modo
    if modo not in ['estrategias', 'metodos']:
        raise ValueError("modo debe ser 'estrategias' o 'metodos'")
    
    # Configurar métricas    
    if metricas is None:
        if modo == 'estrategias':
            metricas = ['MAE', 'RMSE', 'R²']
        else:  # metodos
            metricas = ['MAE', 'RMSE', 'R²', 'Tiempo (s)']
    elif isinstance(metricas, str):
        metricas = [metricas]
    elif not isinstance(metricas, list):
        metricas = [metricas]
    
    # Filtrar y extraer dato 
    if modo == 'estrategias':
        # No filtrar por conjunto (usa Train/Val/Test como columnas)
        df_filtrado = df_resultados.copy()
        columnas = ['Train', 'Val', 'Test']
        lineas = sorted(df_filtrado['Estrategia'].unique())
        columna_linea = 'Estrategia'
        columna_columna = 'Set'
        titulo_base = 'Métricas por Barra y Conjunto'
        
    elif modo == 'metodos':
        # Filtrar por conjunto
        df_filtrado = df_resultados[df_resultados['Set'] == conjunto].copy()
        columnas = sorted(df_filtrado['Estrategia'].unique())
        lineas = sorted(df_filtrado['Método'].unique())
        columna_linea = 'Método'
        columna_columna = 'Estrategia'
        titulo_base = f'Comparación de Métodos ({conjunto})'
    
    # Barras (eje X)
    barras = sorted(df_filtrado['Barra'].unique())
    
    n_metricas = len(metricas)
    n_columnas = len(columnas)
    n_barras = len(barras)
    n_lineas = len(lineas)
    
    # Configurar figura
    if figsize is None:
        ancho = 7 * n_columnas
        alto = 5 * n_metricas
        figsize = (ancho, alto)
    
    fig, axes = plt.subplots(n_metricas, n_columnas, figsize=figsize)
    
    # Asegurar que axes sea 2D
    if n_metricas == 1 and n_columnas == 1:
        axes = np.array([[axes]])
    elif n_metricas == 1:
        axes = axes.reshape(1, -1)
    elif n_columnas == 1:
        axes = axes.reshape(-1, 1)
    
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Configurar colores
    if colores_personalizados is not None:
        if isinstance(colores_personalizados, dict):
            color_map = colores_personalizados
        elif isinstance(colores_personalizados, list):
            color_map = {linea: colores_personalizados[i % len(colores_personalizados)]
                        for i, linea in enumerate(lineas)}
        else:
            raise ValueError("colores_personalizados debe ser dict o list")
    else:
        try:
            cmap = plt.cm.get_cmap(colormap)
            colors = cmap(np.linspace(0, 1, n_lineas))
            color_map = {linea: colors[i] for i, linea in enumerate(lineas)}
        except:
            print(f"Colormap '{colormap}' no válido, usando 'tab10'")
            colors = plt.cm.tab10(np.linspace(0, 1, n_lineas))
            color_map = {linea: colors[i] for i, linea in enumerate(lineas)}
    
    # Estilos de línea
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    
    linestyle_map = {linea: linestyles[i % len(linestyles)]
                     for i, linea in enumerate(lineas)}
    
    # Función para acotar nombres
    def acortar_nombre(nombre, es_estrategia=True):
        if es_estrategia:
            mapeo = {
                'Prophet Solo': 'Prophet',
                f'{nombre_modelo} Precios Solo': f'{nombre_modelo}_P',
                f'{nombre_modelo} Residuos Solo': f'{nombre_modelo}_R',
                f'Prophet + {nombre_modelo} Precios': f'P+{nombre_modelo}_P',
                'Prophet + Residuos (Directo)': 'P+R(D)',
                'Prophet + Residuos (Opt)': 'P+R(O)',
                f'(Prophet + Residuos) + {nombre_modelo} Precios': f'(P+R)+{nombre_modelo}_P'
            }
            
            if nombre in mapeo:
                return mapeo[nombre]
            
            for clave, valor in mapeo.items():
                if clave in nombre:
                    return valor
            
            return nombre.replace('Prophet', 'P') \
                        .replace('Residuos', 'R') \
                        .replace('Precios', 'P')[:50]
        else:
            # Para columnas (estrategias en modo métodos)
            return nombre.replace('Prophet + ', 'P+') \
                        .replace(f'{nombre_modelo} ', '') \
                        .replace('Precios', 'P') \
                        .replace('Residuos', 'R')[:50]
    
    for i_met, metrica in enumerate(metricas):
        
        for i_col, columna_valor in enumerate(columnas):
            ax = axes[i_met, i_col]
            
            # Filtrar datos para esta columna
            df_col = df_filtrado[df_filtrado[columna_columna] == columna_valor].copy()
            
            # Plotear cada línea
            for linea_valor in lineas:
                df_linea = df_col[df_col[columna_linea] == linea_valor].copy()
                
                # Obtener valores por barra
                valores = []
                for barra in barras:
                    df_barra = df_linea[df_linea['Barra'] == barra]
                    if len(df_barra) > 0:
                        valores.append(df_barra[metrica].values[0])
                    else:
                        valores.append(np.nan)
                
                # Nombre para leyenda
                if modo == 'estrategias':
                    nombre_leyenda = acortar_nombre(linea_valor, es_estrategia=True)
                else:
                    nombre_leyenda = linea_valor  # Métodos se quedan igual
                
                # Plotear
                ax.plot(range(n_barras), valores,
                       marker='o', markersize=7, linewidth=2.5,
                       label=nombre_leyenda,
                       color=color_map[linea_valor],
                       linestyle=linestyle_map[linea_valor],
                       alpha=0.85)
            
            # Título solo en primera fila
            if i_met == 0:
                if modo == 'estrategias':
                    titulo = columna_valor  # Train, Val, Test
                else:
                    titulo = acortar_nombre(columna_valor, es_estrategia=False)
                
                ax.set_title(titulo, fontsize=12, fontweight='bold', pad=10)
            
            # Y-label en primera columna
            if i_col == 0:
                ax.set_ylabel(metrica, fontsize=11, fontweight='bold')
            
            # X-labels en todos los plots
            ax.set_xticks(range(n_barras))
            ax.set_xticklabels(barras, rotation=45, ha='right', fontsize=10)
            ax.set_xlabel('Barra', fontsize=10)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
            ax.set_axisbelow(True)
            
            # Leyenda en última columna, fila del medio
            fila_medio = n_metricas // 2
            if i_met == fila_medio and i_col == n_columnas - 1:
                titulo_leyenda = 'Estrategias' if modo == 'estrategias' else 'Métodos'
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                         fontsize=9, framealpha=0.95,
                         title=titulo_leyenda, title_fontsize=10)
    
    if n_metricas == 1:
        titulo_final = f'{titulo_base} - {metricas[0]}\n{nombre_modelo}'
    else:
        titulo_final = f'{titulo_base}\n{nombre_modelo}'
 
    plt.suptitle(titulo_final, fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.show()

# Función para Stacking Optimization de modelos TFT + Prophet
def stacking_optimization(lista_barras,
                          modelos_precios,
                          modelos_residuos,
                          dataloaders_precios,
                          dataloaders_residuos,
                          datasets_norm,
                          diccionario_scalers,
                          nombre_modelo='TFT',
                          grid_resolution=41,
                          metrica_optimizacion='MAE',
                          metricas_grafico=None,
                          linestyles=['-'],
                          colormap = 'cividis',
                          colores_personalizados=None,
                          datos_evaluacion=None):

    resultados_stacking = {}
    tamanios_conjuntos = {}
    
    for barra in lista_barras:

        print(f"Procesando: {barra}")
        
        # Extraer Predicciones
        resultados_stacking[barra] = {}

        if datos_evaluacion is not None:
            # Reusar predicciones ya calculadas por evaluar_modelos_tft (0 inferencias extra)
            def _pred(conj, tipo):
                d = datos_evaluacion[barra][conj][tipo]['datos']
                return {
                    'y_pred':          d['y_pred'],
                    'y_real':          d['y_real'],
                    'fechas':          d['fechas'],
                    'n_puntos':        d['n_validos'],
                    'indices_validos': d['indices_validos'],
                }
            datos_train_precios  = _pred('train', 'Precios')
            datos_train_residuos = _pred('train', 'Residuos')
            datos_val_precios    = _pred('val',   'Precios')
            datos_val_residuos   = _pred('val',   'Residuos')
            datos_test_precios   = _pred('test',  'Precios')
            datos_test_residuos  = _pred('test',  'Residuos')
        else:
            # TRAIN SET - Precios
            datos_train_precios = extraer_predicciones_con_desfase(barra=barra,
                                                                   modelo=modelos_precios[barra],
                                                                   dataloader=dataloaders_precios[barra]['train'],
                                                                   datasets_norm=datasets_norm,
                                                                   diccionario_scalers=diccionario_scalers,
                                                                   conjunto='train')
            # TRAIN SET - Residuos
            datos_train_residuos = extraer_predicciones_con_desfase(barra=barra,
                                                                    modelo=modelos_residuos[barra],
                                                                    dataloader=dataloaders_residuos[barra]['train'],
                                                                    datasets_norm=datasets_norm,
                                                                    diccionario_scalers=diccionario_scalers,
                                                                    conjunto='train',
                                                                    target_key='residuo',
                                                                    scaler_key='residuo')
            # VAL SET - Precios
            datos_val_precios = extraer_predicciones_con_desfase(barra=barra,
                                                                 modelo=modelos_precios[barra],
                                                                 dataloader=dataloaders_precios[barra]['val'],
                                                                 datasets_norm=datasets_norm,
                                                                 diccionario_scalers=diccionario_scalers,
                                                                 conjunto='val')
            # VAL SET - Residuos
            datos_val_residuos = extraer_predicciones_con_desfase(barra=barra,
                                                                  modelo=modelos_residuos[barra],
                                                                  dataloader=dataloaders_residuos[barra]['val'],
                                                                  datasets_norm=datasets_norm,
                                                                  diccionario_scalers=diccionario_scalers,
                                                                  conjunto='val',
                                                                  target_key='residuo',
                                                                  scaler_key='residuo')
            # TEST SET - Precios
            datos_test_precios = extraer_predicciones_con_desfase(barra=barra,
                                                                  modelo=modelos_precios[barra],
                                                                  dataloader=dataloaders_precios[barra]['test'],
                                                                  datasets_norm=datasets_norm,
                                                                  diccionario_scalers=diccionario_scalers,
                                                                  conjunto='test')
            # TEST SET - Residuos
            datos_test_residuos = extraer_predicciones_con_desfase(barra=barra,
                                                                   modelo=modelos_residuos[barra],
                                                                   dataloader=dataloaders_residuos[barra]['test'],
                                                                   datasets_norm=datasets_norm,
                                                                   diccionario_scalers=diccionario_scalers,
                                                                   conjunto='test',
                                                                   target_key='residuo',
                                                                   scaler_key='residuo')

        # Prophet en los 3 conjuntos
        datos_prophet_train = extraer_predicciones_prophet(barra, 
                                                           datasets_norm, 
                                                           diccionario_scalers, 
                                                           'train',
                                                           indices_validos=datos_train_precios['indices_validos'])

        datos_prophet_val = extraer_predicciones_prophet(barra, 
                                                         datasets_norm, 
                                                         diccionario_scalers, 
                                                         'val',
                                                         indices_validos=datos_val_precios['indices_validos'])

        datos_prophet_test = extraer_predicciones_prophet(barra, 
                                                          datasets_norm, 
                                                          diccionario_scalers, 
                                                          'test',
                                                          indices_validos=datos_test_precios['indices_validos'])

        # Determinar tamaño del subset común
        n_train = min(len(datos_train_precios['y_pred']), len(datos_prophet_train['y_pred']))
        n_val = min(len(datos_val_precios['y_pred']), len(datos_prophet_val['y_pred']))
        n_test = min(len(datos_test_precios['y_pred']), len(datos_prophet_test['y_pred']))

        # Guardar tamaños
        tamanios_conjuntos[barra] = {'train': n_train, 'val': n_val, 'test': n_test}

        # TRAIN - Alinear todo
        y_real_train = datos_prophet_train['y_real'][:n_train]
        prophet_train = datos_prophet_train['y_pred'][:n_train]
        tft_precios_train = datos_train_precios['y_pred'][:n_train]
        tft_residuos_train = datos_train_residuos['y_pred'][:n_train]
        fechas_train = datos_prophet_train['fechas'][:n_train]

        # VAL - Alinear todo
        y_real_val = datos_prophet_val['y_real'][:n_val]
        prophet_val = datos_prophet_val['y_pred'][:n_val]
        tft_precios_val = datos_val_precios['y_pred'][:n_val]
        tft_residuos_val = datos_val_residuos['y_pred'][:n_val]
        fechas_val = datos_prophet_val['fechas'][:n_val]

        # TEST - Alinear todo
        y_real_test = datos_prophet_test['y_real'][:n_test]
        prophet_test = datos_prophet_test['y_pred'][:n_test]
        tft_precios_test = datos_test_precios['y_pred'][:n_test]
        tft_residuos_test = datos_test_residuos['y_pred'][:n_test]
        fechas_test = datos_prophet_test['fechas'][:n_test]

        # ESTRATEGIA 1: PROPHET SOLO
        metricas_prophet_train = calcular_metricas(y_real_train, prophet_train)
        metricas_prophet_val = calcular_metricas(y_real_val, prophet_val)
        metricas_prophet_test = calcular_metricas(y_real_test, prophet_test)

        resultados_stacking[barra]['Prophet_Solo'] = {
            'nombre': 'Prophet Solo',
            'pesos': None,
            'prediccion_train': prophet_train,
            'prediccion_val': prophet_val,    
            'prediccion_test': prophet_test,  
            'metricas_train': metricas_prophet_train,
            'metricas_val': metricas_prophet_val,
            'metricas_test': metricas_prophet_test
        }

        # ESTRATEGIA 2: TFT PRECIOS SOLO 
        metricas_tft_train = calcular_metricas(y_real_train, tft_precios_train)
        metricas_tft_val = calcular_metricas(y_real_val, tft_precios_val)
        metricas_tft_test = calcular_metricas(y_real_test, tft_precios_test)

        resultados_stacking[barra]['TFT_Precios_Solo'] = {
            'nombre': f'{nombre_modelo} Precios Solo',
            'pesos': None,
            'prediccion_train': tft_precios_train,
            'prediccion_val': tft_precios_val,
            'prediccion_test': tft_precios_test,
            'metricas_train': metricas_tft_train,
            'metricas_val': metricas_tft_val,
            'metricas_test': metricas_tft_test
        }

        # ESTRATEGIA 2.5: TFT RESIDUOS SOLO
        metricas_tft_res_train = calcular_metricas(y_real_train, tft_residuos_train)
        metricas_tft_res_val = calcular_metricas(y_real_val, tft_residuos_val)
        metricas_tft_res_test = calcular_metricas(y_real_test, tft_residuos_test)

        resultados_stacking[barra]['TFT_Residuos_Solo'] = {
            'nombre': f'{nombre_modelo} Residuos Solo',
            'pesos': None,
            'prediccion_train': tft_residuos_train,
            'prediccion_val': tft_residuos_val,
            'prediccion_test': tft_residuos_test,
            'metricas_train': metricas_tft_res_train,
            'metricas_val': metricas_tft_res_val,
            'metricas_test': metricas_tft_res_test
        }

        # ESTRATEGIA 3: PROPHET + TFT PRECIOS
        result = grid_search_modelos(y_real_train, prophet_train, tft_precios_train,
                                     resolution=grid_resolution, 
                                     metrica=metrica_optimizacion)
        w_prophet, w_tft = result['w1'], result['w2']
        
        pred_train = w_prophet * prophet_train + w_tft * tft_precios_train
        pred_val = w_prophet * prophet_val + w_tft * tft_precios_val
        pred_test = w_prophet * prophet_test + w_tft * tft_precios_test
        
        metricas_train = calcular_metricas(y_real_train, pred_train)
        metricas_val = calcular_metricas(y_real_val, pred_val)
        metricas_test = calcular_metricas(y_real_test, pred_test)
        
        resultados_stacking[barra]['Prophet_TFT_Precios'] = {
            'nombre': f'Prophet + {nombre_modelo} Precios',
            'pesos': {'w_prophet': w_prophet, 'w_tft': w_tft},
            'prediccion_train': pred_train,
            'prediccion_val': pred_val,
            'prediccion_test': pred_test,
            'metricas_train': metricas_train,
            'metricas_val': metricas_val,
            'metricas_test': metricas_test
        }
        
        # ESTRATEGIA 4: PROPHET + TFT RESIDUOS (SUMA DIRECTA)
        pred_train = prophet_train + tft_residuos_train
        pred_val = prophet_val + tft_residuos_val
        pred_test = prophet_test + tft_residuos_test
        
        metricas_train = calcular_metricas(y_real_train, pred_train)
        metricas_val = calcular_metricas(y_real_val, pred_val)
        metricas_test = calcular_metricas(y_real_test, pred_test)
        
        resultados_stacking[barra]['Prophet_Residuos_Directo'] = {
            'nombre': 'Prophet + Residuos (Directo)',
            'pesos': {'w_prophet': 1.0, 'w_residuo': 1.0},
            'prediccion_train': pred_train,
            'prediccion_val': pred_val,
            'prediccion_test': pred_test,
            'metricas_train': metricas_train,
            'metricas_val': metricas_val,
            'metricas_test': metricas_test
        }
        
        # ESTRATEGIA 5: PROPHET + TFT RESIDUOS (OPT)
        result = grid_search_modelos(y_real_train, prophet_train, tft_residuos_train,
                                     resolution=grid_resolution, 
                                     metrica=metrica_optimizacion)

        w_prophet, w_residuo = result['w1'], result['w2']
        
        pred_train = w_prophet * prophet_train + w_residuo * tft_residuos_train
        pred_val = w_prophet * prophet_val + w_residuo * tft_residuos_val
        pred_test = w_prophet * prophet_test + w_residuo * tft_residuos_test
        
        metricas_train = calcular_metricas(y_real_train, pred_train)
        metricas_val = calcular_metricas(y_real_val, pred_val)
        metricas_test = calcular_metricas(y_real_test, pred_test)
        
        resultados_stacking[barra]['Prophet_Residuos_Opt'] = {
            'nombre': 'Prophet + Residuos (Opt)',
            'pesos': {'w_prophet': w_prophet, 'w_residuo': w_residuo},
            'prediccion_train': pred_train,
            'prediccion_val': pred_val,
            'prediccion_test': pred_test,
            'metricas_train': metricas_train,
            'metricas_val': metricas_val,
            'metricas_test': metricas_test
        }

        # ESTRATEGIA 6: (PROPHET + RESIDUOS) + TFT PRECIOS
        prophet_residuos_train = prophet_train + tft_residuos_train
        prophet_residuos_val = prophet_val + tft_residuos_val
        prophet_residuos_test = prophet_test + tft_residuos_test
        
        result = grid_search_modelos(y_real_train, prophet_residuos_train, tft_precios_train,
                                     resolution=grid_resolution, 
                                     metrica=metrica_optimizacion)
        w1, w2 = result['w1'], result['w2']
        
        pred_train = w1 * prophet_residuos_train + w2 * tft_precios_train
        pred_val = w1 * prophet_residuos_val + w2 * tft_precios_val
        pred_test = w1 * prophet_residuos_test + w2 * tft_precios_test
        
        metricas_train = calcular_metricas(y_real_train, pred_train)
        metricas_val = calcular_metricas(y_real_val, pred_val)
        metricas_test = calcular_metricas(y_real_test, pred_test)
        
        resultados_stacking[barra]['Prophet_Residuos_Precios'] = {
            'nombre': f'(Prophet + Residuos) + {nombre_modelo} Precios',
            'pesos': {'w_prophet_residuos': w1, 'w_tft_precios': w2},
            'prediccion_train': pred_train,
            'prediccion_val': pred_val,
            'prediccion_test': pred_test,
            'metricas_train': metricas_train,
            'metricas_val': metricas_val,
            'metricas_test': metricas_test
        }

        # Guardar datos de splits
        resultados_stacking[barra]['_datos_split'] = {
            'y_real_train': y_real_train,
            'y_real_val': y_real_val,
            'y_real_test': y_real_test,
            'fechas_train': fechas_train,
            'fechas_val': fechas_val,
            'fechas_test': fechas_test
        }
        
    # Obtener tamaños únicos para cada conjunto
    trains = set(tam['train'] for tam in tamanios_conjuntos.values())
    vals = set(tam['val'] for tam in tamanios_conjuntos.values())
    tests = set(tam['test'] for tam in tamanios_conjuntos.values())
    
    if len(trains) == 1 and len(vals) == 1 and len(tests) == 1:
        # Todos tienen el mismo tamaño
        n_train = list(trains)[0]
        n_val = list(vals)[0]
        n_test = list(tests)[0]
        print(f"Todos los conjuntos preparados:")
        print(f"   Train: {n_train} puntos  |  Val: {n_val} puntos  |  Test: {n_test} puntos\n")
    else:
        # Hay diferencias
        print("Las barras tienen diferentes tamaños de conjuntos:\n")
        
        for barra in lista_barras:
            tam = tamanios_conjuntos[barra]
            print(f"   {barra:12} → Train: {tam['train']:4}  |  Val: {tam['val']:4}  |  Test: {tam['test']:4}")
        
    # Creamos el dataframe de resultados
    datos_tabla = []
    for barra in lista_barras:
        for key, resultado in resultados_stacking[barra].items():
            if key.startswith('_'):
                continue

            # Filtramos TFT_Residuos_Solo de la tabla
            if key == 'TFT_Residuos_Solo':
                continue
            
            # Pesos
            if resultado['pesos'] is None:
                pesos_str = '-'
            else:
                pesos_list = [f"{k}={v:.2f}" for k, v in resultado['pesos'].items()]
                pesos_str = ', '.join(pesos_list)
            
            # Filas para train/val/test
            for conjunto in ['train', 'val', 'test']:
                datos_tabla.append({
                    'Barra': barra,
                    'Estrategia': resultado['nombre'],
                    'Pesos': pesos_str,
                    'Set': conjunto.capitalize(),
                    'MAE': resultado[f'metricas_{conjunto}']['MAE'],
                    'RMSE': resultado[f'metricas_{conjunto}']['RMSE'],
                    'MSE': resultado[f'metricas_{conjunto}']['MSE'],
                    'R²': resultado[f'metricas_{conjunto}']['R2']
                })
    
    df_comparacion = pd.DataFrame(datos_tabla)
    df_comparacion['MAE'] = df_comparacion['MAE'].round(4)
    df_comparacion['RMSE'] = df_comparacion['RMSE'].round(4)
    df_comparacion['MSE'] = df_comparacion['MSE'].round(4)
    df_comparacion['R²'] = df_comparacion['R²'].round(4)
    
    # Visualizamos los resultados
    grafico_lineas(df_comparacion, 
                   modo='estrategias', 
                   metricas=metricas_grafico, 
                   nombre_modelo=nombre_modelo,
                   linestyles=linestyles, 
                   colormap = colormap, 
                   colores_personalizados=colores_personalizados)

    # Tabla resumen de las mejores estrategias por barra
    print("\n" + "="*80)
    print(f"MEJORES ESTRATEGIAS POR BARRA - {nombre_modelo}")
    print(f"(según {metrica_optimizacion} en TEST)")
    print("="*80 + "\n")
    
    datos_mejores = []
    
    for barra in lista_barras:
        
        mejor_mae_test = np.inf
        mejor_info = None
        
        for key, resultado in resultados_stacking[barra].items():
            if key.startswith('_'):
                continue
            
            mae_test = resultado['metricas_test']['MAE']
            
            if mae_test < mejor_mae_test:
                mejor_mae_test = mae_test
                mejor_info = {
                    'barra': barra,
                    'estrategia': resultado['nombre'],
                    'pesos': resultado['pesos'],
                    'mae_test': resultado['metricas_test']['MAE'],
                    'rmse_test': resultado['metricas_test']['RMSE'],
                    'mse_test': resultado['metricas_test']['MSE'],
                    'r2_test': resultado['metricas_test']['R2']
                }
        
        if mejor_info:
            # Formatear pesos
            if mejor_info['pesos'] is None:
                pesos_str = '-'
            else:
                pesos_list = [f"{k}={v:.3f}" for k, v in mejor_info['pesos'].items()]
                pesos_str = ', '.join(pesos_list)
            
            datos_mejores.append({
                'Barra': mejor_info['barra'],
                'Mejor Estrategia': mejor_info['estrategia'],
                'Pesos': pesos_str,
                'MAE': mejor_info['mae_test'],
                'RMSE': mejor_info['rmse_test'],
                'MSE': mejor_info['mse_test'],
                'R²': mejor_info['r2_test']
            })
    
    # Crear DataFrame de mejores estrategias
    df_mejores = pd.DataFrame(datos_mejores)
    df_mejores['MAE'] = df_mejores['MAE'].round(4)
    df_mejores['RMSE'] = df_mejores['RMSE'].round(4)
    df_mejores['MSE'] = df_mejores['MSE'].round(4)
    df_mejores['R²'] = df_mejores['R²'].round(4)
    
    # Mostrar tabla formateada
    display(df_mejores)
            
    return resultados_stacking, df_comparacion

# ======================== OTROS MÉTODOS DE OPTIMIZACIÓN ==============================

# Función objtivo para optimización con scipy.optimize.minimize
def objective(w, pred1, pred2, y_real, métrica='MAE'):
        w1 = w[0]
        w2 = 1 - w1
        y_pred = w1 * pred1 + w2 * pred2
        score = calcular_metricas(y_real, y_pred)[métrica]
        return score

# Función para optimización con Nelder-Mead usando scipy.optimize.minimize
def scipy_minimize_nelder_mead(y_real, pred1, pred2):
    
    start_time = time.time()
    result = minimize(lambda w: objective(w, pred1, pred2, y_real), x0=[0.5], method='Nelder-Mead', bounds=[(0, 1)])
    elapsed = time.time() - start_time
    
    w1 = result.x[0]
    w2 = 1 - w1
    y_pred = w1 * pred1 + w2 * pred2
    metricas = calcular_metricas(y_real, y_pred)
    
    return {'w1': w1, 'w2': w2,
            'MAE': metricas['MAE'], 'RMSE': metricas['RMSE'], 'MSE': metricas['MSE'], 'R2': metricas['R2'],
            'time': elapsed}

# Función para optimización con SLSQP usando scipy.optimize.minimize
def scipy_minimize_slsqp(y_real, pred1, pred2):

    start_time = time.time()
    bounds = [(0, 1)]
    result = minimize(lambda w: objective(w, pred1, pred2, y_real), x0=[0.5], method='SLSQP',
                     bounds=bounds)
    
    elapsed = time.time() - start_time
    w1 = result.x[0]
    w2 = 1 - w1
    y_pred = w1 * pred1 + w2 * pred2
    metricas = calcular_metricas(y_real, y_pred)
    
    return {'w1': w1, 'w2': w2,
            'MAE': metricas['MAE'], 'RMSE': metricas['RMSE'], 'MSE': metricas['MSE'], 'R2': metricas['R2'],
            'time': elapsed}

# Función para optimización con L-BFGS-B usando scipy.optimize.minimize
def scipy_minimize_lbfgsb(y_real, pred1, pred2):
    
    start_time = time.time()
    bounds = [(0, 1)]
    result = minimize(lambda w: objective(w, pred1, pred2, y_real), x0=[0.5], method='L-BFGS-B', bounds=bounds)
    
    elapsed = time.time() - start_time
    w1 = result.x[0]
    w2 = 1 - w1
    y_pred = w1 * pred1 + w2 * pred2
    metricas = calcular_metricas(y_real, y_pred)
    
    return {'w1': w1, 'w2': w2,
            'MAE': metricas['MAE'], 'RMSE': metricas['RMSE'], 'MSE': metricas['MSE'], 'R2': metricas['R2'],
            'time': elapsed}

# Función para optimización con differential evolution usando scipy.optimize.differential_evolution
def differential_evolution_opt(y_real, pred1, pred2):
    
    start_time = time.time()    
    bounds = [(0, 1)]
    result = differential_evolution(lambda w: objective(w, pred1, pred2, y_real), bounds, seed=42, maxiter=100)
    
    elapsed = time.time() - start_time
    w1 = result.x[0]
    w2 = 1 - w1
    y_pred = w1 * pred1 + w2 * pred2
    metricas = calcular_metricas(y_real, y_pred)
    
    return {'w1': w1, 'w2': w2,
            'MAE': metricas['MAE'], 'RMSE': metricas['RMSE'], 'MSE': metricas['MSE'], 'R2': metricas['R2'],
            'time': elapsed}

# Función para optimización con Ridge Regression
def ridge_regression(y_real, pred1, pred2):

    start_time = time.time()
    X = np.column_stack([pred1, pred2])
    model = Ridge(alpha=0.01, fit_intercept=False, positive=True)
    model.fit(X, y_real)
    
    weights = model.coef_
    weights = weights / weights.sum()
    
    y_pred = weights[0] * pred1 + weights[1] * pred2
    metricas = calcular_metricas(y_real, y_pred)
    
    elapsed = time.time() - start_time
    
    return {'w1': weights[0], 'w2': weights[1],
            'MAE': metricas['MAE'], 'RMSE': metricas['RMSE'], 'MSE': metricas['MSE'], 'R2': metricas['R2'],
            'time': elapsed}

# Función para optimización con búsqueda aleatoria
def random_search(y_real, pred1, pred2, n_iter=1000, metrica='MAE'):

    start_time = time.time()
    np.random.seed(42)

    w1_samples = np.random.uniform(0, 1, n_iter)
    # Shape (n_iter, n_puntos): todas las predicciones de una vez
    preds = w1_samples[:, None] * pred1[None, :] + (1 - w1_samples[:, None]) * pred2[None, :]

    if metrica == 'MAE':
        scores = np.mean(np.abs(preds - y_real[None, :]), axis=1)
    elif metrica in ('MSE', 'RMSE'):
        scores = np.mean((preds - y_real[None, :]) ** 2, axis=1)
    else:
        scores = np.array([calcular_metricas(y_real, preds[i])[metrica] for i in range(n_iter)])

    best_idx = np.argmin(scores)
    best_w1 = w1_samples[best_idx]
    best_y_pred = preds[best_idx]

    elapsed = time.time() - start_time
    metricas = calcular_metricas(y_real, best_y_pred)

    return {'w1': best_w1, 'w2': 1 - best_w1,
            'MAE': metricas['MAE'], 'RMSE': metricas['RMSE'], 'MSE': metricas['MSE'], 'R2': metricas['R2'],
            'time': elapsed}

# Función para comparar diferentes métodos de optimización para stacking
def comparar_metodos_stacking(lista_barras, resultados_stacking,
                              nombre_modelo='TFT',
                              metricas_grafico=None,
                              linestyles=['-'],
                              colormap='tab10',
                              colores_personalizados=None,
                              verbose=True):

    comparacion_metodos = {}
    
    # Definir estrategias que requieren optimización
    estrategias_opt = {
        'Prophet_TFT_Precios': {
            'nombre': f'Prophet + {nombre_modelo} Precios',
            'componente1_key': 'Prophet_Solo',
            'componente2_key': 'TFT_Precios_Solo'
        },
        'Prophet_Residuos_Opt': {
            'nombre': 'Prophet + Residuos (Opt)',
            'componente1_key': 'Prophet_Solo',
            'componente2_key': 'TFT_Residuos_Solo'
        },
        'Prophet_Residuos_Precios': {
            'nombre': f'(Prophet + Residuos) + {nombre_modelo} Precios',
            'componente1_key': 'Prophet_Residuos_Directo',
            'componente2_key': 'TFT_Precios_Solo'
        }
    }
    
    # Optimizar con distintos métodos
    for barra in lista_barras:
        
        print(f"{barra}")
        
        comparacion_metodos[barra] = {}
        
        # Estrategias que si requieren optimización
        for estrategia_key, config in estrategias_opt.items():
            
            if verbose:
                print(f"    {config['nombre']}")
            
            # Extraer componentes de TRAIN
            comp1_train = resultados_stacking[barra][config['componente1_key']]['prediccion_train']
            comp2_train = resultados_stacking[barra][config['componente2_key']]['prediccion_train']
            y_real_train = resultados_stacking[barra]['_datos_split']['y_real_train']
            
            # Extraer componentes de VAL
            comp1_val = resultados_stacking[barra][config['componente1_key']]['prediccion_val']
            comp2_val = resultados_stacking[barra][config['componente2_key']]['prediccion_val']
            y_real_val = resultados_stacking[barra]['_datos_split']['y_real_val']
            
            # Extraer componentes de TEST
            comp1_test = resultados_stacking[barra][config['componente1_key']]['prediccion_test']
            comp2_test = resultados_stacking[barra][config['componente2_key']]['prediccion_test']
            y_real_test = resultados_stacking[barra]['_datos_split']['y_real_test']
            
            comparacion_metodos[barra][estrategia_key] = {}
            
            # Métodos de optimización
            metodos = [
                ('Grid_21', lambda: grid_search_modelos(y_real_train, comp1_train, comp2_train, resolution=21)),
                ('Grid_41', lambda: grid_search_modelos(y_real_train, comp1_train, comp2_train, resolution=41)),
                ('Grid_101', lambda: grid_search_modelos(y_real_train, comp1_train, comp2_train, resolution=101)),
                ('Nelder_Mead', lambda: scipy_minimize_nelder_mead(y_real_train, comp1_train, comp2_train)),
                ('SLSQP', lambda: scipy_minimize_slsqp(y_real_train, comp1_train, comp2_train)),
                ('L_BFGS_B', lambda: scipy_minimize_lbfgsb(y_real_train, comp1_train, comp2_train)),
                ('Diff_Evolution', lambda: differential_evolution_opt(y_real_train, comp1_train, comp2_train)),
                ('Ridge', lambda: ridge_regression(y_real_train, comp1_train, comp2_train)),
                ('Random_1000', lambda: random_search(y_real_train, comp1_train, comp2_train, n_iter=1000))
            ]
            
            for nombre_metodo, funcion in metodos:
                
                # Optimizar en TRAIN
                result = funcion()
                w1, w2 = result['w1'], result['w2']
                tiempo = result['time']
                
                # Aplicar pesos a VAL
                pred_val = w1 * comp1_val + w2 * comp2_val
                metricas_val = calcular_metricas(y_real_val, pred_val)
                
                # Aplicar pesos a TEST
                pred_test = w1 * comp1_test + w2 * comp2_test
                metricas_test = calcular_metricas(y_real_test, pred_test)
                
                # Guardar resultados
                comparacion_metodos[barra][estrategia_key][nombre_metodo] = {
                    'w1': w1,
                    'w2': w2,
                    'time': tiempo,
                    'train': {
                        'MAE': result['MAE'],
                        'RMSE': result['RMSE'],
                        'MSE': result['MSE'],
                        'R2': result['R2']
                    },
                    'val': {
                        'MAE': metricas_val['MAE'],
                        'RMSE': metricas_val['RMSE'],
                        'MSE': metricas_val['MSE'],
                        'R2': metricas_val['R2']
                    },
                    'test': {
                        'MAE': metricas_test['MAE'],
                        'RMSE': metricas_test['RMSE'],
                        'MSE': metricas_test['MSE'],
                        'R2': metricas_test['R2']
                    }
                }
     
    # Crear dataframe de resultados
    datos_tabla = []
    
    for barra in lista_barras:
        for estrategia_key, metodos_dict in comparacion_metodos[barra].items():
            estrategia_nombre = estrategias_opt[estrategia_key]['nombre']
            
            for metodo, resultado in metodos_dict.items():
                
                # Fila TRAIN
                datos_tabla.append({
                    'Barra': barra,
                    'Estrategia': estrategia_nombre,
                    'Método': metodo,
                    'Set': 'Train',
                    'w1': resultado['w1'],
                    'w2': resultado['w2'],
                    'MAE': resultado['train']['MAE'],
                    'RMSE': resultado['train']['RMSE'],
                    'MSE': resultado['train']['MSE'],
                    'R²': resultado['train']['R2'],
                    'Tiempo (s)': resultado['time']
                })
                
                # Fila VAL
                datos_tabla.append({
                    'Barra': barra,
                    'Estrategia': estrategia_nombre,
                    'Método': metodo,
                    'Set': 'Val',
                    'w1': resultado['w1'],
                    'w2': resultado['w2'],
                    'MAE': resultado['val']['MAE'],
                    'RMSE': resultado['val']['RMSE'],
                    'MSE': resultado['val']['MSE'],
                    'R²': resultado['val']['R2'],
                    'Tiempo (s)': resultado['time']
                })
                
                # Fila TEST
                datos_tabla.append({
                    'Barra': barra,
                    'Estrategia': estrategia_nombre,
                    'Método': metodo,
                    'Set': 'Test',
                    'w1': resultado['w1'],
                    'w2': resultado['w2'],
                    'MAE': resultado['test']['MAE'],
                    'RMSE': resultado['test']['RMSE'],
                    'MSE': resultado['test']['MSE'],
                    'R²': resultado['test']['R2'],
                    'Tiempo (s)': resultado['time']
                })
    
    df_metodos = pd.DataFrame(datos_tabla)
    
    # Formatear
    df_metodos['w1'] = df_metodos['w1'].round(4)
    df_metodos['w2'] = df_metodos['w2'].round(4)
    df_metodos['MAE'] = df_metodos['MAE'].round(4)
    df_metodos['RMSE'] = df_metodos['RMSE'].round(4)
    df_metodos['MSE'] = df_metodos['MSE'].round(4)
    df_metodos['R²'] = df_metodos['R²'].round(4)
    df_metodos['Tiempo (s)'] = df_metodos['Tiempo (s)'].round(4)
    
    # Visualización    
    grafico_lineas(df_metodos,
                   modo='metodos',
                   metricas=metricas_grafico,
                   nombre_modelo=nombre_modelo,
                   conjunto='Test',
                   linestyles=linestyles,
                   colormap=colormap,
                   colores_personalizados=colores_personalizados)
    
    # Mejor método por estartegia y barra
    print("\n" + "="*100)
    print(f"MEJORES MÉTODOS POR BARRA Y ESTRATEGIA (TEST) - {nombre_modelo}")
    print("="*100 + "\n")
    
    datos_mejores = []
    
    for barra in lista_barras:
        for estrategia_key, metodos_dict in comparacion_metodos[barra].items():
            estrategia_nombre = estrategias_opt[estrategia_key]['nombre']
            
            # Encontrar mejor método para esta estrategia
            mejor_mae = np.inf
            mejor_metodo = None
            
            for metodo, resultado in metodos_dict.items():
                if resultado['test']['MAE'] < mejor_mae:
                    mejor_mae = resultado['test']['MAE']
                    mejor_metodo = metodo
                    mejor_resultado = resultado
            
            # Formatear pesos
            pesos_str = f"w1={mejor_resultado['w1']:.3f}, w2={mejor_resultado['w2']:.3f}"
            
            datos_mejores.append({
                'Barra': barra,
                'Estrategia': estrategia_nombre,
                'Mejor Método': mejor_metodo,
                'Pesos': pesos_str,
                'MAE': mejor_resultado['test']['MAE'],
                'RMSE': mejor_resultado['test']['RMSE'],
                'R²': mejor_resultado['test']['R2'],
                'Tiempo (s)': mejor_resultado['time']
            })
    
    # Crear DataFrame de mejores
    df_mejores = pd.DataFrame(datos_mejores)
    df_mejores['MAE'] = df_mejores['MAE'].round(4)
    df_mejores['RMSE'] = df_mejores['RMSE'].round(4)
    df_mejores['R²'] = df_mejores['R²'].round(4)
    df_mejores['Tiempo (s)'] = df_mejores['Tiempo (s)'].round(4)
    
    # Ocultar barras duplicadas
    df_mejores_display = df_mejores.copy()
    df_mejores_display['Barra'] = df_mejores_display['Barra'].mask(df_mejores_display['Barra'].duplicated(), '')

    # Mostrar tabla
    display(df_mejores_display)
    
    return comparacion_metodos, df_metodos

# ======================== GRAFICAR ==============================
# Función para graficar resultados de stacking
def graficar_stacking(resultados_stacking, lista_barras,
                      nombre_modelo='TFT',
                      estrategias_plot=None,
                      metrica_mejor='MAE',
                      conjunto_mejor='test',
                      zoom_dias=30,
                      colores_estrategias=None,
                      guardar_graficos=False,
                      mostrar_graficos=True,
                      verbose=True):

    N_INSET = 14 * 24  # 2 semanas en datos horarios

    if colores_estrategias is None:
        colores_disponibles = {
            'Prophet_Solo': '#2CA02C',
            'TFT_Precios_Solo': '#0EE828',
            'Prophet_TFT_Precios': '#121CD3',
            'Prophet_Residuos_Directo': '#9467BD',
            'Prophet_Residuos_Opt': '#D62728',
            'Prophet_Residuos_Precios': '#FF7F0E'
        }
    else:
        colores_disponibles = colores_estrategias

    for barra in lista_barras:

        # Identificar la mejor estrategia para cada barra
        mejor_valor = np.inf if metrica_mejor in ['MAE', 'RMSE', 'MSE'] else -np.inf
        mejor_key = None

        for key, resultado in resultados_stacking[barra].items():
            if key.startswith('_') or key == 'TFT_Residuos_Solo':
                continue
            valor = resultado[f'metricas_{conjunto_mejor}'][metrica_mejor]
            if metrica_mejor in ['MAE', 'RMSE', 'MSE']:
                if valor < mejor_valor:
                    mejor_valor = valor
                    mejor_key = key
            else:
                if valor > mejor_valor:
                    mejor_valor = valor
                    mejor_key = key

        estrategias_a_graficar = [mejor_key]
        if estrategias_plot is not None:
            for est in estrategias_plot:
                if est not in estrategias_a_graficar and est in resultados_stacking[barra]:
                    estrategias_a_graficar.append(est)

        if verbose:
            mejor_nombre = resultados_stacking[barra][mejor_key]['nombre']
            print(f'\n{barra}')
            print(f'   Mejor: {mejor_nombre} ({metrica_mejor}={mejor_valor:.4f} en {conjunto_mejor})')
            if estrategias_plot:
                print(f'   Adicionales: {len(estrategias_a_graficar) - 1}')

        colores = {k: v for k, v in colores_disponibles.items() if k in estrategias_a_graficar}
        if mejor_key not in colores:
            colores[mejor_key] = '#2CA02C'

        fig, axes = plt.subplots(4, 1, figsize=(18, 26))

        datos_split  = resultados_stacking[barra]['_datos_split']
        fechas_train = datos_split['fechas_train']
        fechas_val   = datos_split['fechas_val']
        fechas_test  = datos_split['fechas_test']
        y_real_train = datos_split['y_real_train']
        y_real_val   = datos_split['y_real_val']
        y_real_test  = datos_split['y_real_test']

        # ── SUBPLOT 1: TRAIN ───────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(fechas_train, y_real_train,
                label='Real', color='black', linewidth=2.5, alpha=0.9, zorder=5)
        for estrategia in estrategias_a_graficar:
            if estrategia in resultados_stacking[barra]:
                pred   = resultados_stacking[barra][estrategia]['prediccion_train']
                nombre = resultados_stacking[barra][estrategia]['nombre']
                mae    = resultados_stacking[barra][estrategia]['metricas_train']['MAE']
                lw = 2.5 if estrategia == mejor_key else 1.8
                al = 0.9 if estrategia == mejor_key else 0.7
                ax.plot(fechas_train, pred, label=f'{nombre} (MAE={mae:.2f})',
                        color=colores[estrategia], linewidth=lw, alpha=al)
        ax.set_title(f'{nombre_modelo} - {barra} - SET DE ENTRENAMIENTO\n{len(fechas_train)} puntos',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Fecha', fontsize=11)
        ax.set_ylabel('Precio (USD/MWh)', fontsize=11)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        # Inset: últimas 2 semanas
        n_in  = min(N_INSET, len(fechas_train))
        ax_in = ax.inset_axes([0.62, 0.55, 0.36, 0.38])
        ax_in.plot(fechas_train[-n_in:], y_real_train[-n_in:], color='black', linewidth=1.5, alpha=0.9)
        for estrategia in estrategias_a_graficar:
            if estrategia in resultados_stacking[barra]:
                pred = resultados_stacking[barra][estrategia]['prediccion_train']
                ax_in.plot(fechas_train[-n_in:], pred[-n_in:], color=colores[estrategia], linewidth=1.2, alpha=0.85)
        ax_in.set_title('Últimas 2 sem.', fontsize=8, fontweight='bold')
        ax_in.tick_params(labelsize=7)
        ax_in.tick_params(axis='x', rotation=30)
        ax_in.grid(True, alpha=0.3, linewidth=0.5)
        ax.indicate_inset_zoom(ax_in, edgecolor='0.4', linewidth=1.2)

        # ── SUBPLOT 2: VAL ─────────────────────────────────────────────────────
        ax = axes[1]
        ax.plot(fechas_val, y_real_val,
                label='Real', color='black', linewidth=2.5, alpha=0.9, zorder=5)
        for estrategia in estrategias_a_graficar:
            if estrategia in resultados_stacking[barra]:
                pred   = resultados_stacking[barra][estrategia]['prediccion_val']
                nombre = resultados_stacking[barra][estrategia]['nombre']
                mae    = resultados_stacking[barra][estrategia]['metricas_val']['MAE']
                r2     = resultados_stacking[barra][estrategia]['metricas_val']['R2']
                lw = 2.5 if estrategia == mejor_key else 1.8
                al = 0.9 if estrategia == mejor_key else 0.7
                ax.plot(fechas_val, pred, label=f'{nombre} (MAE={mae:.2f}, R\u00b2={r2:.3f})',
                        color=colores[estrategia], linewidth=lw, alpha=al)
        ax.set_title(f'{nombre_modelo} - {barra} - SET DE VALIDACI\u00d3N\n{len(fechas_val)} puntos',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Fecha', fontsize=11)
        ax.set_ylabel('Precio (USD/MWh)', fontsize=11)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        # Inset: últimas 2 semanas
        n_in  = min(N_INSET, len(fechas_val))
        ax_in = ax.inset_axes([0.62, 0.55, 0.36, 0.38])
        ax_in.plot(fechas_val[-n_in:], y_real_val[-n_in:], color='black', linewidth=1.5, alpha=0.9)
        for estrategia in estrategias_a_graficar:
            if estrategia in resultados_stacking[barra]:
                pred = resultados_stacking[barra][estrategia]['prediccion_val']
                ax_in.plot(fechas_val[-n_in:], pred[-n_in:], color=colores[estrategia], linewidth=1.2, alpha=0.85)
        ax_in.set_title('\u00daltimas 2 sem.', fontsize=8, fontweight='bold')
        ax_in.tick_params(labelsize=7)
        ax_in.tick_params(axis='x', rotation=30)
        ax_in.grid(True, alpha=0.3, linewidth=0.5)
        ax.indicate_inset_zoom(ax_in, edgecolor='0.4', linewidth=1.2)

        # ── SUBPLOT 3: TEST ────────────────────────────────────────────────────
        ax = axes[2]
        ax.plot(fechas_test, y_real_test,
                label='Real', color='black', linewidth=2.5, alpha=0.9, zorder=5)
        for estrategia in estrategias_a_graficar:
            if estrategia in resultados_stacking[barra]:
                pred   = resultados_stacking[barra][estrategia]['prediccion_test']
                nombre = resultados_stacking[barra][estrategia]['nombre']
                mae    = resultados_stacking[barra][estrategia]['metricas_test']['MAE']
                r2     = resultados_stacking[barra][estrategia]['metricas_test']['R2']
                lw = 2.5 if estrategia == mejor_key else 1.8
                al = 0.9 if estrategia == mejor_key else 0.7
                ax.plot(fechas_test, pred, label=f'{nombre} (MAE={mae:.2f}, R\u00b2={r2:.3f})',
                        color=colores[estrategia], linewidth=lw, alpha=al)
        ax.set_title(f'{nombre_modelo} - {barra} - SET DE TEST\n{len(fechas_test)} puntos',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Fecha', fontsize=11)
        ax.set_ylabel('Precio (USD/MWh)', fontsize=11)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        # Inset: últimas 2 semanas
        n_in  = min(N_INSET, len(fechas_test))
        ax_in = ax.inset_axes([0.62, 0.55, 0.36, 0.38])
        ax_in.plot(fechas_test[-n_in:], y_real_test[-n_in:], color='black', linewidth=1.5, alpha=0.9)
        for estrategia in estrategias_a_graficar:
            if estrategia in resultados_stacking[barra]:
                pred = resultados_stacking[barra][estrategia]['prediccion_test']
                ax_in.plot(fechas_test[-n_in:], pred[-n_in:], color=colores[estrategia], linewidth=1.2, alpha=0.85)
        ax_in.set_title('\u00daltimas 2 sem.', fontsize=8, fontweight='bold')
        ax_in.tick_params(labelsize=7)
        ax_in.tick_params(axis='x', rotation=30)
        ax_in.grid(True, alpha=0.3, linewidth=0.5)
        ax.indicate_inset_zoom(ax_in, edgecolor='0.4', linewidth=1.2)

        # ── SUBPLOT 4: ZOOM ÚLTIMOS N DÍAS DEL TEST ────────────────────────────
        ax = axes[3]
        n_zoom      = min(zoom_dias * 24, len(fechas_test))
        fechas_zoom = fechas_test[-n_zoom:]
        y_real_zoom = y_real_test[-n_zoom:]

        ax.plot(fechas_zoom, y_real_zoom,
                label='Real', color='black', linewidth=2.5, alpha=0.9, zorder=5)
        for estrategia in estrategias_a_graficar:
            if estrategia in resultados_stacking[barra]:
                pred_test = resultados_stacking[barra][estrategia]['prediccion_test']
                pred_zoom = pred_test[-n_zoom:]
                nombre    = resultados_stacking[barra][estrategia]['nombre']
                mae_zoom  = np.mean(np.abs(y_real_zoom - pred_zoom))
                r2_zoom   = resultados_stacking[barra][estrategia]['metricas_test']['R2']
                lw = 2.5 if estrategia == mejor_key else 1.8
                al = 0.9 if estrategia == mejor_key else 0.7
                ax.plot(fechas_zoom, pred_zoom,
                        label=f'{nombre} (MAE={mae_zoom:.2f}, R\u00b2={r2_zoom:.3f})',
                        color=colores[estrategia], linewidth=lw, alpha=al)

        fecha_ini = pd.Timestamp(fechas_zoom[0]).strftime('%Y-%m-%d')
        fecha_fin = pd.Timestamp(fechas_zoom[-1]).strftime('%Y-%m-%d')
        ax.set_title(
            f'{nombre_modelo} - {barra} - ZOOM \u00daltimos {zoom_dias} D\u00cdAS (TEST)\n'
            f'{fecha_ini} \u2192 {fecha_fin}',
            fontsize=14, fontweight='bold', pad=15
        )
        ax.set_xlabel('Fecha', fontsize=11)
        ax.set_ylabel('Precio (USD/MWh)', fontsize=11)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if guardar_graficos:
            filename = f'series_temporales_{nombre_modelo}_{barra}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f'   Guardado: {filename}')

        if mostrar_graficos:
            plt.show()
        else:
            plt.close()

    # ── Gráfico de barras para comparación de MAE ──────────────────────────────
    n_barras = len(lista_barras)
    n_cols = 2
    n_rows = (n_barras + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7 * n_rows))

    if n_barras == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    axes_flat = axes.flatten()

    for idx, barra in enumerate(lista_barras):
        ax = axes_flat[idx]

        estrategias = ['Prophet_Solo',
                       'TFT_Precios_Solo',
                       'Prophet_TFT_Precios',
                       'Prophet_Residuos_Directo',
                       'Prophet_Residuos_Opt',
                       'Prophet_Residuos_Precios']

        nombres, mae_train_list, mae_val_list, mae_test_list = [], [], [], []

        for estrategia in estrategias:
            if estrategia in resultados_stacking[barra]:
                nombres.append(resultados_stacking[barra][estrategia]['nombre'])
                mae_train_list.append(resultados_stacking[barra][estrategia]['metricas_train']['MAE'])
                mae_val_list.append(resultados_stacking[barra][estrategia]['metricas_val']['MAE'])
                mae_test_list.append(resultados_stacking[barra][estrategia]['metricas_test']['MAE'])

        x = np.arange(len(nombres))
        width = 0.25

        bars1 = ax.bar(x - width, mae_train_list, width, label='Train',
                       color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x, mae_val_list, width, label='Val',
                       color='coral', alpha=0.8, edgecolor='black', linewidth=1.2)
        bars3 = ax.bar(x + width, mae_test_list, width, label='Test',
                       color='green', alpha=0.8, edgecolor='black', linewidth=1.2)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        idx_mejor = np.argmin(mae_test_list)
        ax.axhline(mae_test_list[idx_mejor], color='darkgreen', linestyle='--',
                   linewidth=2, alpha=0.6, label=f'Mejor Test: {mae_test_list[idx_mejor]:.2f}')

        ax.set_ylabel('MAE (USD/MWh)', fontsize=12, fontweight='bold')
        ax.set_title(f'{nombre_modelo} - {barra}\nComparaci\u00f3n de MAE por Estrategia',
                     fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(nombres, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

    for idx in range(n_barras, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()

    if guardar_graficos:
        filename = f'comparacion_mae_barras_{nombre_modelo}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f'Guardado: {filename}\n')

    if mostrar_graficos:
        plt.show()
    else:
        plt.close()

    # ── Tabla resumen completa ─────────────────────────────────────────────────
    datos_resumen = []

    for barra in lista_barras:
        estrategias_orden = [
            'Prophet_Solo', 'TFT_Precios_Solo', 'Prophet_TFT_Precios',
            'Prophet_Residuos_Directo', 'Prophet_Residuos_Opt', 'Prophet_Residuos_Precios'
        ]
        for estrategia in estrategias_orden:
            if estrategia in resultados_stacking[barra]:
                resultado = resultados_stacking[barra][estrategia]
                if resultado['pesos'] is None:
                    pesos_str = '-'
                else:
                    pesos_items = []
                    for key, val in resultado['pesos'].items():
                        key_short = (key.replace('w_prophet_residuos', 'w_PR')
                                       .replace('w_tft_precios', 'w_TP')
                                       .replace('w_prophet', 'w_P')
                                       .replace('w_tft', 'w_T')
                                       .replace('w_residuo', 'w_R'))
                        pesos_items.append(f'{key_short}={val:.2f}')
                    pesos_str = ', '.join(pesos_items)
                for set_nombre, metricas in [('Train', resultado['metricas_train']),
                                              ('Val',   resultado['metricas_val']),
                                              ('Test',  resultado['metricas_test'])]:
                    datos_resumen.append({
                        'Barra': barra, 'Estrategia': resultado['nombre'],
                        'Pesos': pesos_str, 'Set': set_nombre,
                        'MAE': metricas['MAE'], 'RMSE': metricas['RMSE'],
                        'MSE': metricas['MSE'], 'R\u00b2': metricas['R2']
                    })

    df_resumen = pd.DataFrame(datos_resumen)
    for col in ['MAE', 'RMSE', 'MSE']:
        df_resumen[col] = df_resumen[col].round(2)
    df_resumen['R\u00b2'] = df_resumen['R\u00b2'].round(3)

    df_resumen_display = df_resumen.copy()
    df_resumen_display['Barra'] = df_resumen_display['Barra'].mask(
        df_resumen_display['Barra'].duplicated(), ''
    )

    if guardar_graficos:
        filename = f'tabla_resumen_stacking_{nombre_modelo}.csv'
        df_resumen.to_csv(filename, index=False)
        if verbose:
            print(f'\nTabla guardada: {filename}\n')

    # ── Tabla de mejores estrategias ───────────────────────────────────────────
    mejores = []
    for barra in lista_barras:
        mejor_mae = np.inf
        mejor_resultado = None
        for key, resultado in resultados_stacking[barra].items():
            if key.startswith('_') or key == 'TFT_Residuos_Solo':
                continue
            mae_test = resultado['metricas_test']['MAE']
            if mae_test < mejor_mae:
                mejor_mae = mae_test
                mejor_resultado = resultado

        pesos_str = ('No aplica' if mejor_resultado['pesos'] is None
                     else ', '.join([f'{k}={v:.3f}' for k, v in mejor_resultado['pesos'].items()]))

        mae_prophet_test = resultados_stacking[barra]['Prophet_Solo']['metricas_test']['MAE']
        mae_tft_test     = resultados_stacking[barra]['TFT_Precios_Solo']['metricas_test']['MAE']
        mejora_prophet = ((mae_prophet_test - mejor_mae) / mae_prophet_test * 100) if mae_prophet_test > 0 else 0
        mejora_tft     = ((mae_tft_test - mejor_mae) / mae_tft_test * 100) if mae_tft_test > 0 else 0

        mejores.append({
            'Barra': barra,
            'Mejor Estrategia': mejor_resultado['nombre'],
            'Pesos': pesos_str,
            'MAE (Test)': mejor_resultado['metricas_test']['MAE'],
            'RMSE (Test)': mejor_resultado['metricas_test']['RMSE'],
            'MSE (Test)': mejor_resultado['metricas_test']['MSE'],
            'R\u00b2 (Test)': mejor_resultado['metricas_test']['R2'],
            'Mejora vs Prophet (%)': mejora_prophet,
            'Mejora vs TFT (%)': mejora_tft
        })

    df_mejores = pd.DataFrame(mejores)
    df_mejores['MAE (Test)']            = df_mejores['MAE (Test)'].round(2)
    df_mejores['RMSE (Test)']           = df_mejores['RMSE (Test)'].round(2)
    df_mejores['MSE (Test)']            = df_mejores['MSE (Test)'].round(2)
    df_mejores['R\u00b2 (Test)']             = df_mejores['R\u00b2 (Test)'].round(3)
    df_mejores['Mejora vs Prophet (%)'] = df_mejores['Mejora vs Prophet (%)'].round(2)
    df_mejores['Mejora vs TFT (%)']     = df_mejores['Mejora vs TFT (%)'].round(2)
    display(df_mejores)

    if guardar_graficos:
        filename = f'mejores_estrategias_{nombre_modelo}.csv'
        df_mejores.to_csv(filename, index=False)
        print(f'\nTabla guardada: {filename}\n')

    return {'df_resumen': df_resumen_display, 'df_mejores': df_mejores}







# ======================== FUNCIONES DE VISUALIZACIÓN ADICIONALES ==============================

def grafico_metricas_test(df_resultados,
                          nombre_modelo='TFT',
                          metricas=None,
                          colormap='cividis',
                          colores_personalizados=None,
                          linestyles=None,
                          figsize=None,
                          guardar=False,
                          nombre_archivo=None):
    """
    Genera una figura con las métricas del conjunto Test únicamente.

    Parámetros
    ----------
    df_resultados : pd.DataFrame
        DataFrame con columnas ['Barra', 'Estrategia', 'Set', 'MAE', 'RMSE', 'R²'].
        Equivalente al df_comparacion devuelto por stacking_optimization.
    nombre_modelo : str
        Nombre del modelo (e.g. 'TFT_LN', 'TFT_DyT'). Aparece en el título.
    metricas : list[str] | None
        Lista de métricas a graficar. Por defecto ['MAE', 'RMSE', 'R²'].
    colormap : str
        Colormap de matplotlib para asignar colores a las estrategias.
    colores_personalizados : dict | list | None
        Colores explícitos por estrategia. Si es dict, debe mapear nombre → color.
        Si es list, se asignan en orden. Sobreescribe colormap.
    linestyles : list[str] | None
        Estilos de línea a rotar entre estrategias. Por defecto ['-', '--', '-.', ':'].
    figsize : tuple | None
        Tamaño de la figura. Por defecto (7 * n_metricas, 5).
    guardar : bool
        Si True, guarda la figura en PNG con dpi=300.
    nombre_archivo : str | None
        Nombre del archivo PNG. Si None, se genera automáticamente.
    """

    if metricas is None:
        metricas = ['MAE', 'RMSE', 'R²']
    elif isinstance(metricas, str):
        metricas = [metricas]

    # Filtrar solo Test
    df_test = df_resultados[df_resultados['Set'] == 'Test'].copy()

    barras      = sorted(df_test['Barra'].unique())
    estrategias = sorted(df_test['Estrategia'].unique())
    n_barras     = len(barras)
    n_estrategias = len(estrategias)
    n_metricas   = len(metricas)

    # Colores
    if colores_personalizados is not None:
        if isinstance(colores_personalizados, dict):
            color_map = colores_personalizados
        else:
            color_map = {e: colores_personalizados[i % len(colores_personalizados)]
                         for i, e in enumerate(estrategias)}
    else:
        try:
            cmap   = plt.cm.get_cmap(colormap)
            colors = cmap(np.linspace(0, 1, n_estrategias))
        except Exception:
            colors = plt.cm.tab10(np.linspace(0, 1, n_estrategias))
        color_map = {e: colors[i] for i, e in enumerate(estrategias)}

    # Estilos de línea
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':', '-', '--']
    ls_map = {e: linestyles[i % len(linestyles)] for i, e in enumerate(estrategias)}

    # Abreviaciones de nombres (reutiliza la lógica de grafico_lineas)
    def _abreviar(nombre):
        mapeo = {
            'Prophet Solo': 'Prophet',
            f'{nombre_modelo} Precios Solo': f'{nombre_modelo}_P',
            f'{nombre_modelo} Residuos Solo': f'{nombre_modelo}_R',
            f'Prophet + {nombre_modelo} Precios': f'P+{nombre_modelo}_P',
            'Prophet + Residuos (Directo)': 'P+R(D)',
            'Prophet + Residuos (Opt)': 'P+R(O)',
            f'(Prophet + Residuos) + {nombre_modelo} Precios': f'(P+R)+{nombre_modelo}_P',
        }
        if nombre in mapeo:
            return mapeo[nombre]
        for clave, valor in mapeo.items():
            if clave in nombre:
                return valor
        return nombre.replace('Prophet', 'P').replace('Residuos', 'R').replace('Precios', 'P')[:50]

    if figsize is None:
        figsize = (7 * n_metricas, 5)

    fig, axes = plt.subplots(1, n_metricas, figsize=figsize)
    if n_metricas == 1:
        axes = [axes]

    plt.subplots_adjust(wspace=0.30)

    for i_met, metrica in enumerate(metricas):
        ax = axes[i_met]

        for estrategia in estrategias:
            df_e = df_test[df_test['Estrategia'] == estrategia]
            valores = []
            for barra in barras:
                fila = df_e[df_e['Barra'] == barra]
                valores.append(fila[metrica].values[0] if len(fila) > 0 else np.nan)

            ax.plot(range(n_barras), valores,
                    marker='o', markersize=7, linewidth=2.5,
                    label=_abreviar(estrategia),
                    color=color_map[estrategia],
                    linestyle=ls_map[estrategia],
                    alpha=0.85)

        ax.set_title(metrica, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(range(n_barras))
        ax.set_xticklabels(barras, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('Barra', fontsize=10)
        ax.set_ylabel(metrica, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        ax.set_axisbelow(True)

        # Leyenda en el último subplot
        if i_met == n_metricas - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=9, framealpha=0.95,
                      title='Estrategias', title_fontsize=10)

    plt.suptitle(f'Métricas en Test por Barra\n{nombre_modelo}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if guardar:
        fname = nombre_archivo or f'metricas_test_{nombre_modelo}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'Guardado: {fname}')

    plt.show()


def grafico_zoom_ultimos_dias(resultados_stacking,
                              lista_barras=None,
                              nombre_modelo='TFT',
                              n_dias=30,
                              horas_por_dia=24,
                              color_real='black',
                              color_mejor='tab:blue',
                              figsize=(14, 4),
                              guardar=False,
                              mostrar=True):
    """
    Genera una figura por barra con zoom en los últimos n_dias del conjunto Test,
    superponiendo la mejor estrategia (menor MAE en Test) sobre los valores reales.

    Parámetros
    ----------
    resultados_stacking : dict
        Diccionario devuelto por stacking_optimization. Cada barra debe contener
        la clave '_datos_split' con 'y_real_test' y 'fechas_test', y al menos
        una estrategia con 'prediccion_test' y 'metricas_test'.
    lista_barras : list[str] | None
        Barras a graficar. Si None, se usan todas las claves del diccionario.
    nombre_modelo : str
        Nombre del modelo para los títulos y nombres de archivo.
    n_dias : int
        Número de días finales del Test a mostrar. Por defecto 30.
    horas_por_dia : int
        Resolución temporal en puntos por día. Por defecto 24 (datos horarios).
    color_real : str
        Color de la serie de valores reales.
    color_mejor : str
        Color de la serie de la mejor estrategia.
    figsize : tuple
        Tamaño de cada figura individual.
    guardar : bool
        Si True, guarda cada figura como PNG con dpi=300.
    mostrar : bool
        Si True, muestra cada figura con plt.show().
    """

    if lista_barras is None:
        lista_barras = [b for b in resultados_stacking.keys() if not b.startswith('_')]

    n_puntos_zoom = n_dias * horas_por_dia

    for barra in lista_barras:

        datos_barra = resultados_stacking[barra]

        # ── Valores reales y fechas del Test ──────────────────────────────────
        split       = datos_barra['_datos_split']
        y_real_test = split['y_real_test']
        fechas_test = split['fechas_test']

        # ── Identificar la mejor estrategia (menor MAE en Test) ───────────────
        mejor_key   = None
        mejor_mae   = np.inf

        for key, resultado in datos_barra.items():
            if key.startswith('_'):
                continue
            mae_test = resultado['metricas_test']['MAE']
            if mae_test < mejor_mae:
                mejor_mae   = mae_test
                mejor_key   = key
                mejor_nombre = resultado['nombre']
                mejor_pred  = resultado['prediccion_test']

        if mejor_key is None:
            print(f'[ADVERTENCIA] No se encontraron estrategias para la barra {barra}. Se omite.')
            continue

        # ── Zoom: últimos n_puntos_zoom puntos ────────────────────────────────
        n_total  = min(len(y_real_test), len(mejor_pred), len(fechas_test))
        n_zoom   = min(n_puntos_zoom, n_total)
        inicio   = n_total - n_zoom

        fechas_zoom = fechas_test[inicio:]
        y_real_zoom = y_real_test[inicio:]
        y_pred_zoom = mejor_pred[inicio:]

        # ── Métricas en la ventana de zoom ────────────────────────────────────
        mae_zoom  = mean_absolute_error(y_real_zoom, y_pred_zoom)
        rmse_zoom = np.sqrt(mean_squared_error(y_real_zoom, y_pred_zoom))
        r2_zoom   = r2_score(y_real_zoom, y_pred_zoom)

        # ── Figura ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(fechas_zoom, y_real_zoom,
                color=color_real, linewidth=1.5,
                label='Real', zorder=3)
        ax.plot(fechas_zoom, y_pred_zoom,
                color=color_mejor, linewidth=1.8, linestyle='--',
                label=f'Mejor: {mejor_nombre}', zorder=4)

        # Anotación de métricas
        metricas_txt = (f'MAE = {mae_zoom:.2f}  |  '
                        f'RMSE = {rmse_zoom:.2f}  |  '
                        f'R² = {r2_zoom:.3f}')
        ax.text(0.01, 0.97, metricas_txt,
                transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

        ax.set_title(
            f'{nombre_modelo} — {barra}\n'
            f'Últimos {n_dias} días del conjunto Test',
            fontsize=13, fontweight='bold', pad=12
        )
        ax.set_xlabel('Fecha', fontsize=11)
        ax.set_ylabel('Costo Marginal (USD/MWh)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if guardar:
            fname = f'zoom_test_{nombre_modelo}_{barra}.png'
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            print(f'Guardado: {fname}')

        if mostrar:
            plt.show()
        else:
            plt.close()

#grafico_metricas_test(df_stacking_ln, nombre_modelo='TFT_LN')
#grafico_zoom_ultimos_dias(resultados_stacking_ln, nombre_modelo='TFT_LN', n_dias=30)