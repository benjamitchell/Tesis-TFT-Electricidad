import io
import numpy as np
import pandas as pd
import holidays, sys
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from IPython.display import display

# Función de preprocesamiento para Prophet
def preprocesamiento_prophet(df_por_barra, BARRA, freq, año_inicio, año_fin, proporciones, umbral_outliers, graficos=None, df_feriados=None):

    df_prophet = df_por_barra[BARRA].copy()
    print(f'Valor máximo: {df_prophet["Valor"].max()}, valor mínimo: {df_prophet["Valor"].min()}')

    # Vemos los datos faltantes
    columna_fecha = 'Fecha'
    analizar_completitud(df_prophet, columna_fecha, freq, año_inicio, año_fin)
    df_prophet = completar_fechas(df_prophet, freq, columna_fecha)
    analizar_completitud(df_prophet, columna_fecha, freq, año_inicio, año_fin)

    # Generamos los conjuntos de train, val y test
    df_train_prophet, df_val_prophet, df_test_prophet = dividir_serie_temporal(df_prophet, proporciones)

    # Preprocesamos los datos del conjunto train
    df_train_prophet_clean, train_stats = preprocesar_serie(df=df_train_prophet, col_fecha='Fecha', col_valor='Valor',
                                                            freq = freq, rango_años=(año_inicio, año_fin), 
                                                            zscore_stats=None, umbral_outliers=umbral_outliers)

    print(f"\nSeries limpias para {BARRA}: {len(df_train_prophet_clean)} registros.")

    # Preprocesamos los datos del conjunto val
    df_val_prophet_clean, _ = preprocesar_serie(df=df_val_prophet, col_fecha='Fecha', col_valor='Valor',
                                                freq = freq, rango_años=(año_inicio, año_fin), 
                                                zscore_stats=train_stats)

    print(f"\nSeries limpias para {BARRA}: {len(df_val_prophet_clean)} registros.")

    # Preprocesamos los datos del conjunto test
    df_test_prophet_clean, _ = preprocesar_serie(df=df_test_prophet, col_fecha='Fecha', col_valor='Valor',
                                                 freq = freq, rango_años=(año_inicio, año_fin),
                                                 zscore_stats=train_stats)

    print(f"\nSeries limpias para {BARRA}: {len(df_test_prophet_clean)} registros.")

    # Calculamos feriados solo si no fueron provistos externamente
    if df_feriados is None:
        años = list(range(año_inicio, año_fin + 1))
        ch_holidays = holidays.Chile(years=años)
        df_feriados = pd.DataFrame(ch_holidays.items(), columns=['ds', 'holiday'])
        df_feriados['ds'] = pd.to_datetime(df_feriados['ds'])
        df_feriados = df_feriados.sort_values(by='ds', ascending=True).reset_index(drop=True)

    if graficos:
        # Graficamos
        plt.figure(figsize=(12, 5))

        # Histograma del conjunto de entrenamiento
        sns.kdeplot(df_train_prophet_clean['y'], label='Train', fill=True)

        # Histograma del conjunto de validación (distribución real con el scaler del train)
        sns.kdeplot(df_val_prophet_clean['y'], label='Validation', fill=True)

        # Histograma del conjunto de prueba (distribución real con el scaler del train)
        sns.kdeplot(df_test_prophet_clean['y'], label='Test', fill=True)
        plt.title(f'Distribución de Valores por conjunto de datos ({BARRA})')
        plt.legend()
        plt.show()

    return df_train_prophet_clean, df_val_prophet_clean, df_test_prophet_clean, df_feriados

# Context manager para capturar prints
@contextmanager
def solo_prints_clave():
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        yield
    finally:
        output = captured_output.getvalue()
        sys.stdout = old_stdout
        
        # Palabras clave que quieres mantener
        keywords = [
            'Valor máximo',
            'Valor mínimo', 
            'Rango de fechas',
            'Estadísticas calculadas'
        ]
        
        # Filtrar líneas
        for line in output.split('\n'):
            if any(keyword in line for keyword in keywords):
                print(line)

# Función para procesar todas las localidades y graficar resultados
def procesamiento_por_barras(df_por_barra, barras, freq, año_inicio, año_fin, proporciones, umbral_outliers, graficos=None):

    n_barras = len(barras)
    resultados = {}

    # Calculamos feriados una sola vez para todas las barras
    años = list(range(año_inicio, año_fin + 1))
    ch_holidays = holidays.Chile(years=años)
    df_feriados_comun = pd.DataFrame(ch_holidays.items(), columns=['ds', 'holiday'])
    df_feriados_comun['ds'] = pd.to_datetime(df_feriados_comun['ds'])
    df_feriados_comun = df_feriados_comun.sort_values(by='ds', ascending=True).reset_index(drop=True)

    # PROCESAR CADA LOCALIDAD CON PRINTS FILTRADOS
    for i, barra in enumerate(barras, 1):
        print(f"{'─'*80}")
        print(f"Barra {i}/{n_barras}: {barra}")
        print(f"{'─'*80}")

        # Usar context manager para filtrar prints
        with solo_prints_clave():
            df_train, df_val, df_test, df_fer = preprocesamiento_prophet(df_por_barra, barra, freq, año_inicio, año_fin, proporciones, umbral_outliers, graficos, df_feriados=df_feriados_comun)
        
        # Guardar resultados
        resultados[barra] = {
            'df_train': df_train,
            'df_val': df_val,
            'df_test': df_test,
            'df_feriados': df_fer
        }
        
        print()  # Línea en blanco entre localidades
    
    # Layout: 2 filas x 4 columnas
    n_filas = 2
    n_cols = 4
    
    fig, axes = plt.subplots(n_filas, n_cols, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, barra in enumerate(barras):
        ax = axes[i]
        
        df_train = resultados[barra]['df_train']
        df_val = resultados[barra]['df_val']
        df_test = resultados[barra]['df_test']
        
        # Graficar KDE para cada conjunto
        sns.kdeplot(df_train['y'], label='Train', fill=True, ax=ax, alpha=0.6, color='#1f77b4')
        sns.kdeplot(df_val['y'], label='Val', fill=True, ax=ax, alpha=0.6, color='#ff7f0e')
        sns.kdeplot(df_test['y'], label='Test', fill=True, ax=ax, alpha=0.6, color='#2ca02c')
        
        ax.set_title(f'{barra} \n Media: {df_train["y"].mean():.2f} | STD: {df_train["y"].std():.2f}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Valor', fontsize=10)
        ax.set_ylabel('Densidad', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Ocultar ejes sobrantes si hay menos de 8 barras
    for j in range(n_barras, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Distribución de Valores por Barra y Conjunto de Datos', fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.show()
    
    return resultados

# Función para generar tabla de estadísticas
def tabla_stats(df_train, df_val, df_test, col):

    df_completo = pd.concat([df_train, df_val, df_test])

    # Crear tabla de estadísticas
    tabla = pd.DataFrame({'Métrica': ['Fecha Inicial', 'Fecha Final', 'Valor Mínimo', 'Valor Máximo', 'Media', 'Desviación Estándar'],
                                'Train': [df_train['ds'].min(), df_train['ds'].max(), df_train[col].min(), df_train[col].max(), df_train[col].mean(), df_train[col].std()],
                                'Val': [df_val['ds'].min(), df_val['ds'].max(), df_val[col].min(), df_val[col].max(), df_val[col].mean(), df_val[col].std()],
                                'Test': [df_test['ds'].min(), df_test['ds'].max(), df_test[col].min(), df_test[col].max(), df_test[col].mean(), df_test[col].std()],
                                'Completo': [df_completo['ds'].min(), df_completo['ds'].max(), df_completo[col].min(), df_completo[col].max(), df_completo[col].mean(), df_completo[col].std()]})
    return tabla

# Función para generar estadísticas
def generar_estadisticas_todas_barras(prepro_por_barras, lista_barras):
    
    estadisticas_por_barra = {}
    
    for barra in lista_barras:
        print(f"\n{'─'*80}")
        print(f"BARRA: {barra}")
        print("─"*80)
        
        # Obtener dataframes
        df_train = prepro_por_barras[barra]['df_train']
        df_val = prepro_por_barras[barra]['df_val']
        df_test = prepro_por_barras[barra]['df_test']
        
        # Generar tabla de estadísticas
        tabla = tabla_stats(df_train, df_val, df_test, col='y')
        
        # Guardar en diccionario
        estadisticas_por_barra[barra] = tabla
        
        # Mostrar
        fmt = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
        display(tabla.style.format({col: fmt for col in ['Train', 'Val', 'Test', 'Completo']}))
    
    return estadisticas_por_barra

# ====================== FUNCIONES DE PROCESAMIENTO DE DATOS ======================

# Analiza si faltan datos
def analizar_completitud(df, columna_fecha, freq, año_inicio, año_fin):

    inicio = df[columna_fecha].min()
    fin = df[columna_fecha].max()

    rango_teorico = pd.date_range(start=inicio, end=fin, freq=freq)

    fechas_reales = set(df[columna_fecha])
    fechas_faltantes = sorted(set(rango_teorico) - fechas_reales)

    print('\n' + "="*50)
    print(f"ANÁLISIS DE COMPLETITUD EN LAS FECHAS")
    print(f"Fechas esperadas: {len(rango_teorico)} | Fechas reales: {len(df)}")
    print(f"Fechas faltantes: {len(fechas_faltantes)}%)")

    if fechas_faltantes:
        print(f"Detalle por año:")
        for year in range(año_inicio, año_fin + 1):
            n_falt = sum(1 for f in fechas_faltantes if f.year == year)
            if n_falt > 0:
                print(f"   - {year}: {n_falt:,} huecos")

    return fechas_faltantes

def completar_fechas(df, frecuencia, columna_fecha):

    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])

    if df.duplicated(subset=[columna_fecha]).any():
        df = df.groupby(columna_fecha, as_index=False).mean(numeric_only=True)

    df = df.sort_values(columna_fecha)

    fecha_inicio = df[columna_fecha].min()
    fecha_fin = df[columna_fecha].max()

    print("=" * 50)
    print("COMPLETANDO FECHAS FALTANTES")
    print(f"Rango de fechas: {fecha_inicio} a {fecha_fin}")

    rango_completo = pd.date_range(start=fecha_inicio, end=fecha_fin, freq=frecuencia)
    df_completo = df.set_index(columna_fecha).reindex(rango_completo)
    df_completo = df_completo.reset_index().rename(columns={'index': columna_fecha})

    agregados = len(df_completo) - len(df)
    print(f"RESULTADOS:")
    print(f"   Registros agregados (NaN): {agregados:,}")
    return df_completo

def dividir_serie_temporal(df, proporciones):

    if len(proporciones) != 3:
        raise ValueError("La lista 'proporciones' debe contener exactamente 3 valores (Train, Val, Test).")
    if not (0.99 <= sum(proporciones) <= 1.01):
        raise ValueError(f"La suma de las proporciones debe ser 1.0, pero se obtuvo: {sum(proporciones):.2f}")

    n = len(df)
    n_train = int(proporciones[0] * n)
    n_val = int(proporciones[1] * n)

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train : n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val :].copy()

    if len(df_train) + len(df_val) + len(df_test) != n:
        print("Advertencia: El total de registros divididos no coincide con el original.")

    print("Resumen de la división de la serie temporal")
    print(f"Total registros: {n}")
    print(f"Train: {len(df_train)} ({len(df_train)/n:.2%})")
    print(f"Validation: {len(df_val)} ({len(df_val)/n:.2%})")
    print(f"Test: {len(df_test)} ({len(df_test)/n:.2%})")

    return df_train, df_val, df_test

def preprocesar_serie(df, col_fecha, col_valor, freq, rango_años, zscore_stats, umbral_outliers=3):

    print("Iniciando preprocesamiento de la serie")

    is_training_set = (zscore_stats is None)
    df_clean = df.copy()

    df_clean = df_clean.rename(columns={col_fecha: 'ds', col_valor: 'y'})
    df_clean['ds'] = pd.to_datetime(df_clean['ds'])

    y_no_nan = df_clean['y'].dropna()

    if is_training_set:
        mean = y_no_nan.mean()
        std = y_no_nan.std()
        zscore_stats = {'mean': mean, 'std': std}
        print(f"Estadísticas del conjunto Train: Media={mean:.2f}, Std={std:.2f}")
    else:
        mean = zscore_stats['mean']
        std = zscore_stats['std']

    z_score = np.abs((y_no_nan - mean) / std)
    outliers = z_score[z_score >= umbral_outliers]

    if not outliers.empty:
        print(f"Outliers detectados (Z-Score >= {umbral_outliers}): {len(outliers)}")
        df_clean.loc[outliers.index, 'y'] = np.nan
    else:
        print("No se detectaron outliers significativos.")

    df_clean = df_clean.set_index('ds').asfreq(freq)
    df_clean['y'] = df_clean['y'].interpolate(method='linear')
    df_clean['y'] = df_clean['y'].bfill().ffill()
    df_clean = df_clean.reset_index()

    return df_clean, zscore_stats

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

# función test de estacionariedad
def test_estacionariedad(diccionario_dfs, columna_objetivo='Valor'):
    resultados = []

    for nombre_barra, df in diccionario_dfs.items():
        serie = df[columna_objetivo].dropna()
        
        # Test ADF
        adf_res = adfuller(serie)
        # Test KPSS (regression='c' asume constante, 'ct' asume tendencia)
        kpss_res = kpss(serie, regression='c')
        
        resultados.append({
            'Barra': nombre_barra,
            'ADF Stat': round(adf_res[0], 4),
            'ADF p-value': round(adf_res[1], 4),
            'KPSS Stat': round(kpss_res[0], 4),
            'KPSS p-value': round(kpss_res[1], 4),
            'Estacionaria (ADF)': 'Sí' if adf_res[1] < 0.05 else 'No',
            'Estacionaria (KPSS)': 'Sí' if kpss_res[1] > 0.05 else 'No'
        })
    
    return pd.DataFrame(resultados)