import requests
import pandas as pd
import time
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TimeSeriesDataSet

# Coordenadas de las 8 barras del SEN
# Fuentes: https://www.geodatos.net/coordenadas/chile/iquique y https://www.igm.cl
COORDENADAS_BARRAS = {
    'ATACAMA':  (-28.57617, -70.75938),   # Vallenar
    'CARDONES': (-27.36737, -70.33219),   # Copiapó
    'CHARRUA':  (-36.82699, -73.04977),   # Concepción
    'CRUCERO':  (-23.65094, -70.39752),   # Antofagasta
    'P.AZUCAR': (-29.90591, -71.25014),   # La Serena
    'P.MONTT':  (-41.4693,  -72.94237),   # Puerto Montt
    'QUILLOTA': (-33.036,   -71.62963),   # Valparaíso
    'TARAPACA': (-20.21326, -70.15027),   # Iquique
}

# Función para obtener datos climáticos desde Open-Meteo
def obtener_clima_historico(lat, lon, fecha_inicio, fecha_fin, freq):
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {"latitude": lat,
              "longitude": lon,
              "start_date": fecha_inicio,
              "end_date": fecha_fin,
              "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "cloud_cover"],
              "timezone": "America/Santiago"}
    
    # Configuración de paciencia
    max_retries = 10        # Intentaremos hasta 10 veces
    wait_time = 20          # Empezamos esperando 20 segundos si falla
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            
            # Si nos bloquean (429), lanzamos error manual para activar la espera
            if response.status_code == 429:
                print(f"BLOQUEO API (429) detectado en intento {attempt+1}...", end=" ")
                raise requests.exceptions.RequestException("Rate Limit Exceeded")

            response.raise_for_status()
            data = response.json()
            
            # Procesar datos
            df = pd.DataFrame({'ds': pd.to_datetime(data['hourly']['time']),
                               'temperatura': data['hourly']['temperature_2m'],
                               'humedad': data['hourly']['relative_humidity_2m'],
                               'velocidad_viento': data['hourly']['wind_speed_10m'],
                               'precipitacion': data['hourly']['precipitation'],
                               'nubosidad': data['hourly']['cloud_cover']})
            
            # Si la frecuencia es diaria reagrupamos
            if freq == 'D':
                df = df.resample('D', on='ds').agg({'temperatura': 'mean',
                                                    'humedad': 'mean',
                                                    'velocidad_viento': 'mean',
                                                    'precipitacion': 'sum',
                                                    'nubosidad': 'mean'}).reset_index()

            return df

        except Exception as e:
            # Si fallamos en el último intento, nos rendimos
            if attempt == max_retries - 1:
                print(f"\nError fatal tras {max_retries} intentos: {e}")
                return None
            
            # Estrategia de espera exponencial: 20s -> 40s -> 80s...
            print(f"Esperando {wait_time}s para reintentar...", end=" ", flush=True)
            time.sleep(wait_time)
            wait_time *= 2 # Duplicamos la espera cada vez
            
    return None

# Función para obtener datos climáticos con manejo de errores
def generar_clima(fecha_inicio, fecha_fin, coordenadas, carpeta_salida, freq):

    dict_clima = {}
    total = len(coordenadas)
    
    os.makedirs(carpeta_salida, exist_ok=True)
    for i, (nombre, (lat, lon)) in enumerate(coordenadas.items(), 1):
        nombre_archivo = f"{nombre}_{freq}.csv"
        ruta_completa = os.path.join(carpeta_salida, nombre_archivo)
        
        # Verificamos si existe el archivo
        if os.path.exists(ruta_completa):
            try:
                df = pd.read_csv(ruta_completa, parse_dates=['ds'])
                #print(f"EXISTE -> Cargado ({len(df)} filas)")
                dict_clima[nombre] = df
                continue
            except:
                print("Archivo corrupto, re-descargando", end=" ")

        # Descargamos si es necesario
        print("DESCARGANDO...", end=" ", flush=True)
        df = obtener_clima_historico(lat, lon, fecha_inicio, fecha_fin, freq)
        
        if df is not None:
            df.to_csv(ruta_completa, index=False)
            dict_clima[nombre] = df
            print("GUARDADO.")
            
            # Esperamos 10 segundos entre descargas exitosas para enfriar la API y evitar otro 429 inmediato.
            time.sleep(10) 
        else:
            print("FALLÓ DEFINITIVAMENTE.")
    
    return dict_clima

# Juntamos las predicciones con los datos climáticos
def predicciones_con_clima(df_preds, dict_clima):

    diccionario_final = {}
    
    # Obtenemos la lista de barras en las predicciones
    localidades = df_preds['Barra'].unique()
    
    for localidad in localidades:

        # Filtramos las predicciones solo para esta localidad
        mask = df_preds['Barra'] == localidad
        df_subset = df_preds[mask].copy()
        
        # Buscamos si existe clima para esta localidad
        if localidad in dict_clima:
            df_clima_loc = dict_clima[localidad]
            
            # Juntamos los dataframes
            df_merged = pd.merge(df_subset, df_clima_loc, on='ds', how='left')
            
            # Guardamos el diccionario
            diccionario_final[localidad] = df_merged
        
        else:
            print(f"{localidad:<10}: No se encontraron datos climáticos.")
            diccionario_final[localidad] = df_subset
            
    return diccionario_final

# Juntar y convertir feriados, 1 si es y 0 si no es feriado
def holiday_binary(df, holiday):

    # Juntar el df con el df_feriados
    df = df.merge(holiday, on='ds', how='left')

    # Crear columna con 1 si es feriado y 0 si no
    df['is_holiday'] = df['holiday'].notna().astype(int)

    # Eliminar la columna del feriado
    df = df.drop(columns = ['holiday'])
    return df

def incluir_features_horario(dict_datos):

    dict_enriquecido = {}
    
    for localidad, df in dict_datos.items():

        df = df.copy()
        df = df.sort_values('ds').reset_index(drop=True)

        nan_mask = df['residuo'].isna()
        df.loc[nan_mask, 'residuo'] = df.loc[nan_mask, 'y_real'] - df.loc[nan_mask, 'yhat']
        
        # ── Características temporales de calendario ──────────────────────────
        df['hour']        = df['ds'].dt.hour
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['month']       = df['ds'].dt.month
        df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

        # ── Codificación cíclica ───────────────────────────────────────────────
        # Hora del día (ciclo de 24h)
        df['hour_sin']      = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos']      = np.cos(2 * np.pi * df['hour'] / 24)
        # Día de la semana (ciclo de 7 días)
        df['day_week_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_week_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)
        # Mes (ciclo de 12 meses)
        df['month_sin']     = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos']     = np.cos(2 * np.pi * df['month'] / 12)

        # ── Lags (en horas) ───────────────────────────────────────────────────
        df['y_lag1']   = df['y_real'].shift(1)    # 1 hora atrás
        df['y_lag2']   = df['y_real'].shift(2)    # 2 horas atrás
        df['y_lag3']   = df['y_real'].shift(3)    # 3 horas atrás
        df['y_lag6']   = df['y_real'].shift(6)    # 6 horas atrás
        df['y_lag8']   = df['y_real'].shift(8)    # 8 horas atrás
        df['y_lag12']  = df['y_real'].shift(12)   # 12 horas atrás
        df['y_lag24']  = df['y_real'].shift(24)   # mismo momento ayer
        df['y_lag168'] = df['y_real'].shift(168)  # mismo momento hace 1 semana
        df['y_lag336'] = df['y_real'].shift(336)  # mismo momento hace 2 semanas
        df['y_lag720'] = df['y_real'].shift(720)  # mismo momento hace 30 días
        df['y_lag8760']= df['y_real'].shift(8760) # mismo momento hace 1 año

        # Lags de residuos
        df['resid_lag1']   = df['residuo'].shift(1)
        df['resid_lag2']   = df['residuo'].shift(2)
        df['resid_lag3']   = df['residuo'].shift(3)
        df['resid_lag6']   = df['residuo'].shift(6)
        df['resid_lag8']   = df['residuo'].shift(8)
        df['resid_lag12']  = df['residuo'].shift(12)
        df['resid_lag24']  = df['residuo'].shift(24)
        df['resid_lag168'] = df['residuo'].shift(168)
        df['resid_lag336'] = df['residuo'].shift(336)
        df['resid_lag720'] = df['residuo'].shift(720)
        df['resid_lag8760']= df['residuo'].shift(8760)

        # ── Rolling windows (en horas) ────────────────────────────────────────
        # Últimas 24h (patrón diario)
        df['rolling_mean_24'] = df['y_real'].shift(1).rolling(window=24).mean()
        df['rolling_std_24']  = df['y_real'].shift(1).rolling(window=24).std()
        # Últimos 7 días = 168h (patrón semanal)
        df['rolling_mean_168'] = df['y_real'].shift(1).rolling(window=168).mean()
        df['rolling_std_168']  = df['y_real'].shift(1).rolling(window=168).std()

        # Rolling windows de residuos
        df['rolling_mean_resid_24']  = df['residuo'].shift(1).rolling(window=24).mean()
        df['rolling_std_resid_24']   = df['residuo'].shift(1).rolling(window=24).std()
        df['rolling_mean_resid_168'] = df['residuo'].shift(1).rolling(window=168).mean()
        df['rolling_std_resid_168']  = df['residuo'].shift(1).rolling(window=168).std()

        # ── Limpieza de NaNs generados por lags ──────────────────────────────
        # Los primeros 168 registros (1 semana) tendrán NaN por lag168/rolling_168
        df = df.dropna().reset_index(drop=True)

        # ── Parámetros obligatorios para PyTorch Forecasting ──────────────────
        df['time_idx'] = range(len(df))
        df['serie_id'] = localidad

        dict_enriquecido[localidad] = df

    print(f"Cols agregadas. Total features: {df.shape[1]}")

    return dict_enriquecido

# Función para normalizar 
def normalizar_datos(datasets, feature_cols, target_cols):
    scalers_finales = {}
    
    for localidad, sets in datasets.items():
        
        # Guardar y_real original antes de normalizar
        if 'y_real' in target_cols and 'y_real' in sets['train'].columns:
            y_real_original_train = sets['train']['y_real'].values.copy()
        else:
            y_real_original_train = None
        
        # --- SOLUCIÓN: Separar yhat de los features normales ---
        yhat_cols = ['yhat', 'yhat_lower', 'yhat_upper']
        features_x_puros = [col for col in feature_cols if col not in yhat_cols]
        
        # Normalizar features (ahora sin yhat)
        scaler_x = StandardScaler()
        if features_x_puros:
            sets['train'][features_x_puros] = scaler_x.fit_transform(sets['train'][features_x_puros])
            for set_name in ['val', 'test']:
                sets[set_name][features_x_puros] = scaler_x.transform(sets[set_name][features_x_puros])
        else:
            scaler_x.fit([[0]])  # scaler vacío pero válido para serializar
        
        # Normalizar targets
        scalers_y = {target: StandardScaler() for target in target_cols}
        for target in target_cols:
            sets['train'][[target]] = scalers_y[target].fit_transform(sets['train'][[target]])
            for set_name in ['val', 'test']:
                sets[set_name][[target]] = scalers_y[target].transform(sets[set_name][[target]])
        
        # Normalizar yhat EXCLUSIVAMENTE con el scaler de y_real
        yhat_cols_existentes = [col for col in yhat_cols if col in sets['train'].columns]
        
        if yhat_cols_existentes and 'y_real' in target_cols and y_real_original_train is not None:
            scaler_yhat = StandardScaler()
            # Ajustamos usando los datos reales crudos
            scaler_yhat.fit(y_real_original_train.reshape(-1, 1))
            
            for set_name in ['train', 'val', 'test']:
                for col in yhat_cols_existentes:
                    # Como aquí yhat NO fue tocado por scaler_x, esta es su primera y única transformación
                    sets[set_name][col] = scaler_yhat.transform(sets[set_name][[col]])
            
            scalers_y['yhat'] = scaler_yhat
        
        # Guardar scalers
        scalers_finales[localidad] = {'scaler_x': scaler_x,
                                      'scalers_target': scalers_y}
        
    return datasets, scalers_finales

# Función para construir los conjuntos de train, val y test
def dividir_serie_temporal(df, proporciones, verbose=False):
    
    # Validación de inputs
    if len(proporciones) != 3:
        raise ValueError("La lista 'proporciones' debe contener exactamente 3 valores (Train, Val, Test).")
    if not (0.99 <= sum(proporciones) <= 1.01):
        # Permite una pequeña tolerancia por errores de punto flotante
        raise ValueError(f"La suma de las proporciones debe ser 1.0, pero se obtuvo: {sum(proporciones):.2f}")
        
    # Calculamos los puntos de corte
    n = len(df)
    n_train = int(proporciones[0] * n)
    n_val = int(proporciones[1] * n)
    
    # Hacemos la división temporal
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train : n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val :].copy()
    
    # Verificamos el tamaño
    if len(df_train) + len(df_val) + len(df_test) != n:
        print("Advertencia: El total de registros divididos no coincide con el original.")

    if verbose:
        print("Resumen de la división de la serie temporal")
        print(f"Total registros: {n}")
        print(f"Train: {len(df_train)} ({len(df_train)/n:.2%})")
        print(f"Validation: {len(df_val)} ({len(df_val)/n:.2%})")
        print(f"Test: {len(df_test)} ({len(df_test)/n:.2%})")
    
    return df_train, df_val, df_test

def limpiar_nans(lista_barras, datasets_norm, verbose=False):
    # Limpiamos los NaNs
    for barra in lista_barras:
        for set_name in ['train', 'val', 'test']:
            df = datasets_norm[barra][set_name]
            n_antes = len(df)
            cols_con_nan = df.columns[df.isna().any()].tolist()

            if cols_con_nan:
                print(f"\n{barra} - {set_name}: Columnas con NaNs:")
                for col in cols_con_nan:
                    n_nan = df[col].isna().sum()
                    print(f"  - {col}: {n_nan} NaNs")

            df_clean = df.dropna()
            n_despues = len(df_clean)
            n_eliminados = n_antes - n_despues

            if n_eliminados > 0:
                print(f"Eliminados {n_eliminados} registros")

            df_clean = df_clean.reset_index(drop=True)
            datasets_norm[barra][set_name] = df_clean

    # Recalculamos el time_idx con continuidad
    for barra in lista_barras:
        df_train = datasets_norm[barra]['train']
        df_val = datasets_norm[barra]['val']
        df_test = datasets_norm[barra]['test']
        
        df_train['time_idx'] = range(len(df_train))
        offset_val = len(df_train)
        df_val['time_idx'] = range(offset_val, offset_val + len(df_val))
        offset_test = offset_val + len(df_val)
        df_test['time_idx'] = range(offset_test, offset_test + len(df_test))
        
        datasets_norm[barra]['train'] = df_train
        datasets_norm[barra]['val'] = df_val
        datasets_norm[barra]['test'] = df_test

        if verbose:
            print(f"{barra}: Train[0-{df_train['time_idx'].max()}] | Val[{df_val['time_idx'].min()}-{df_val['time_idx'].max()}] | Test[{df_test['time_idx'].min()}-{df_test['time_idx'].max()}]")

    return datasets_norm

def crear_dataloaders(lista_barras, datasets_norm, known_reals, unknown_reals_price, unknown_reals_resid, max_encoder_length, max_prediction_length, batch_size):
    
    # Creamos los Dataset y Dataloaders
    datasets_precios = {}
    datasets_residuos = {}
    dataloaders_precios = {}
    dataloaders_residuos = {}

    for barra in lista_barras:
        
        # Obtenemos los splits normalizados
        df_train = datasets_norm[barra]['train'].copy()
        df_val = datasets_norm[barra]['val'].copy()
        df_test = datasets_norm[barra]['test'].copy()
        
        # Creamos los Datasets base (train)
        dataset_precios = TimeSeriesDataSet(
            df_train,
            time_idx="time_idx",
            target="y_real",
            group_ids=["serie_id"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["serie_id"],
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals_price,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True)
        
        dataset_residuos = TimeSeriesDataSet(
            df_train,
            time_idx="time_idx",
            target="residuo",
            group_ids=["serie_id"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["serie_id"],
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals_resid,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True)
        
        # Concatenamos, sin extender test con val, para VAL: Train + Val
        df_train_val = pd.concat([df_train, df_val], ignore_index=True)
        df_train_val = df_train_val.sort_values('ds').reset_index(drop=True)
        
        # Usamos los time_idx originales que ya están continuos
        train_time_idx = df_train['time_idx'].values
        val_time_idx = df_val['time_idx'].values
        df_train_val['time_idx'] = list(train_time_idx) + list(val_time_idx)
        df_train_val['serie_id'] = barra
        
        # Para TEST: Train + Val + Test
        df_train_val_test = pd.concat([df_train, df_val, df_test], ignore_index=True)
        df_train_val_test = df_train_val_test.sort_values('ds').reset_index(drop=True)
        
        # Mantenemos la continuidad de time_idx
        test_time_idx = df_test['time_idx'].values
        df_train_val_test['time_idx'] = list(train_time_idx) + list(val_time_idx) + list(test_time_idx)
        df_train_val_test['serie_id'] = barra
        
        # Verificamos la continuidad
        assert df_train_val['time_idx'].is_monotonic_increasing, f"❌ {barra}: train_val time_idx NO continuo"
        assert df_train_val_test['time_idx'].is_monotonic_increasing, f"❌ {barra}: train_val_test time_idx NO continuo"

        # Crear Datasets extendidos
        val_dataset_precios = TimeSeriesDataSet.from_dataset(dataset_precios, df_train_val, stop_randomization=True)
        test_dataset_precios = TimeSeriesDataSet.from_dataset(dataset_precios, df_train_val_test, stop_randomization=True)
        
        val_dataset_residuos = TimeSeriesDataSet.from_dataset(dataset_residuos, df_train_val, stop_randomization=True)
        test_dataset_residuos = TimeSeriesDataSet.from_dataset(dataset_residuos, df_train_val_test, stop_randomization=True)
        
        # Crear dataloaders
        dataloaders_precios[barra] = {'train': dataset_precios.to_dataloader(train=True, batch_size=batch_size, shuffle=True),
                                    'val': val_dataset_precios.to_dataloader(train=False, batch_size=batch_size * 10, shuffle=False),
                                    'test': test_dataset_precios.to_dataloader(train=False, batch_size=batch_size * 10, shuffle=False)}
        
        dataloaders_residuos[barra] = {'train': dataset_residuos.to_dataloader(train=True, batch_size=batch_size, shuffle=True),
                                    'val': val_dataset_residuos.to_dataloader(train=False, batch_size=batch_size * 10, shuffle=False),
                                    'test': test_dataset_residuos.to_dataloader(train=False, batch_size=batch_size * 10, shuffle=False)}
    
        # Guardamos los Datasets
        datasets_precios[barra] = {'train': dataset_precios,
                                'val': val_dataset_precios,
                                'test': test_dataset_precios}
        
        datasets_residuos[barra] = {'train': dataset_residuos,
                                    'val': val_dataset_residuos,
                                    'test': test_dataset_residuos}
        
    return dataloaders_precios, dataloaders_residuos, datasets_precios, datasets_residuos


def agregar_features_solares(datos_para_transformer, coordenadas):
    try:
        import pvlib
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pvlib', '-q'])
        import pvlib

    _CAMBIOS_INVIERNO = pd.to_datetime([
        '2020-04-04','2021-04-03','2022-04-02','2023-04-01',
        '2024-04-06','2025-04-05','2026-04-04',
    ])
    _CAMBIOS_VERANO = pd.to_datetime([
        '2019-09-07','2020-09-05','2021-09-04','2022-09-10',
        '2023-09-02','2024-09-07','2025-09-06',
    ])
    _REGIMENES = sorted(
        [(f, 1) for f in _CAMBIOS_VERANO] + [(f, 0) for f in _CAMBIOS_INVIERNO],
        key=lambda x: x[0]
    )

    def _es_verano(ts):
        ts = pd.Timestamp(ts)
        regimen = 1  # enero 2020 comienza en verano (UTC-3)
        for fecha, val in _REGIMENES:
            if fecha <= ts:
                regimen = val
            else:
                break
        return regimen

    print("Agregando features solares a datos_para_transformer...")

    for barra, df in datos_para_transformer.items():
        lat, lon = coordenadas[barra]

        df['es_horario_verano'] = df['ds'].apply(_es_verano).astype(float)

        ds_local = pd.to_datetime(df['ds'])
        ds_utc = ds_local.dt.tz_localize(
            'America/Santiago',
            ambiguous='NaT',
            nonexistent='shift_forward'
        ).dt.tz_convert('UTC')

        nat_mask = ds_utc.isna()
        if nat_mask.any():
            ds_utc = ds_utc.ffill() + pd.Timedelta(hours=1)

        hora_utc = ds_utc.dt.hour
        df['hora_utc_sin'] = np.sin(2 * np.pi * hora_utc / 24)
        df['hora_utc_cos'] = np.cos(2 * np.pi * hora_utc / 24)

        sol = pvlib.solarposition.get_solarposition(ds_utc, lat, lon)
        df['elevacion_solar'] = sol['elevation'].values
        df['cos_elevacion']   = np.cos(np.radians(sol['elevation'].values))

        datos_para_transformer[barra] = df

    print("✓ Features solares agregadas: es_horario_verano, hora_utc_sin/cos, elevacion_solar, cos_elevacion")
    return datos_para_transformer


def guardar_experimento(carpeta, dataloaders_dict_precios, dataloaders_dict_residuos,
                        diccionario_scalers, feature_config, tft_config_total):
    import pickle, json

    def _convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert_ndarray(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_ndarray(i) for i in obj]
        return obj

    os.makedirs(carpeta, exist_ok=True)

    with open(f'{carpeta}/dataloaders_precios.pkl', 'wb') as f:
        pickle.dump(dataloaders_dict_precios, f)
    print("✓ Dataloaders precios guardados")

    with open(f'{carpeta}/dataloaders_residuos.pkl', 'wb') as f:
        pickle.dump(dataloaders_dict_residuos, f)
    print("✓ Dataloaders residuos guardados")

    with open(f'{carpeta}/scalers.pkl', 'wb') as f:
        pickle.dump(diccionario_scalers, f)
    print("✓ Scalers guardados")

    with open(f'{carpeta}/feature_config.json', 'w') as f:
        json.dump(_convert_ndarray(feature_config), f, indent=2)
    print("✓ Feature config guardado")

    with open(f'{carpeta}/tft_config.json', 'w') as f:
        json.dump(tft_config_total, f, indent=2)
    print("✓ TFT config guardado")


def verificar_dataloaders(ruta_precios, ruta_residuos):
    import pickle

    def _ver(ruta, nombre):
        print(f"\n{'='*60}")
        print(f" {nombre}")
        print(f"{'='*60}")
        with open(ruta, "rb") as f:
            dl = pickle.load(f)
        barras = list(dl.keys())
        print(f"Barras disponibles ({len(barras)}): {barras}")
        for barra in barras:
            dataset = dl[barra]["train"].dataset
            print(f"\n  ── {barra} ──")
            print(f"  time_varying_known_reals:    {dataset.time_varying_known_reals}")
            print(f"  time_varying_unknown_reals:  {dataset.time_varying_unknown_reals}")
            print(f"  static_reals:                {dataset.static_reals}")
            print(f"  time_varying_known_cats:     {dataset.time_varying_known_categoricals}")
            print(f"  static_cats:                 {dataset.static_categoricals}")
            print(f"  target:                      {dataset.target}")

    _ver(ruta_precios,  "DATALOADERS PRECIOS")
    _ver(ruta_residuos, "DATALOADERS RESIDUOS")