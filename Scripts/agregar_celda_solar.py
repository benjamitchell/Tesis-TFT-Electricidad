"""
Inyecta una celda de features solares (hora UTC + elevación solar) en la
sección de Preprocesamiento Transformer del notebook TFT Horario,
y actualiza feature_cols y known_reals para incluir las nuevas variables.
"""
import json, uuid

NOTEBOOK_PATH = r'C:\Users\56977\OneDrive\Escritorio\Tesis - copia\Multi_modelo TFT Horario.ipynb'

IDX_INCLUIR_FEATURES = 12   # celda con incluir_features_horario
IDX_FEATURE_COLS     = 14   # celda con feature_cols  (pre-inserción)
IDX_KNOWN_REALS      = 15   # celda con known_reals   (pre-inserción)

# ── Celda nueva: hora UTC + elevación solar ───────────────────────────────────
CODE_SOLAR = r"""# ── Features solares: hora UTC + elevación astronómica ──────────────────────
# Requiere: pip install pvlib
# Estas features son covariables FUTURAS CONOCIDAS (known_reals) porque
# la posición del sol es determinista para cualquier fecha y coordenada.

try:
    import pvlib
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pvlib', '-q'])
    import pvlib

import numpy as np
import pandas as pd

# ── Fechas de cambio de hora Chile Continental (para es_horario_verano) ──────
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
    # Devuelve 1 si el timestamp esta en UTC-3 (verano), 0 si en UTC-4 (invierno)
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
    lat, lon = Coordenadas[barra]

    # ── 1. es_horario_verano ─────────────────────────────────────────────────
    df['es_horario_verano'] = df['ds'].apply(_es_verano).astype(float)

    # ── 2. Convertir hora local → UTC ────────────────────────────────────────
    ds_local = pd.to_datetime(df['ds'])
    ds_utc = ds_local.dt.tz_localize(
        'America/Santiago',
        ambiguous='NaT',        # hora ambigua (retroceso) → NaT
        nonexistent='shift_forward'  # hora inexistente (adelanto) → siguiente hora válida
    ).dt.tz_convert('UTC')

    # Rellenar NaTs (hora ambigua): usar hora anterior + 1h en UTC
    nat_mask = ds_utc.isna()
    if nat_mask.any():
        ds_utc = ds_utc.fillna(method='ffill') + pd.Timedelta(hours=1)

    hora_utc = ds_utc.dt.hour

    # Codificación cíclica de la hora UTC (ciclo de 24h)
    df['hora_utc_sin'] = np.sin(2 * np.pi * hora_utc / 24)
    df['hora_utc_cos'] = np.cos(2 * np.pi * hora_utc / 24)

    # ── 3. Elevación solar astronómica (pvlib) ───────────────────────────────
    sol = pvlib.solarposition.get_solarposition(ds_utc, lat, lon)

    df['elevacion_solar'] = sol['elevation'].values          # grados, -90 a +90
    df['cos_elevacion']   = np.cos(np.radians(sol['elevation'].values))  # 0..1

    datos_para_transformer[barra] = df

print("✓ Features solares agregadas:")
print("  - es_horario_verano  (1=UTC-3, 0=UTC-4)  → known_real")
print("  - hora_utc_sin/cos   (hora UTC cíclica)   → known_real")
print("  - elevacion_solar    (grados, -90..+90)   → known_real")
print("  - cos_elevacion      (proxy irradiancia)  → known_real")

_barra_ejemplo = list(datos_para_transformer.keys())[0]
_df_ej = datos_para_transformer[_barra_ejemplo]
print(f"\nEjemplo {_barra_ejemplo} — primeras 3 filas:")
display(_df_ej[['ds','es_horario_verano','hora_utc_sin','hora_utc_cos',
                 'elevacion_solar','cos_elevacion']].head(3))
"""

# ── Nuevo contenido de feature_cols (con las 4 features solares) ─────────────
CODE_FEATURE_COLS = r"""from Modulos.Preprocesamiento_Transformer import normalizar_datos, limpiar_nans

# Definimos los features y targets
feature_cols = ['trend', 'yearly', 'weekly', 'daily',                                           # Variables Prophet
                'temperatura', 'humedad', 'velocidad_viento', 'precipitacion', 'nubosidad',     # Variables climáticas
                'is_holiday',                                                                   # Indicador de feriado
                'y_lag1', 'y_lag24','y_lag168', 'resid_lag1', 'resid_lag24', 'resid_lag168',
                'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168', 'rolling_std_168',
                'rolling_mean_resid_24', 'rolling_std_resid_24', 'rolling_mean_resid_168', 'rolling_std_resid_168',
                'es_horario_verano',                                                            # Régimen horario (UTC-3/UTC-4)
                'hora_utc_sin', 'hora_utc_cos',                                                # Hora UTC cíclica
                'elevacion_solar', 'cos_elevacion']                                            # Posición solar astronómica

# Normalizamos
datasets_norm, diccionario_scalers = normalizar_datos(datasets_transformer, feature_cols, targets)

datasets_norm = limpiar_nans(lista_barras, datasets_norm)

#  Validamos las escalas, rangos y continuidad de time_idx en cada barra
for barra in lista_barras:
    print(f"\n{barra}:")
    df_train = datasets_norm[barra]['train']

    # Escalas
    print(f"  1. Escalas:")
    print(f"     y_real:  media={df_train['y_real'].mean():.3f}, std={df_train['y_real'].std():.3f}")
    print(f"     yhat:    media={df_train['yhat'].mean():.3f},   std={df_train['yhat'].std():.3f}")
    diff_std = abs(df_train['y_real'].std() - df_train['yhat'].std())

    # Continuidad time_idx
    df_val = datasets_norm[barra]['val']
    df_test = datasets_norm[barra]['test']

    train_max = df_train['time_idx'].max()
    val_min = df_val['time_idx'].min()
"""

# ── Nuevo contenido de known_reals (con las 4 features solares) ──────────────
CODE_KNOWN_REALS = r"""from Modulos.Preprocesamiento_Transformer import crear_dataloaders

# Configuración de features
known_reals = ['trend', 'yearly', 'weekly', 'daily', 'is_holiday',
               'es_horario_verano',           # Régimen UTC-3 / UTC-4 (conocido de antemano)
               'hora_utc_sin', 'hora_utc_cos', # Hora UTC cíclica  (conocida de antemano)
               'elevacion_solar', 'cos_elevacion']  # Posición solar (determinista)

var_clima = ['temperatura', 'humedad', 'velocidad_viento', 'precipitacion', 'nubosidad']

unknown_reals_price = ['y_real',
                       'y_lag1', 'y_lag24', 'y_lag168',
                       'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168', 'rolling_std_168'] + var_clima

unknown_reals_resid = ['residuo',
                       'resid_lag1', 'resid_lag24', 'resid_lag168',
                       'rolling_mean_resid_24', 'rolling_std_resid_24', 'rolling_mean_resid_168', 'rolling_std_resid_168'] + var_clima

dataloaders_precios, dataloaders_residuos, datasets_precios, datasets_residuos = crear_dataloaders(lista_barras, datasets_norm,
                                                                                                   known_reals, unknown_reals_price, unknown_reals_resid,
                                                                                                   max_encoder_length, max_prediction_length, batch_size)

# Validaciones
for barra in lista_barras:
    print(f"\n{barra}:")
"""

# ── Leer notebook y aplicar cambios ──────────────────────────────────────────
with open(NOTEBOOK_PATH, encoding='utf-8') as f:
    nb = json.load(f)

n_original = len(nb['cells'])
print(f"Celdas antes: {n_original}")

def make_code(source):
    return {"cell_type": "code", "execution_count": None,
            "id": uuid.uuid4().hex[:8], "metadata": {}, "outputs": [],
            "source": source}

# 1. Insertar celda solar después de IDX_INCLUIR_FEATURES
nueva_celda = make_code(CODE_SOLAR)
nb['cells'].insert(IDX_INCLUIR_FEATURES + 1, nueva_celda)
# Ahora IDX_FEATURE_COLS y IDX_KNOWN_REALS se desplazan +1
idx_fc = IDX_FEATURE_COLS + 1
idx_kr = IDX_KNOWN_REALS + 1

# 2. Reemplazar celda feature_cols
nb['cells'][idx_fc] = make_code(CODE_FEATURE_COLS)

# 3. Reemplazar celda known_reals
nb['cells'][idx_kr] = make_code(CODE_KNOWN_REALS)

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Celdas después: {len(nb['cells'])} (+1 nueva celda solar)")
print(f"✓ Celda solar inyectada en posición {IDX_INCLUIR_FEATURES + 1}")
print(f"✓ feature_cols actualizado (posición {idx_fc})")
print(f"✓ known_reals actualizado (posición {idx_kr})")
print("\nSiguientes pasos en el notebook:")
print("  1. Recargar el notebook (Ctrl+Shift+P → Revert File)")
print("  2. Ejecutar la nueva celda solar ANTES de normalizar")
print("  3. Reentrenar el TFT con los nuevos dataloaders")
