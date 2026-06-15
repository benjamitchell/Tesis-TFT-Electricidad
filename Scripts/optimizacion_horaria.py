# optimizacion_horaria_final.py
import sys
import os
import pickle
import pandas as pd
from datetime import datetime

# Agregar la carpeta actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from Modulos.Prophet_Modular import optimizacion_prophet

print("="*80)
print("OPTIMIZACIÓN PROPHET - DATOS HORARIOS")
print("="*80)
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 1. CARGAR EL PREPROCESAMIENTO GUARDADO
print("Cargando preprocesamiento guardado...")
with open('prepro_por_barras_horario.pkl', 'rb') as f:
    prepro_por_barras = pickle.load(f)

print(f"✅ Preprocesamiento cargado")
print(f"   Barras: {len(prepro_por_barras)}")
print(f"   Ejemplo estructura: {list(prepro_por_barras.keys())[0]}")
print(f"   Train size: {len(prepro_por_barras[list(prepro_por_barras.keys())[0]]['df_train'])}")
print(f"   Val size: {len(prepro_por_barras[list(prepro_por_barras.keys())[0]]['df_val'])}")
print(f"   Test size: {len(prepro_por_barras[list(prepro_por_barras.keys())[0]]['df_test'])}\n")

# 2. PARÁMETROS PARA OPTIMIZACIÓN HORARIA
freq = 'h'  # horario
ano_inicio = 2020
ano_fin = 2026

# Ajustar parámetros para datos horarios
parametros = {
    'grid': {
        'changepoint_prior_scale': [0.01, 0.02, 0.05, 0.1],  # Valores más pequeños para horario
        'seasonality_prior_scale': [0.5, 1, 5, 10],
        'seasonality_mode': ['additive'],  # Multiplicative puede dar problemas
        'growth': ['linear']  # Logistic requiere definir techo
    },
    'daily_seasonality': True,   # Importante para datos horarios
    'weekly_seasonality': True,
    'yearly_seasonality': True,
    'interval_width': 0.95
}

# 3. CONFIGURACIÓN
ruta_archivo = f"Modelos_Prophet/{freq}/params_prophet_{freq}_{ano_inicio}-{ano_fin}_v2.json"
modo = 'optimizar'  # 'optimizar' o 'usar'
metrica = 'MAE'

# Crear directorio
os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)

# Calcular total de combinaciones
total_combinaciones = len(parametros['grid']['changepoint_prior_scale']) * \
                     len(parametros['grid']['seasonality_prior_scale']) * \
                     len(parametros['grid']['seasonality_mode'])

print(f"Configuración:")
print(f"   Frecuencia: {freq}")
print(f"   Grid: {total_combinaciones} combinaciones por barra")
print(f"   Métrica: {metrica}")
print(f"   Modo: {modo}")
print(f"   Usar CV: False (Train/Val simple)")
print(f"   Salida: {ruta_archivo}\n")

# 4. EJECUTAR OPTIMIZACIÓN
print("Iniciando optimización...")
print("="*80)

mejores_params = optimizacion_prophet(
    prepro_por_barras, 
    ruta_archivo, 
    parametros, 
    modo, 
    metrica, 
    usar_cv=False,  # False para más rápido, True para más robusto
    n_splits=5, 
    hmap=False
)

# 5. RESULTADOS
print("\n" + "="*80)
print("RESULTADOS DE LA OPTIMIZACIÓN")
print("="*80)

df_params = pd.DataFrame.from_dict(mejores_params, orient='index')
df_params.index.name = 'Localidad'
df_params.reset_index(inplace=True)

print("\nTABLA DE HIPERPARÁMETROS PROPHET:")
print(df_params.to_string())

# Guardar CSV con resultados
output_csv = f'parametros_prophet_horarios_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
df_params.to_csv(output_csv, sep=';', decimal=',', index=False)
print(f"\n✅ Resultados guardados en: {output_csv}")

# Mostrar resumen
print("\n" + "="*80)
print("RESUMEN DE MÉTRICAS")
print("="*80)
for _, row in df_params.iterrows():
    print(f"{row['Localidad']:12} | CPS: {row['changepoint_prior_scale']:5.3f} | SPS: {row['seasonality_prior_scale']:5.1f} | {metrica}: {row[metrica]:8.2f}")

print(f"\n✅ Optimización completada")
print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")