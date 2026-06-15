# entrenar_prophet_horario.py
import sys
import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Agregar la carpeta actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from Modulos.Prophet_Modular import multi_prophet

def main():
    print("="*80)
    print("ENTRENAMIENTO DE MODELOS PROPHET - DATOS HORARIOS")
    print("="*80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. CONFIGURACIÓN
    freq = 'h'  # horario
    modo = 'entrenar'  # 'entrenar' o 'cargar'
    growth = 'linear'  # 'linear' o 'logistic'
    techo = 1.5  # solo para logistic
    mostrar_graficos = False  # False para terminal (True para notebook)
    mostrar_componentes = False
    mostrar_params = True
    
    # 2. RUTAS DE ARCHIVOS
    pkl_path = 'prepro_por_barras_horario.pkl'
    params_json_path = 'Modelos_Prophet/h/params_prophet_h_2020-2026.json'
    carpeta_modelos = 'Modelos_Prophet/h/modelos'
    
    # Crear carpeta para modelos si no existe
    os.makedirs(carpeta_modelos, exist_ok=True)
    
    # 3. VERIFICAR ARCHIVOS NECESARIOS
    print("Verificando archivos necesarios...")
    
    if not os.path.exists(pkl_path):
        print(f"❌ Error: No se encuentra {pkl_path}")
        print("   Primero ejecuta en tu notebook el código para guardar prepro_por_barras")
        return
    
    if not os.path.exists(params_json_path):
        print(f"❌ Error: No se encuentra {params_json_path}")
        print("   Primero ejecuta la optimización de parámetros")
        return
    
    print(f"✅ preprocesamiento: {pkl_path}")
    print(f"✅ parámetros: {params_json_path}\n")
    
    # 4. CARGAR PREPROCESAMIENTO
    print("Cargando preprocesamiento...")
    with open(pkl_path, 'rb') as f:
        prepro_por_barras = pickle.load(f)
    
    print(f"✅ Cargado: {len(prepro_por_barras)} barras")
    
    # Mostrar información de los datos
    primera_barra = list(prepro_por_barras.keys())[0]
    print(f"\nEstructura de datos (ejemplo {primera_barra}):")
    print(f"  Train: {len(prepro_por_barras[primera_barra]['df_train'])} registros")
    print(f"  Val: {len(prepro_por_barras[primera_barra]['df_val'])} registros")
    print(f"  Test: {len(prepro_por_barras[primera_barra]['df_test'])} registros\n")
    
    # 5. CARGAR PARÁMETROS OPTIMIZADOS
    print("Cargando parámetros optimizados...")
    with open(params_json_path, 'r', encoding='utf-8') as f:
        parametros_optimizados = json.load(f)
    
    print(f"✅ Cargados parámetros para {len(parametros_optimizados)} barras\n")
    
    # 6. PARÁMETROS BASE
    params_base = {
        'daily_seasonality': True,   # Importante para datos horarios
        'weekly_seasonality': True,
        'yearly_seasonality': True
    }

    # Modificar los parámetros optimizados para incluir daily_seasonality
    parametros_con_daily = {}
    for barra, params in parametros_optimizados.items():
        parametros_con_daily[barra] = {
            **params,  # copiar los parámetros existentes
            'daily_seasonality': True,   # forzar daily_seasonality=True
            'weekly_seasonality': True,
            'yearly_seasonality': True
        }
    
    print("Configuración del entrenamiento:")
    print(f"  Frecuencia: {freq}")
    print(f"  Modo: {modo}")
    print(f"  Growth: {growth}")
    print(f"  Techo: {techo if growth == 'logistic' else 'N/A'}")
    print(f"  Mostrar gráficos: {mostrar_graficos}")
    print(f"  Mostrar componentes: {mostrar_componentes}")
    print(f"  Carpeta modelos: {carpeta_modelos}\n")
    
    # 7. EJECUTAR ENTRENAMIENTO
    print("="*80)
    print("INICIANDO ENTRENAMIENTO DE MODELOS")
    print("="*80)
    
    try:
        modelos, df_predicciones, df_metricas = multi_prophet(
            prepro_por_barras=prepro_por_barras,
            parametros=parametros_con_daily,
            params_base=params_base,
            modo=modo,
            mostrar_params=mostrar_params,
            mostrar_graficos=mostrar_graficos,
            mostrar_componentes=mostrar_componentes,
            growth=growth,
            techo=techo
        )
        
        # 8. GUARDAR RESULTADOS
        print("\n" + "="*80)
        print("GUARDANDO RESULTADOS")
        print("="*80)
        
        # Guardar modelos entrenados (usando pickle)
        for barra, modelo in modelos.items():
            modelo_path = os.path.join(carpeta_modelos, f'modelo_{barra}.pkl')
            with open(modelo_path, 'wb') as f:
                pickle.dump(modelo, f)
            print(f"✅ Modelo guardado: {modelo_path}")
        
        # Guardar predicciones
        predicciones_path = f'predicciones_prophet_horario_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        df_predicciones.to_csv(predicciones_path, sep=';', decimal=',', index=False)
        print(f"✅ Predicciones guardadas: {predicciones_path}")
        
        # Guardar métricas
        metricas_path = f'metricas_prophet_horario_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        df_metricas.to_csv(metricas_path, sep=';', decimal=',', index=False)
        print(f"✅ Métricas guardadas: {metricas_path}")
        
        # 9. MOSTRAR RESUMEN DE MÉTRICAS
        print("\n" + "="*80)
        print("RESUMEN DE MÉTRICAS POR BARRA (TEST)")
        print("="*80)
        
        df_test_metrics = df_metricas[df_metricas['Conjunto'] == 'Test']
        if not df_test_metrics.empty:
            for _, row in df_test_metrics.iterrows():
                print(f"{row['Barra']:12} | MAE: {row['MAE']:8.2f} | RMSE: {row['RMSE']:8.2f} | R2: {row['R2']:.4f}")
        
        # 10. GUARDAR CONFIGURACIÓN DEL ENTRENAMIENTO
        config = {
            'fecha_entrenamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'freq': freq,
            'growth': growth,
            'techo': techo,
            'params_base': params_base,
            'num_barras': len(modelos),
            'carpeta_modelos': carpeta_modelos
        }
        
        config_path = f'config_entrenamiento_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Configuración guardada: {config_path}")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nFin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()