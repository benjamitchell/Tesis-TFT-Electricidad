import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURACIÓN =====
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'serif'

# Rutas
CARPETA_MODELOS = 'C:\\Users\\56977\\OneDrive\\Escritorio\\Tesis - copia\\Modelos_Prophet\\h\\modelos'
CARPETA_SALIDA = 'C:\\Users\\56977\\OneDrive\\Escritorio\\Tesis - copia\\Modelos_Prophet\\h\\Graficos_Prophet'
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ===== FUNCIONES =====
def cargar_modelos(barras, carpeta_modelos):
    modelos = {}
    print(f"  {'Barra':<12} {'CPS':>8} {'SPS':>8} {'Mode':<12} {'Daily':<8}")
    print(f"  {'-'*52}")
    for barra in barras:
        ruta = os.path.join(carpeta_modelos, f'modelo_{barra}.pkl')
        if os.path.exists(ruta):
            with open(ruta, 'rb') as f:
                modelos[barra] = pickle.load(f)
            m = modelos[barra]
            print(f"  {barra:<12} {m.changepoint_prior_scale:>8} "
                  f"{m.seasonality_prior_scale:>8} "
                  f"{m.seasonality_mode:<12} {str(m.daily_seasonality):<8}")
        else:
            print(f"❌ No encontrado: Barra {barra}")
    return modelos

def generar_predicciones(modelo, df_historial, prediccion=48):
    last_date = df_historial['ds'].max()
    future_dates = pd.date_range(start=last_date, periods=prediccion+1, freq='h')[1:]
    
    df_future = pd.DataFrame({'ds': future_dates})
    
    if modelo.growth == 'logistic':
        df_future['cap'] = modelo.history['cap'].iloc[0]
        df_future['floor'] = modelo.history['floor'].iloc[0]
    
    forecast = modelo.predict(df_future)
    
    return forecast

def extraer_componentes_agrupados(modelos, df_historico_dict):
    componentes = {
        'trend': [],
        'yearly': [],
        'weekly': [],
        'daily': [],
        'holidays': []
    }
    
    for barra, modelo in modelos.items():
        print(f"  Extrayendo componentes de Barra {barra}...")
        
        df_hist = df_historico_dict[barra]
        
        df_plot = df_hist[['ds']].copy()
        if modelo.growth == 'logistic':
            df_plot['cap'] = modelo.history['cap'].iloc[0]
            df_plot['floor'] = modelo.history['floor'].iloc[0]
        
        forecast = modelo.predict(df_plot)
        
        for comp in componentes.keys():
            if comp in forecast.columns:
                df_comp = forecast[['ds', comp]].copy()
                df_comp['Barra'] = barra
                componentes[comp].append(df_comp)
    
    for comp in componentes:
        if componentes[comp]:
            componentes[comp] = pd.concat(componentes[comp], ignore_index=True)
        else:
            componentes[comp] = pd.DataFrame()
    
    return componentes

def graficar_predicciones_por_barra(modelos, df_historico_dict, prediccion=48):
    n_barras = len(modelos)
    n_cols = 2
    n_rows = (n_barras + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    axes = axes.flatten() if n_barras > 1 else [axes]
    
    for idx, (barra, modelo) in enumerate(modelos.items()):
        print(f"  Graficando predicciones Barra {barra}...")
        
        forecast = generar_predicciones(modelo, df_historico_dict[barra], prediccion)
        df_hist = df_historico_dict[barra]
        
        ax = axes[idx]
        
        dias_hist = min(7*24, len(df_hist))
        df_hist_reciente = df_hist.tail(dias_hist)
        
        ax.plot(df_hist_reciente['ds'], df_hist_reciente['y'], 
                'b-', linewidth=1.5, label='Histórico', alpha=0.7)
        ax.plot(forecast['ds'], forecast['yhat'], 
                'r-', linewidth=2, label=f'Predicción {prediccion}h')
        
        cps = modelo.changepoint_prior_scale
        sps = modelo.seasonality_prior_scale
        ax.set_title(f'Barra {barra} - Predicción {prediccion} horas\n'
                     f'CPS={cps} | SPS={sps}', fontweight='bold')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('CMg [USD/MWh]')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    for j in range(idx+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'Predicciones Prophet por Barra\n', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(CARPETA_SALIDA, 'predicciones_por_barra.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def graficar_componentes_agrupados(componentes, tipo_componente):
    if componentes[tipo_componente].empty:
        print(f"  No hay datos para {tipo_componente}")
        return
    
    barras = componentes[tipo_componente]['Barra'].unique()
    n_barras = len(barras)
    
    n_cols = 2
    n_rows = (n_barras + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    axes = axes.flatten() if n_barras > 1 else [axes]
    
    for idx, barra in enumerate(barras):
        df_comp = componentes[tipo_componente][componentes[tipo_componente]['Barra'] == barra]
        
        ax = axes[idx]
        ax.plot(df_comp['ds'], df_comp[tipo_componente], linewidth=1.5)
        ax.set_title(f'Barra {barra}', fontweight='bold')
        ax.set_xlabel('Fecha')
        ax.set_ylabel(f'{tipo_componente.capitalize()}')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    for j in range(idx+1, len(axes)):
        axes[j].set_visible(False)
    
    titulo = f'Componente: {tipo_componente.upper()}\n'
    plt.suptitle(titulo, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(CARPETA_SALIDA, f'componente_{tipo_componente}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def graficar_comparativa_modelos(modelos, df_historico_dict):
    N_INSET = 14 * 24  # 2 semanas en horas para el inset

    n_barras = len(modelos)
    n_cols = 2
    n_rows = (n_barras + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    axes = axes.flatten() if n_barras > 1 else [axes]

    for idx, (barra, modelo) in enumerate(modelos.items()):
        df_hist = df_historico_dict[barra]

        df_plot = df_hist[['ds']].copy()
        if modelo.growth == 'logistic':
            df_plot['cap'] = modelo.history['cap'].iloc[0]
            df_plot['floor'] = modelo.history['floor'].iloc[0]

        forecast = modelo.predict(df_plot)

        ax = axes[idx]
        ax.plot(df_hist['ds'], df_hist['y'], 'b-', linewidth=1.5, label='Real', alpha=0.8)
        ax.plot(forecast['ds'], forecast['yhat'], 'r-', linewidth=1.5, label='Predicho', alpha=0.8)

        cps = modelo.changepoint_prior_scale
        sps = modelo.seasonality_prior_scale
        ax.set_title(f'Barra {barra}\n'
                     f'CPS={cps} | SPS={sps}', fontweight='bold')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('CMg [USD/MWh]')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # ── Inset: zoom a las últimas 2 semanas ─────────────────────────
        n_in = min(N_INSET, len(df_hist))
        ax_in = ax.inset_axes([0.62, 0.55, 0.36, 0.38])
        ax_in.plot(df_hist['ds'].tail(n_in), df_hist['y'].tail(n_in),
                   'b-', linewidth=1.2, alpha=0.8)
        ax_in.plot(forecast['ds'].tail(n_in), forecast['yhat'].tail(n_in),
                   'r-', linewidth=1.2, alpha=0.8)
        ax_in.tick_params(labelsize=6)
        ax_in.tick_params(axis='x', rotation=30)
        ax_in.grid(True, alpha=0.3, linewidth=0.5)
        ax.indicate_inset_zoom(ax_in, edgecolor='0.4', linewidth=1.0)

    for j in range(idx+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Comparativa Real vs Predicción\n', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(CARPETA_SALIDA, 'comparativa_real_vs_predicho.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

# ===== MAIN =====
def main():
    print("="*60)
    print("SCRIPT DE GRAFICACIÓN DE MODELOS PROPHET")
    print("="*60)
    
    print("\n1. Cargando datos históricos...")
    with open('prepro_por_barras_horario.pkl', 'rb') as f:
         prepro_por_barras = pickle.load(f)
    
    df_historico = {}
    for barra in prepro_por_barras.keys():
        dfs = []
        for conjunto in ['df_train', 'df_val', 'df_test']:
            if conjunto in prepro_por_barras[barra]:
                df_conj = prepro_por_barras[barra][conjunto].copy()
                dfs.append(df_conj)
        df_historico[barra] = pd.concat(dfs, ignore_index=True)
    print(f"  ✅ Datos cargados para {len(df_historico)} barras")
    
    print("\n2. Cargando modelos Prophet...")
    barras = list(df_historico.keys())
    modelos = cargar_modelos(barras, CARPETA_MODELOS)
    
    if not modelos:
        print("  ❌ No se encontraron modelos")
        return
    
    print(f"\n✅ Modelos cargados: {len(modelos)}/{len(barras)}")
    
    print("\n3. Extrayendo componentes...")
    componentes = extraer_componentes_agrupados(modelos, df_historico)
    
    print("\n4. Generando gráficos...")
    
    print("\n  📊 Predicciones por barra...")
    graficar_predicciones_por_barra(modelos, df_historico, prediccion=48)
    
    print("\n  📊 Comparativa real vs predicción...")
    graficar_comparativa_modelos(modelos, df_historico)
    
    componentes_disponibles = ['trend', 'yearly', 'weekly', 'daily', 'holidays']
    for comp in componentes_disponibles:
        if not componentes[comp].empty:
            print(f"\n  📊 Graficando componente: {comp}...")
            graficar_componentes_agrupados(componentes, comp)
    
    print("\n" + "="*60)
    print(f"✅ PROCESO COMPLETADO")
    print(f"📁 Gráficos guardados en: {CARPETA_SALIDA}")
    print("="*60)
    
    print("\nArchivos generados:")
    for archivo in os.listdir(CARPETA_SALIDA):
        if archivo.endswith('.png'):
            tamaño = os.path.getsize(os.path.join(CARPETA_SALIDA, archivo)) / 1024
            print(f"  - {archivo} ({tamaño:.1f} KB)")

if __name__ == "__main__":
    main()