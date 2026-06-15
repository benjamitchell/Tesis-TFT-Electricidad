import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from datetime import timedelta
import matplotlib.dates as mdates

# Al inicio del código, agrega:
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.transparent'] = False  # Evita transparencias
plt.rcParams['figure.facecolor'] = 'white'   # Fondo blanco
plt.rcParams['savefig.facecolor'] = 'white'

# ===== CONFIGURACIÓN DE FECHAS PARA ZOOM =====
# Define aquí las fechas para el zoom de cada barra
# Formato: 'YYYY-MM-DD'
ZOOM_CONFIG = {
    'fecha_inicio': '2024-05-15',  # Fecha de inicio del zoom
    'fecha_fin': '2024-05-30',     # Fecha de fin del zoom
    # También puedes especificar por barra:
    # 'barra_1': {'inicio': '2023-01-01', 'fin': '2023-01-15'},
    # 'barra_2': {'inicio': '2023-06-01', 'fin': '2023-06-15'},
}
# ============================================

# Cargar datos
df = pd.read_csv('C:\\Users\\56977\\OneDrive\\Escritorio\\Tesis - copia\\Datos\\h\\2020-2026.csv', sep=';', decimal=',')

# Convertir tipos de datos
df['Barra'] = df['Barra'].astype(str)
df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce')
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Filtrar datos válidos
df = df.dropna(subset=['Valor', 'Fecha'])
df = df.sort_values('Fecha')

# Convertir fechas de configuración
fecha_inicio_zoom_config = pd.to_datetime(ZOOM_CONFIG['fecha_inicio'])
fecha_fin_zoom_config = pd.to_datetime(ZOOM_CONFIG['fecha_fin'])

# Obtener barras únicas
barras = df['Barra'].unique()
n_barras = len(barras)

# Calcular número de filas y columnas para subplots
n_cols = min(2, n_barras)
n_rows = (n_barras + n_cols - 1) // n_cols

# Crear figura con subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten() if n_barras > 1 else [axes]

# Graficar cada barra en un subplot
for i, barra in enumerate(barras):
    df_barra = df[df['Barra'] == barra].sort_values('Fecha')
    
    # Gráfico principal
    axes[i].plot(df_barra['Fecha'], df_barra['Valor'], linewidth=1.5, color='steelblue')
    axes[i].set_title(f'Barra {barra}', fontweight='bold')
    axes[i].set_xlabel('Fecha')
    axes[i].set_ylabel('Valor')
    axes[i].grid(True, alpha=0.3)
    axes[i].tick_params(axis='x', rotation=45)
    
    # Filtrar zoom por fechas específicas
    df_zoom = df_barra[(df_barra['Fecha'] >= fecha_inicio_zoom_config) & 
                       (df_barra['Fecha'] <= fecha_fin_zoom_config)].copy()
    
    # Crear inset axes - Ajustar posición Y para subir el contenido (reducir el espacio superior)
    inset_x, inset_y, inset_width, inset_height = 0.55, 0.60, 0.4, 0.35  # Aumenté inset_y de 0.55 a 0.60
    inset_ax = axes[i].inset_axes([inset_x, inset_y, inset_width, inset_height])
    
    if len(df_zoom) > 0:
        inset_ax.plot(df_zoom['Fecha'], df_zoom['Valor'], linewidth=1.8, color='darkred', marker='.', markersize=2, alpha=0.7)
        # Título eliminado (comentado)
        inset_ax.set_ylabel('Valor', fontsize=7)
        inset_ax.grid(True, alpha=0.3)
        
        # Configurar eje X solo con fechas de inicio y fin
        inset_ax.set_xticks([df_zoom['Fecha'].min(), df_zoom['Fecha'].max()])
        inset_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        inset_ax.tick_params(axis='x', rotation=45, labelsize=7)
        inset_ax.tick_params(axis='y', labelsize=6)
        
        # Ajustar límites
        fecha_margin = timedelta(hours=12)
        valor_min, valor_max = df_zoom['Valor'].min(), df_zoom['Valor'].max()
        valor_margin = (valor_max - valor_min) * 0.05
        inset_ax.set_xlim(df_zoom['Fecha'].min() - fecha_margin, df_zoom['Fecha'].max() + fecha_margin)
        inset_ax.set_ylim(valor_min - valor_margin, valor_max + valor_margin)
        
        # Información del zoom
        horas_zoom = len(df_zoom)
        inset_ax.text(0.02, 0.98, f'{horas_zoom} horas\n({horas_zoom/24:.1f} días)', 
                     transform=inset_ax.transAxes, fontsize=5,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Marcar área en gráfico principal
        rect = Rectangle((df_zoom['Fecha'].min(), df_zoom['Valor'].min()),
                        df_zoom['Fecha'].max() - df_zoom['Fecha'].min(),
                        df_zoom['Valor'].max() - df_zoom['Valor'].min(),
                        linewidth=1, edgecolor='red', facecolor='none', alpha=0.3, linestyle='--')
        axes[i].add_patch(rect)
    else:
        inset_ax.text(0.5, 0.5, 'Sin datos\nen el período seleccionado', 
                     transform=inset_ax.transAxes, ha='center', va='center', fontsize=8)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

# Ocultar subplots vacíos
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Evolución de Valores por Barra con Zoom en Fechas Específicas\n', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('grafico_zoom_fechas_especificas.pdf', dpi=300, bbox_inches='tight', format='pdf')
print("Gráfico guardado como 'grafico_zoom_fechas_especificas.jpg'")
plt.close()