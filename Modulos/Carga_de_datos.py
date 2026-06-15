import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import math
import warnings

# Suprimimos alertas de pandas para limpieza
warnings.filterwarnings('ignore')

def Carga_Universal_Chile(ruta_base, mapeo_columnas, renombrar_cols=None,
                          ordenar_cols=None, eliminar_cols=None, rango_anos=None,
                          extension="*.csv", sep=";", decimal=',',
                          verbose=True, mostrar_graficos=True):
    
    if verbose:
        print(f"Cargando desde: {ruta_base}")
        print(f"Configuración: Ext='{extension}' | Sep='{sep}' | Decimal='{decimal}'")
    
    # Búsqueda Recursiva
    patron_busqueda = os.path.join(ruta_base, "**", extension)
    archivos = glob.glob(patron_busqueda, recursive=True)
    
    if not archivos:
        print(f"Error: No se encontraron archivos {extension} en {ruta_base}")
        return {}, []
    
    if verbose: print(f"Archivos encontrados: {len(archivos)}")

    # Lectura y Concatenación
    lista_dfs = []
    for archivo in archivos:
        try:
            if extension.endswith('xlsx') or extension.endswith('xls'):
                df_temp = pd.read_excel(archivo, dtype=str)
            else:
                df_temp = pd.read_csv(archivo, sep=sep, engine='python', dtype=str)
            
            lista_dfs.append(df_temp)
        except Exception as e:
            print(f"Error leyendo {os.path.basename(archivo)}: {e}")

    if not lista_dfs: return {}, []

    df_total = pd.concat(lista_dfs, ignore_index=True)
    
    # Normalización y Mapeo
    df_total.columns = df_total.columns.str.lower().str.strip()
    mapeo_ajustado = {k.lower(): v for k, v in mapeo_columnas.items()}
    
    # Filtramos el mapeo solo para columnas que existen en el DF
    mapeo_final = {k: v for k, v in mapeo_ajustado.items() if k in df_total.columns}
    
    # Renombramos
    df_total = df_total.rename(columns=mapeo_final)

    # Eliminación de columnas
    if eliminar_cols:
        df_total = df_total.drop(columns=eliminar_cols, errors='ignore')

    # Lógica de fechas
    if verbose: print("Procesando Fechas y Horas")
    
    cols = df_total.columns.tolist()

    # Formato Antiguo (year, month, day separados)
    # El mapeo debió renombrar las columnas originales a 'year', 'month', 'day'
    if all(x in cols for x in ['year', 'month', 'day']):

        if verbose: print("Detección: Formato [Año, Mes, Día]")

        df_total['Fecha'] = pd.to_datetime(df_total[['year', 'month', 'day']])
        df_total = df_total.drop(columns=['year', 'month', 'day'], errors='ignore')

    # Formato Nuevo (Fecha + Hora 1-24)
    # El mapeo debió renombrar a 'Fecha' y 'Hora'
    elif 'Fecha' in cols and 'Hora' in cols:
        
        if verbose: print("Detección: Formato [Fecha + Hora (1-24)]")
        
        fechas = pd.to_datetime(df_total['Fecha'], errors='coerce')
        horas = pd.to_numeric(df_total['Hora'], errors='coerce')
        
        # Lógica CEN: Fecha + (Hora - 1)
        df_total['Fecha'] = fechas + pd.to_timedelta(horas - 1, unit='h')
        df_total = df_total.drop(columns=['Hora'], errors='ignore')

    # Formato solo Fecha
    elif 'Fecha' in cols:
        if verbose: print("Detección: Formato [Solo Fecha]")

        df_total['Fecha'] = pd.to_datetime(df_total['Fecha'], errors='coerce')
    
    else:
        print(f"Error Crítico: No se pudo construir la fecha. Columnas actuales: {cols}")
        print("   Revisa que tu diccionario 'mapeo_columnas' apunte a los nombres correctos.")
        return {}, []

    # Limpieza de valores y barras
    if 'Valor' in df_total.columns:
        df_total['Valor'] = df_total['Valor'].astype(str).str.replace(decimal, '.', regex=False)
        df_total['Valor'] = pd.to_numeric(df_total['Valor'], errors='coerce')
    
    if 'Barra' in df_total.columns:
        df_total['Barra'] = df_total['Barra'].astype(str).str.strip()
        
        # Cambiamos los nombres de las barras
        if renombrar_cols:
            if verbose: 
                print(f"Renombrando barras.")
            df_total['Barra'] = df_total['Barra'].replace(renombrar_cols)

        # Limpieza automática, quita ____ y códigos extra si no fueron renombrados
        df_total['Barra'] = df_total['Barra'].str.split('__', expand=True)[0].str.strip()

    # Eliminar NaNs críticos (filas vacías o errores de parseo)
    df_total = df_total.dropna(subset=['Fecha', 'Valor'])

    # Filtro de años
    if rango_anos:
        anio_i, anio_f = rango_anos
        if verbose: print(f"Filtrando años: {anio_i} - {anio_f}")
        df_total = df_total[(df_total['Fecha'].dt.year >= anio_i) & (df_total['Fecha'].dt.year <= anio_f)]

    # Reordenamiento de las columnas
    if ordenar_cols:
        if verbose: print(f"Reordenando columnas: {ordenar_cols}")

        # Filtramos solo las columnas que realmente existen para no causar error
        cols_validas = [c for c in ordenar_cols if c in df_total.columns]
        
        # Avisar si faltó alguna
        faltantes = set(ordenar_cols) - set(cols_validas)
        if faltantes and verbose:
            print(f"Aviso: Las siguientes columnas pedidas no existen y se ignorarán: {faltantes}")
        
        df_total = df_total[cols_validas]

    # Ordenar Final
    df_total = df_total.sort_values(['Barra', 'Fecha']).reset_index(drop=True)

    # Resumen
    if verbose:
        print("="*50)
        print(f"Rango de fechas: {df_total['Fecha'].min()} a {df_total['Fecha'].max()}")
        print(f"Rango de valores: {df_total['Valor'].min()} a {df_total['Valor'].max()}")
        print(f"Número de barras: {df_total['Barra'].nunique()}")
        print(f"Registros totales: {len(df_total):,}")
        print("="*50)

    # Generamos el diccionario
    df_por_barra = {loc: data.reset_index(drop=True) for loc, data in df_total.groupby('Barra')}

    # Visualización
    if mostrar_graficos:
        num_plots = len(df_por_barra)
        if num_plots > 0:
            columnas = 4
            
            # Calculamos filas
            filas = math.ceil(num_plots / columnas)
            
            # Forzamos mínimo 2 filas, aunque haya pocas barras
            if filas < 2: filas = 2
            
            fig, axes = plt.subplots(filas, columnas, figsize=(20, 5 * filas), sharex=False)
            axes = axes.flatten()

            for i, (barra, df) in enumerate(df_por_barra.items()):

                # Protección de índice
                if i < len(axes):
                    ax = axes[i]
                    sns.lineplot(x='Fecha', y='Valor', data=df, ax=ax, linewidth=1.0, color='mediumblue')
                    
                    # Título dinámico
                    min_year = df["Fecha"].dt.year.min()
                    max_year = df["Fecha"].dt.year.max()
                    ax.set_title(f'{barra} ({min_year} - {max_year})', fontweight='bold')
                    
                    ax.tick_params(axis='x', rotation=45)

            # Borrar ejes vacíos
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()
    
    return df_por_barra, list(df_por_barra.keys())