"""
Script para agregar el análisis de cambios de hora al notebook TFT Horario.
Elimina las últimas 7 celdas (si ya existen del run anterior) y las reemplaza
con la versión corregida que resuelve el KeyError nombre→clave.
"""
import json
import uuid

NOTEBOOK_PATH = r'C:\Users\56977\OneDrive\Escritorio\Tesis - copia\Multi_modelo TFT Horario.ipynb'
N_CELDAS_AGREGAR = 7  # las que agregamos la vez anterior

def make_markdown(source):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": source}

def make_code(source):
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8],
            "metadata": {}, "outputs": [], "source": source}

# ── Celda 1: Título ──────────────────────────────────────────────────────────
MD_TITULO = """\
# Análisis: Errores extremos y cambios de hora en Chile

**Hipótesis:** Los errores más grandes del TFT ocurren en fechas cercanas a los cambios de hora \
en Chile, porque el modelo aprendió el patrón de transición solar según el reloj y se desajusta \
cuando el reloj adelanta o atrasa 1 hora.

**Cambios de hora confirmados (Chile Continental, excluye Magallanes):**
- **Inicio invierno (UTC-4, reloj atrasa 1 h):** primer sábado de abril cada año.
- **Inicio verano (UTC-3, reloj adelanta 1 h):** primer sábado de septiembre — excepción 2022: \
10-sep por decreto especial (DS N°224/2022).

Fuentes: directemar.cl, elmostrador.cl, decreto DS N°224/2022 Ministerio del Interior.\
"""

# ── Celda 2: Setup ───────────────────────────────────────────────────────────
CODE_SETUP = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Fechas de cambio de hora Chile Continental (2020-2026) ──
FECHAS_INVIERNO = pd.to_datetime([
    '2020-04-04', '2021-04-03', '2022-04-02', '2023-04-01',
    '2024-04-06', '2025-04-05', '2026-04-04',
])
FECHAS_VERANO = pd.to_datetime([
    '2019-09-07', '2020-09-05', '2021-09-04', '2022-09-10',
    '2023-09-02', '2024-09-07', '2025-09-06',
])
FECHAS_CAMBIO_HORA = FECHAS_INVIERNO.append(FECHAS_VERANO).sort_values()

TIPO_CAMBIO = {}
for _f in FECHAS_INVIERNO: TIPO_CAMBIO[str(_f.date())] = 'invierno'
for _f in FECHAS_VERANO:
    TIPO_CAMBIO[str(_f.date())] = 'verano*' if str(_f.date()) == '2022-09-10' else 'verano'

def _dias_min(ts):
    return int(np.abs((FECHAS_CAMBIO_HORA - pd.Timestamp(ts)).days).min())

def _cambio_cercano(ts):
    diffs = np.abs((FECHAS_CAMBIO_HORA - pd.Timestamp(ts)).days)
    return FECHAS_CAMBIO_HORA[diffs.argmin()]

def _get_key(resultados, barra, nombre_buscado):
    \"\"\"Resuelve nombre legible → clave real del dict de resultados.\"\"\"
    for key, val in resultados[barra].items():
        if not key.startswith('_') and isinstance(val, dict):
            if val.get('nombre') == nombre_buscado:
                return key
    # Fallback: buscar por MAE en validación (mismo criterio que error_distribucion)
    y_val = np.array(resultados[barra]['_datos_split']['y_real_val'])
    claves = [k for k in resultados[barra] if not k.startswith('_')]
    maes   = {k: np.abs(y_val - np.array(resultados[barra][k]['prediccion_val'])).mean()
              for k in claves}
    return min(maes, key=maes.get)

print(f"Fechas de cambio de hora cargadas: {len(FECHAS_CAMBIO_HORA)}")
display(pd.DataFrame({'Fecha': FECHAS_CAMBIO_HORA,
                      'Tipo':  [TIPO_CAMBIO[str(f.date())] for f in FECHAS_CAMBIO_HORA]}))\
"""

# ── Celda 3: Paso 1 ──────────────────────────────────────────────────────────
CODE_PASO1 = """\
# ── Paso 1: Top-20 errores por barra con indicador de cambio de hora ──
VENTANA_DIAS = 7

filas_top20 = []
for barra in lista_barras:
    nombre_est = df_mejores_ln.loc[df_mejores_ln['Barra'] == barra, 'Mejor Estrategia'].values[0]
    est_key    = _get_key(resultados_stacking_ln, barra, nombre_est)

    y_val      = np.array(resultados_stacking_ln[barra]['_datos_split']['y_real_val'])
    yhat_val   = np.array(resultados_stacking_ln[barra][est_key]['prediccion_val'])
    fechas_val = pd.to_datetime(resultados_stacking_ln[barra]['_datos_split']['fechas_val'])

    errores  = np.abs(y_val - yhat_val)
    df_barra = pd.DataFrame({
        'Barra': barra, 'Estrategia': nombre_est,
        'timestamp': fechas_val, 'y_real': y_val, 'y_pred': yhat_val, 'error_abs': errores,
    }).nlargest(20, 'error_abs').reset_index(drop=True)
    df_barra['rank'] = df_barra.index + 1

    df_barra['dias_al_cambio'] = df_barra['timestamp'].apply(_dias_min)
    df_barra['cambio_cercano'] = df_barra['timestamp'].apply(_cambio_cercano)
    df_barra['tipo_cambio']    = df_barra['cambio_cercano'].apply(
        lambda d: TIPO_CAMBIO.get(str(d.date()), '?'))
    df_barra['en_ventana_1d']  = df_barra['dias_al_cambio'] <= 1
    df_barra['en_ventana_7d']  = df_barra['dias_al_cambio'] <= VENTANA_DIAS
    filas_top20.append(df_barra)

df_top20_all = pd.concat(filas_top20, ignore_index=True)

resumen = df_top20_all.groupby('Barra').agg(
    n_en_1d=('en_ventana_1d', 'sum'),
    pct_1d =('en_ventana_1d', lambda x: f"{x.mean()*100:.0f}%"),
    n_en_7d=('en_ventana_7d', 'sum'),
    pct_7d =('en_ventana_7d', lambda x: f"{x.mean()*100:.0f}%"),
).reset_index()

print("=== Cuántos del top-20 caen en ventana de cambio de hora (por barra) ===")
display(resumen)
print("\\n=== Detalle top-20 (todas las barras) ===")
cols = ['Barra', 'rank', 'timestamp', 'error_abs', 'dias_al_cambio', 'cambio_cercano',
        'tipo_cambio', 'en_ventana_7d']
display(df_top20_all[cols].sort_values(['Barra', 'rank']))\
"""

# ── Celda 4: Paso 2 ──────────────────────────────────────────────────────────
CODE_PASO2 = """\
# ── Paso 2: MAE por grupo temporal (±1d, ±2-7d, resto del año) ──
filas_mae = []
for barra in lista_barras:
    nombre_est = df_mejores_ln.loc[df_mejores_ln['Barra'] == barra, 'Mejor Estrategia'].values[0]
    est_key    = _get_key(resultados_stacking_ln, barra, nombre_est)

    y_val      = np.array(resultados_stacking_ln[barra]['_datos_split']['y_real_val'])
    yhat_val   = np.array(resultados_stacking_ln[barra][est_key]['prediccion_val'])
    fechas_val = pd.to_datetime(resultados_stacking_ln[barra]['_datos_split']['fechas_val'])
    errores    = np.abs(y_val - yhat_val)
    dias       = np.array([_dias_min(t) for t in fechas_val])

    m1   = dias <= 1
    m7   = (dias > 1) & (dias <= 7)
    mres = dias > 7

    mae_1d   = errores[m1].mean()   if m1.any()   else np.nan
    mae_7d   = errores[m7].mean()   if m7.any()   else np.nan
    mae_rest = errores[mres].mean() if mres.any() else np.nan

    filas_mae.append({
        'Barra':           barra,
        'Estrategia':      nombre_est,
        'MAE ±1d':         mae_1d,
        'n ±1d':           int(m1.sum()),
        'MAE ±2-7d':       mae_7d,
        'n ±2-7d':         int(m7.sum()),
        'MAE resto':       mae_rest,
        'n resto':         int(mres.sum()),
        'Ratio ±1d/resto': round(mae_1d / mae_rest, 2) if not np.isnan(mae_1d) else np.nan,
    })

df_mae = pd.DataFrame(filas_mae).round(3)
prom   = df_mae[['MAE ±1d', 'MAE ±2-7d', 'MAE resto', 'Ratio ±1d/resto']].mean().round(3)
df_mae_display = pd.concat(
    [df_mae, pd.DataFrame([{'Barra': 'PROMEDIO', **prom.to_dict()}])], ignore_index=True)

print("=== MAE por grupo temporal — modelo LN ===")
display(df_mae_display[['Barra', 'Estrategia', 'MAE ±1d', 'n ±1d',
                          'MAE ±2-7d', 'n ±2-7d', 'MAE resto', 'n resto', 'Ratio ±1d/resto']])\
"""

# ── Celda 5: Paso 3A ─────────────────────────────────────────────────────────
CODE_PASO3A = """\
# ── Paso 3A: Serie completa + zooms ±14d alrededor de cada cambio de hora ──
for barra in lista_barras:
    nombre_est = df_mejores_ln.loc[df_mejores_ln['Barra'] == barra, 'Mejor Estrategia'].values[0]
    est_key    = _get_key(resultados_stacking_ln, barra, nombre_est)

    y_val      = np.array(resultados_stacking_ln[barra]['_datos_split']['y_real_val'])
    yhat_val   = np.array(resultados_stacking_ln[barra][est_key]['prediccion_val'])
    fechas_val = pd.to_datetime(resultados_stacking_ln[barra]['_datos_split']['fechas_val'])
    errores    = np.abs(y_val - yhat_val)

    f_min, f_max = fechas_val.min(), fechas_val.max()
    cambios_prox  = [f for f in FECHAS_CAMBIO_HORA
                     if (f_min - pd.Timedelta(days=14)) <= f <= (f_max + pd.Timedelta(days=14))]

    if not cambios_prox:
        print(f"{barra}: sin cambios de hora en el período de validación.")
        continue

    # ── Panel A1: serie completa ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(fechas_val, errores, color='steelblue', lw=0.5, alpha=0.7, label='Error absoluto')
    ax.axhline(errores.mean(), color='gray', lw=1, ls='--', label=f'MAE={errores.mean():.2f}')
    added = set()
    for fecha in cambios_prox:
        tipo  = TIPO_CAMBIO.get(str(fecha.date()), 'cambio')
        color = '#e74c3c' if 'verano' in tipo else '#2980b9'
        lbl   = f'Cambio {tipo}' if tipo not in added else '_nolegend_'
        added.add(tipo)
        ax.axvline(fecha, color=color, lw=1.5, ls=':', alpha=0.85, label=lbl)
    ax.set_title(f'{barra} [{nombre_est}] — Error absoluto (validación) con cambios de hora',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Error absoluto (USD/MWh)')
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.show()

    # ── Panel A2: zooms ±14 días por cada cambio dentro del período val ──────
    cambios_en_val = [f for f in FECHAS_CAMBIO_HORA if f_min <= f <= f_max]
    if not cambios_en_val:
        continue

    n = len(cambios_en_val)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows), sharey=True,
                             squeeze=False)
    axes_flat = axes.flatten()

    mae_global = errores.mean()
    for i, fecha in enumerate(cambios_en_val):
        ax = axes_flat[i]
        mask = ((fechas_val >= fecha - pd.Timedelta(days=14)) &
                (fechas_val <= fecha + pd.Timedelta(days=14)))
        tipo  = TIPO_CAMBIO.get(str(fecha.date()), 'cambio')
        color = '#e74c3c' if 'verano' in tipo else '#2980b9'
        mae_ventana = errores[mask].mean() if mask.any() else np.nan

        ax.plot(fechas_val[mask], errores[mask], color='steelblue', lw=0.9, alpha=0.85)
        ax.axhline(mae_global,   color='gray',  lw=1,   ls='--', alpha=0.6,
                   label=f'MAE global={mae_global:.2f}')
        ax.axhline(mae_ventana,  color=color,   lw=1.2, ls='-.',  alpha=0.75,
                   label=f'MAE ventana={mae_ventana:.2f}')
        ax.axvline(fecha, color=color, lw=2, ls=':', label=f'{tipo} ({fecha.date()})')
        ax.set_title(f'Zoom ±14d | {fecha.date()} ({tipo})', fontsize=9, fontweight='bold')
        ax.set_xlabel('Fecha')
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel('Error absoluto (USD/MWh)')
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f'{barra} — Zoom ±14 días alrededor de cada cambio de hora (validación)',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.show()\
"""

# ── Celda 6: Paso 3B ─────────────────────────────────────────────────────────
CODE_PASO3B = """\
# ── Paso 3B: Violinplot de errores por grupo temporal ──
n_cols = 4
n_rows = (len(lista_barras) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

COLORES = ['#e74c3c', '#f39c12', '#2980b9']
LABELS  = ['±1d\\ncambio hora', '±2–7d\\ncambio hora', 'Resto\\ndel año']

for i, barra in enumerate(lista_barras):
    ax = axes[i]
    nombre_est = df_mejores_ln.loc[df_mejores_ln['Barra'] == barra, 'Mejor Estrategia'].values[0]
    est_key    = _get_key(resultados_stacking_ln, barra, nombre_est)

    y_val      = np.array(resultados_stacking_ln[barra]['_datos_split']['y_real_val'])
    yhat_val   = np.array(resultados_stacking_ln[barra][est_key]['prediccion_val'])
    fechas_val = pd.to_datetime(resultados_stacking_ln[barra]['_datos_split']['fechas_val'])
    errores    = np.abs(y_val - yhat_val)
    dias       = np.array([_dias_min(t) for t in fechas_val])

    grupos = [errores[dias <= 1], errores[(dias > 1) & (dias <= 7)], errores[dias > 7]]
    datos  = [g for g in grupos if len(g) > 5]
    labels = [l for g, l in zip(grupos, LABELS) if len(g) > 5]
    cols   = [c for g, c in zip(grupos, COLORES) if len(g) > 5]

    if not datos:
        ax.set_visible(False)
        continue

    vp = ax.violinplot(datos, showmedians=True, showextrema=False)
    for body, color in zip(vp['bodies'], cols):
        body.set_facecolor(color)
        body.set_alpha(0.65)
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(1.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Error absoluto (USD/MWh)', fontsize=8)
    ax.set_title(f'{barra}', fontsize=10, fontweight='bold')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribución de errores absolutos por grupo temporal — LN (validación)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()\
"""

# ── Celda 7: Paso 4 ──────────────────────────────────────────────────────────
CODE_PASO4 = """\
# ── Paso 4: Gráficos ±48h para casos del top-20 cercanos a cambio de hora ──
casos = df_top20_all[df_top20_all['en_ventana_7d']].copy()
print(f"Casos del top-20 dentro de ventana ±7d: {len(casos)} "
      f"(de {len(df_top20_all)} totales en {df_top20_all['Barra'].nunique()} barras)")
display(casos[['Barra', 'rank', 'timestamp', 'error_abs',
               'dias_al_cambio', 'cambio_cercano', 'tipo_cambio']])

for _, fila in casos.iterrows():
    barra      = fila['Barra']
    nombre_est = fila['Estrategia']
    est_key    = _get_key(resultados_stacking_ln, barra, nombre_est)
    t_error    = fila['timestamp']
    t_cambio   = fila['cambio_cercano']
    tipo       = fila['tipo_cambio']

    y_val      = np.array(resultados_stacking_ln[barra]['_datos_split']['y_real_val'])
    yhat_val   = np.array(resultados_stacking_ln[barra][est_key]['prediccion_val'])
    fechas_val = pd.to_datetime(resultados_stacking_ln[barra]['_datos_split']['fechas_val'])

    mask = ((fechas_val >= t_error - pd.Timedelta(hours=48)) &
            (fechas_val <= t_error + pd.Timedelta(hours=48)))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(fechas_val[mask], y_val[mask],    label='Real',       color='#2c3e50', lw=1.8)
    ax.plot(fechas_val[mask], yhat_val[mask], label='Predicción', color='#e74c3c', lw=1.8, ls='--')
    ax.axvline(t_error,  color='#f39c12', lw=2.5, ls=':',
               label=f'Error pico #{int(fila["rank"])} ({fila["error_abs"]:.1f} USD/MWh)')
    ax.axvline(t_cambio, color='#27ae60', lw=2.5, ls='--',
               label=f'Cambio {tipo} ({t_cambio.date()}, Δ{int(fila["dias_al_cambio"])}d)')

    ax.set_title(f'{barra} [{nombre_est}] — Ventana ±48h | Error #{int(fila["rank"])} '
                 f'| {t_error}', fontweight='bold')
    ax.set_xlabel('Fecha / Hora')
    ax.set_ylabel('CMg (USD/MWh)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()\
"""

# ── Leer notebook, eliminar últimas N celdas si ya existen, agregar corregidas ─
with open(NOTEBOOK_PATH, encoding='utf-8') as f:
    nb = json.load(f)

# Quitar las 7 celdas del run anterior si el total > 76
TOTAL_ORIGINAL = 76
if len(nb['cells']) > TOTAL_ORIGINAL:
    nb['cells'] = nb['cells'][:TOTAL_ORIGINAL]
    print(f"Celdas previas eliminadas. Quedan {TOTAL_ORIGINAL}.")

nuevas = [
    make_markdown(MD_TITULO),
    make_code(CODE_SETUP),
    make_code(CODE_PASO1),
    make_code(CODE_PASO2),
    make_code(CODE_PASO3A),
    make_code(CODE_PASO3B),
    make_code(CODE_PASO4),
]
nb['cells'].extend(nuevas)

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"OK: notebook actualizado con {len(nb['cells'])} celdas ({len(nuevas)} nuevas).")
