import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import norm


def _dm_desde_diferencia(d, h=1, max_lag=None):
    """Núcleo del test de Diebold-Mariano dada la diferencia de pérdidas d_t = L(e_a) - L(e_b),
    con varianza de largo plazo estimada vía Newey-West (Bartlett)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)

    if max_lag is None:
        max_lag = h + 1

    d_mean = float(np.mean(d))
    d_centrado = d - d_mean

    gamma_0 = np.mean(d_centrado ** 2)
    s = gamma_0
    for k in range(1, max_lag + 1):
        gamma_k = np.mean(d_centrado[k:] * d_centrado[:-k])
        peso = 1 - k / (max_lag + 1)  # ponderación de Bartlett
        s += 2 * peso * gamma_k

    var_d_mean = s / n
    dm_stat = d_mean / np.sqrt(var_d_mean)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return {'DM': float(dm_stat), 'p_value': float(p_value), 'd_mean': d_mean,
            'max_lag': max_lag, 'n': n}


def diebold_mariano(y_real, y_pred_a, y_pred_b, h=1, loss='mae', max_lag=None):
    """
    Test de Diebold-Mariano (1995) para comparar la exactitud predictiva de dos
    modelos sobre la misma serie, con varianza de largo plazo estimada vía
    Newey-West (Bartlett) para tolerar autocorrelación en la diferencia de
    pérdidas -- a diferencia de Wilcoxon, que asume observaciones independientes
    y no es válido sobre errores horarios serialmente correlacionados.

    Parámetros
    ----------
    y_real, y_pred_a, y_pred_b : array-like
        Serie real y las dos predicciones a comparar, mismo orden temporal.
    h : int
        Horizonte de pronóstico en pasos (por defecto 1, el usado en este trabajo).
    loss : 'mae' | 'mse'
        Función de pérdida sobre la que se calcula la diferencia d_t.
    max_lag : int | None
        Truncamiento de Newey-West. Si None, se usa h+1 (Diebold y Mariano, 1995,
        recomiendan al menos h-1; h+1 da un margen adicional).

    Retorna
    -------
    dict con 'DM' (estadístico), 'p_value' (dos colas), 'd_mean' (diferencia de
    pérdida promedio, negativo favorece a `y_pred_a`), 'max_lag' usado, 'n'.
    """
    y_real = np.asarray(y_real, dtype=float)
    a = np.asarray(y_pred_a, dtype=float)
    b = np.asarray(y_pred_b, dtype=float)

    mask = np.isfinite(y_real) & np.isfinite(a) & np.isfinite(b)
    y_real, a, b = y_real[mask], a[mask], b[mask]

    if loss == 'mae':
        d = np.abs(y_real - a) - np.abs(y_real - b)
    elif loss == 'mse':
        d = (y_real - a) ** 2 - (y_real - b) ** 2
    else:
        raise ValueError("loss debe ser 'mae' o 'mse'")

    return _dm_desde_diferencia(d, h=h, max_lag=max_lag)


def diebold_mariano_desde_errores(err_a, err_b, h=1, max_lag=None):
    """
    Variante de `diebold_mariano` para cuando ya se cuenta con las series de error
    absoluto (o cuadrático) de cada modelo por separado -- por ejemplo, al comparar
    predicciones que provienen de *pipelines* distintos (P+C vía Stacking vs.\ S+C
    directo) donde reconstruir un y_real común no es directo, pero ambas series de
    error ya están alineadas por fecha.
    """
    err_a = np.asarray(err_a, dtype=float)
    err_b = np.asarray(err_b, dtype=float)
    mask = np.isfinite(err_a) & np.isfinite(err_b)
    d = err_a[mask] - err_b[mask]
    return _dm_desde_diferencia(d, h=h, max_lag=max_lag)


def analisis_comparativo(
    metricas_ln,
    metricas_dyt,
    df_mejores_ln,
    df_mejores_dyt,
    tiempo_entrenamiento_ln=None,
    tiempo_entrenamiento_dyt=None,
    lista_barras=None,
    comparar='Test',
    guardar_resultados=True,
    mostrar_graficos=True,
    verbose=True):

    metricas_ln = metricas_ln[metricas_ln['Set'] == comparar].copy()
    metricas_dyt = metricas_dyt[metricas_dyt['Set'] == comparar].copy()

    if lista_barras is None:
        lista_barras = metricas_ln['Barra'].dropna().unique().tolist()

    resultados = {}

    # ── TABLA TRANSFORMERS PUROS ────────────────────────────────────────────────
    datos_transformers = []

    for barra in lista_barras:
        for modelo_tipo in ['PRECIOS', 'RESIDUOS']:
            ln_row = metricas_ln[(metricas_ln['Barra'] == barra) &
                                 (metricas_ln['Modelo'] == modelo_tipo)]
            dyt_row = metricas_dyt[(metricas_dyt['Barra'] == barra) &
                                   (metricas_dyt['Modelo'] == modelo_tipo)]

            if not ln_row.empty:
                datos_transformers.append({
                    'Barra': barra, 'Tipo': modelo_tipo, 'Arquitectura': 'LN',
                    'MAE':  ln_row['MAE'].values[0],
                    'RMSE': ln_row['RMSE'].values[0],
                    'R²':   ln_row['R²'].values[0]
                })
            if not dyt_row.empty:
                datos_transformers.append({
                    'Barra': barra, 'Tipo': modelo_tipo, 'Arquitectura': 'DyT',
                    'MAE':  dyt_row['MAE'].values[0],
                    'RMSE': dyt_row['RMSE'].values[0],
                    'R²':   dyt_row['R²'].values[0]
                })

    df_transformers = pd.DataFrame(datos_transformers)

    comparacion_transformers = []

    for barra in lista_barras:
        for modelo_tipo in ['PRECIOS', 'RESIDUOS']:
            ln_data  = df_transformers[(df_transformers['Barra'] == barra) &
                                       (df_transformers['Tipo'] == modelo_tipo) &
                                       (df_transformers['Arquitectura'] == 'LN')]
            dyt_data = df_transformers[(df_transformers['Barra'] == barra) &
                                       (df_transformers['Tipo'] == modelo_tipo) &
                                       (df_transformers['Arquitectura'] == 'DyT')]

            if not ln_data.empty and not dyt_data.empty:
                mae_ln   = ln_data['MAE'].values[0]
                mae_dyt  = dyt_data['MAE'].values[0]
                rmse_ln  = ln_data['RMSE'].values[0]
                rmse_dyt = dyt_data['RMSE'].values[0]
                r2_ln    = ln_data['R²'].values[0]
                r2_dyt   = dyt_data['R²'].values[0]

                comparacion_transformers.append({
                    'Barra':       barra,
                    'Tipo':        modelo_tipo,
                    'MAE_LN':      round(mae_ln,   3),
                    'MAE_DyT':     round(mae_dyt,  3),
                    'RMSE_LN':     round(rmse_ln,  3),
                    'RMSE_DyT':    round(rmse_dyt, 3),
                    'R²_LN':       round(r2_ln,    4),
                    'R²_DyT':      round(r2_dyt,   4),
                    'Ganador_MAE':  'DyT' if mae_dyt  < mae_ln  else 'LN' if mae_dyt  > mae_ln  else 'Empate',
                    'Ganador_RMSE': 'DyT' if rmse_dyt < rmse_ln else 'LN' if rmse_dyt > rmse_ln else 'Empate',
                    'Ganador_R²':   'DyT' if r2_dyt   > r2_ln   else 'LN' if r2_dyt   < r2_ln   else 'Empate',
                })

    df_comp_transformers = pd.DataFrame(comparacion_transformers)

    if verbose:
        print("TABLA COMPARATIVA: TRANSFORMERS PUROS")
        if df_comp_transformers.empty:
            print("  ADVERTENCIA: Sin datos para comparar. Verifique que metricas_ln y metricas_dyt "
                  "contengan datos para las mismas barras y tipos de modelo (PRECIOS/RESIDUOS).")
        else:
            _disp = df_comp_transformers.copy()
            _disp.loc[_disp['Barra'].duplicated(), 'Barra'] = ''
            display(_disp.reset_index(drop=True).style.hide(axis='index').format({
                'MAE_LN': '{:.3f}', 'MAE_DyT': '{:.3f}',
                'RMSE_LN': '{:.3f}', 'RMSE_DyT': '{:.3f}',
                'R²_LN': '{:.3f}', 'R²_DyT': '{:.3f}',
            }))

    resultados['transformers_puros'] = {
        'df_comparacion': df_comp_transformers,
        'df_completo':    df_transformers
    }

    # ── TABLA STACKING (formato largo) ─────────────────────────────────────────
    stacking_rows = []

    for barra in lista_barras:
        ln_mejor  = df_mejores_ln[df_mejores_ln['Barra']  == barra].iloc[0]
        dyt_mejor = df_mejores_dyt[df_mejores_dyt['Barra'] == barra].iloc[0]

        mae_ln_stack  = ln_mejor['MAE (Test)']
        mae_dyt_stack = dyt_mejor['MAE (Test)']
        r2_ln_stack   = ln_mejor['R² (Test)']
        r2_dyt_stack  = dyt_mejor['R² (Test)']

        # RMSE en stacking (opcional, según columnas disponibles)
        rmse_ln_stack  = ln_mejor['RMSE (Test)']  if 'RMSE (Test)' in ln_mejor.index  else np.nan
        rmse_dyt_stack = dyt_mejor['RMSE (Test)'] if 'RMSE (Test)' in dyt_mejor.index else np.nan

        mae_ln_puro  = df_comp_transformers[
            (df_comp_transformers['Barra'] == barra) &
            (df_comp_transformers['Tipo']  == 'PRECIOS')
        ]['MAE_LN'].values[0]
        mae_dyt_puro = df_comp_transformers[
            (df_comp_transformers['Barra'] == barra) &
            (df_comp_transformers['Tipo']  == 'PRECIOS')
        ]['MAE_DyT'].values[0]

        ganador_mae  = 'DyT' if mae_dyt_stack  < mae_ln_stack  else 'LN' if mae_dyt_stack  > mae_ln_stack  else 'Empate'
        ganador_rmse = ('DyT' if rmse_dyt_stack < rmse_ln_stack else 'LN' if rmse_dyt_stack > rmse_ln_stack else 'Empate') \
                       if not (np.isnan(rmse_ln_stack) or np.isnan(rmse_dyt_stack)) else np.nan
        ganador_r2   = 'DyT' if r2_dyt_stack   > r2_ln_stack   else 'LN' if r2_dyt_stack   < r2_ln_stack   else 'Empate'

        for arq, mae_stack, mae_puro, rmse_stack, r2_stack, estrategia in [
            ('LN',  mae_ln_stack,  mae_ln_puro,  rmse_ln_stack,  r2_ln_stack,  ln_mejor['Mejor Estrategia']),
            ('DyT', mae_dyt_stack, mae_dyt_puro, rmse_dyt_stack, r2_dyt_stack, dyt_mejor['Mejor Estrategia']),
        ]:
            mejora = ((mae_puro - mae_stack) / mae_puro * 100) if mae_puro != 0 else 0
            row = {
                'Barra':        barra,
                'Arquitectura': arq,
                'Estrategia':   estrategia,
                'MAE_Stacking': round(mae_stack, 3),
                'MAE_Puro':     round(mae_puro,  3),
                'Mejora_%':     round(mejora,    2),
                'R²_Stacking':  round(r2_stack,  4),
                'Ganador_MAE':  ganador_mae,
                'Ganador_R²':   ganador_r2,
            }
            if not np.isnan(rmse_stack):
                row['RMSE_Stacking'] = round(rmse_stack, 3)
                row['Ganador_RMSE']  = ganador_rmse
            stacking_rows.append(row)

    df_comp_stacking = pd.DataFrame(stacking_rows)

    if verbose:
        print("TABLA COMPARATIVA: MEJORES ESTRATEGIAS DE STACKING")
        if df_comp_stacking.empty:
            print("  ADVERTENCIA: Sin datos de stacking para mostrar.")
        else:
            _disp = df_comp_stacking.copy()
            _disp.loc[_disp['Barra'].duplicated(), 'Barra'] = ''
            _fmt = {c: '{:.3f}' for c in ['MAE_Stacking', 'MAE_Puro', 'Mejora_%',
                                           'R²_Stacking', 'RMSE_Stacking']
                    if c in _disp.columns}
            display(_disp.reset_index(drop=True).style.hide(axis='index').format(_fmt))

    resultados['stacking'] = {'df_comparacion': df_comp_stacking}

    # ── TIEMPOS DE ENTRENAMIENTO ────────────────────────────────────────────────
    if tiempo_entrenamiento_ln is not None and tiempo_entrenamiento_dyt is not None:

        if verbose:
            print("\n" + "─"*70)
            print("TIEMPOS DE ENTRENAMIENTO")
            print("─"*70 + "\n")

        comparacion_tiempos = []

        for barra in lista_barras:
            if barra in tiempo_entrenamiento_ln and barra in tiempo_entrenamiento_dyt:
                for tipo in ['Precios', 'Residuos']:
                    if tipo in tiempo_entrenamiento_ln[barra] and tipo in tiempo_entrenamiento_dyt[barra]:
                        tiempo_ln  = tiempo_entrenamiento_ln[barra][tipo]
                        tiempo_dyt = tiempo_entrenamiento_dyt[barra][tipo]
                        comparacion_tiempos.append({
                            'Barra':        barra,
                            'Tipo':         tipo,
                            'Tiempo_LN_min':  round(tiempo_ln,  2),
                            'Tiempo_DyT_min': round(tiempo_dyt, 2),
                            'Diff_min':       round(tiempo_dyt - tiempo_ln, 2),
                            'Diff_%':         round((tiempo_dyt - tiempo_ln) / tiempo_ln * 100, 2) if tiempo_ln != 0 else 0,
                            'Más_Rápido':     'LN' if tiempo_ln < tiempo_dyt else 'DyT' if tiempo_dyt < tiempo_ln else 'Igual'
                        })

        if comparacion_tiempos:
            df_comp_tiempos = pd.DataFrame(comparacion_tiempos)

            if verbose:
                print("Tabla Comparativa: Tiempos de Entrenamiento\n")
                _disp = df_comp_tiempos.copy()
                _disp.loc[_disp['Barra'].duplicated(), 'Barra'] = ''
                display(_disp.reset_index(drop=True).style.hide(axis='index'))

                tiempo_total_ln  = df_comp_tiempos['Tiempo_LN_min'].sum()
                tiempo_total_dyt = df_comp_tiempos['Tiempo_DyT_min'].sum()
                diff_total = tiempo_total_dyt - tiempo_total_ln

                print(f"\nResumen Tiempos:")
                print(f"  Tiempo total LN:  {tiempo_total_ln:.2f} min ({tiempo_total_ln/60:.2f} horas)")
                print(f"  Tiempo total DyT: {tiempo_total_dyt:.2f} min ({tiempo_total_dyt/60:.2f} horas)")
                print(f"  Diferencia:       {diff_total:+.2f} min ({diff_total/tiempo_total_ln*100:+.2f}%)")
                print(f"  Más rápido:       {df_comp_tiempos['Más_Rápido'].value_counts().to_dict()}\n")

            resultados['tiempos'] = {
                'df_comparacion':  df_comp_tiempos,
                'tiempo_total_ln':  df_comp_tiempos['Tiempo_LN_min'].sum(),
                'tiempo_total_dyt': df_comp_tiempos['Tiempo_DyT_min'].sum()
            }

    # ── GRÁFICO: desempeño por métrica ─────────────────────────────────────────
    metricas_plot = [('MAE', 'MAE (USD/MWh)'), ('RMSE', 'RMSE (USD/MWh)'), ('R²', 'R²')]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(lista_barras))
    width = 0.35

    for ax, (metric, ylabel) in zip(axes, metricas_plot):
        ln_vals  = df_transformers[
            (df_transformers['Tipo'] == 'PRECIOS') &
            (df_transformers['Arquitectura'] == 'LN')
        ].sort_values('Barra')[metric].values

        dyt_vals = df_transformers[
            (df_transformers['Tipo'] == 'PRECIOS') &
            (df_transformers['Arquitectura'] == 'DyT')
        ].sort_values('Barra')[metric].values

        bars_ln  = ax.bar(x - width/2, ln_vals,  width, label='TFT_LN',
                          color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars_dyt = ax.bar(x + width/2, dyt_vals, width, label='TFT_DyT',
                          color='coral',     alpha=0.8, edgecolor='black', linewidth=1.5)

        for bar in [*bars_ln, *bars_dyt]:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'Comparación {metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(lista_barras, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Desempeño Transformers Puros (Modelo de Precios): LN vs DyT',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if guardar_resultados:
        plt.savefig('comparacion_completa_LN_vs_DyT.png', dpi=300, bbox_inches='tight')
        if verbose:
            print("comparacion_completa_LN_vs_DyT.png")

    if mostrar_graficos:
        plt.show()
    else:
        plt.close()

    # ── GUARDAR TABLAS ──────────────────────────────────────────────────────────
    if guardar_resultados:
        df_comp_transformers.to_csv('comparacion_transformers_puros_LN_vs_DyT.csv', index=False)
        df_comp_stacking.to_csv('comparacion_stacking_LN_vs_DyT.csv', index=False)
        if 'tiempos' in resultados:
            resultados['tiempos']['df_comparacion'].to_csv('comparacion_tiempos_LN_vs_DyT.csv', index=False)
        if verbose:
            print("\nTablas guardadas:")
            print("   • comparacion_transformers_puros_LN_vs_DyT.csv")
            print("   • comparacion_stacking_LN_vs_DyT.csv")
            if 'tiempos' in resultados:
                print("   • comparacion_tiempos_LN_vs_DyT.csv")

    return resultados
