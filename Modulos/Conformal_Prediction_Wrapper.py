import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

import sys  
sys.path.append(r'C:\Users\56977\OneDrive\Escritorio\AdaptiveConformalPredictionsTimeSeries') # Path al repo de mzaffran
from enbpi.PI_class_EnbPI import prediction_interval

# ========================= CONFORMAL PREDICTION CLÁSICO ======================

# Función para calibrar conformal prediction
def split_cp_from_calibration(y_cal, yhat_cal, y_test, yhat_test, alpha=0.05):

    y_cal = np.asarray(y_cal, dtype=float)
    yhat_cal = np.asarray(yhat_cal, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    yhat_test = np.asarray(yhat_test, dtype=float)

    # limpiar NaNs calibración
    mask_cal = np.isfinite(y_cal) & np.isfinite(yhat_cal)
    y_cal = y_cal[mask_cal]
    yhat_cal = yhat_cal[mask_cal]

    # limpiar NaNs test
    mask_test = np.isfinite(y_test) & np.isfinite(yhat_test)
    y_test_clean = y_test[mask_test]
    yhat_test_clean = yhat_test[mask_test]

    if len(y_cal) == 0:
        return {"error": "No hay datos válidos en calibración (VAL)"}
    if len(y_test_clean) == 0:
        return {"error": "No hay datos válidos en test (TEST)"}

    scores = np.abs(y_cal - yhat_cal)

    n = len(scores)
    q_level = np.ceil((1 - alpha) * (n + 1)) / (n + 1)
    q = np.quantile(scores, q_level, method="higher")

    lower = yhat_test_clean - q
    upper = yhat_test_clean + q

    coverage = np.mean((y_test_clean >= lower) & (y_test_clean <= upper))
    diff_coverage = abs(coverage - (1 - alpha))
    width = np.mean(upper - lower)
    mae = np.mean(np.abs(y_test_clean - yhat_test_clean))
    rmse = np.sqrt(np.mean((y_test_clean - yhat_test_clean) ** 2))

    return {"lower": lower,
            "upper": upper,
            "coverage": float(coverage),
            "expected_coverage": 1 - alpha,
            'diff_coverage': float(diff_coverage),
            "width": float(width),
            'mae': float(mae),
            'rmse': float(rmse),
            "q": float(q),
            "y_test": y_test_clean,
            "y_pred": yhat_test_clean,
            "mask_test": mask_test}

# Función para correr conformal prediction solo en la mejor estrategia de cada barra
def cp_clasic_implementation(resultados_stacking,
                             df_mejores,
                             split_cp_fn,
                             alpha = 0.05,
                             split_key = "_datos_split",
                             y_val_key = "y_real_val",
                             y_test_key = "y_real_test",
                             pred_val_key = "prediccion_val",
                             pred_test_key = "prediccion_test",
                             mapping = 'LN',
                             verbose = True):

    # mapping: nombre bonito -> nombre en resultados_stacking
    if mapping == 'LN':
        mapping = {"Prophet Solo": "Prophet_Solo",
                   "TFT_LN Precios Solo": "TFT_Precios_Solo",
                   "Prophet+TFT_LN Precios": "Prophet_TFT_Precios",
                   "TFT Residuos Solo": "TFT_Residuos_Solo",
                   "Prophet + Residuos (Directo)": "Prophet_Residuos_Directo",
                   "Prophet + Residuos (Opt)": "Prophet_Residuos_Opt",
                   "(Prophet + Residuos) + TFT_LN Precios": "Prophet_Residuos_Precios"}
        
    elif mapping == 'DyT':
        mapping = {"Prophet Solo": "Prophet_Solo",
                   "TFT_DyT Precios Solo": "TFT_Precios_Solo",
                   "Prophet+TFT_DyT Precios": "Prophet_TFT_Precios",
                   "TFT_DyT Residuos Solo": "TFT_Residuos_Solo",
                   "Prophet + Residuos (Directo)": "Prophet_Residuos_Directo",
                   "Prophet + Residuos (Opt)": "Prophet_Residuos_Opt",
                   "(Prophet + Residuos) + TFT_DyT Precios": "Prophet_Residuos_Precios"}

    best_strategy_by_barra = (df_mejores.set_index("Barra")["Mejor Estrategia"].to_dict())
    cp_best = {}
    rows = []

    for barra, data in resultados_stacking.items():
        best_pretty = best_strategy_by_barra.get(barra)

        if best_pretty is None:
            if verbose:
                print(f"[SKIP] {barra}: no aparece en df_mejores_ln")
            continue

        # Traducir a nombre técnico
        best = mapping.get(best_pretty)

        if best is None:
            raise ValueError(f"{barra}: estrategia '{best_pretty}' no está en el mapping")

        if best not in data:
            raise ValueError(f"{barra}: '{best}' no existe en resultados_stacking[{barra}]")

        y_val = data[split_key][y_val_key]
        y_test = data[split_key][y_test_key]
        yhat_val = data[best][pred_val_key]
        yhat_test = data[best][pred_test_key]

        res = split_cp_fn(y_cal=y_val, yhat_cal=yhat_val, y_test=y_test, yhat_test=yhat_test, alpha=alpha)
        cp_best[barra] = {"Estrategia": best_pretty, **res}

        if verbose:
            if "error" in res:
                print(f"{barra:10s} | {best_pretty:30s} | ERROR: {res['error']}")
            else:
                print(f"{barra:10s} | {best_pretty:30s} | coverage={res['coverage']:.3f} | width={res['width']:.2f} | q={res.get('q', np.nan):.2f}")

        if "error" not in res:
            rows.append({
                "Barra": barra,
                "Estrategia": best_pretty,
                "Coverage": res["coverage"],
                "Expected": res.get("expected_coverage", 1 - alpha),
                "Coverage_Diff": res.get("diff_coverage", np.nan),
                "Width": res["width"],
                "MAE": res.get("mae", np.nan),
                "RMSE": res.get("rmse", np.nan),
                "q": res.get("q", np.nan),
                "N_test": len(res.get("y_test", [])),
            })

    df_cp_best = pd.DataFrame(rows).sort_values(["Barra"]).reset_index(drop=True)
    return cp_best, df_cp_best

# ========================= CONFORMAL PREDICTION ENBPI ======================

# Función para correr EnbPI usando un modelo base (RF, GB o Ridge) y todas las predicciones como features
def cp_enbpi_implementation(resultados_stacking, barras, estrategia_barras, modelo_base,  # 'RF', 'GB', o 'Ridge'
                            alpha=0.05,
                            B=30,
                            stride=1,
                            miss_test_idx=[]):
    
    tabla = []
    dict_cp = {}
    
    # Seleccionar modelo base
    if modelo_base == 'RF':
        fit_func = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    elif modelo_base == 'GB':
        fit_func = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

    else:
        fit_func = Ridge(alpha=1.0)
    
    # Scaler para normalización
    scaler = StandardScaler()

    for barra in barras:
        try:
            
            # Construir features: Usar todas las predicciones
            X_val = np.column_stack([
                resultados_stacking[barra]['Prophet_Solo']['prediccion_val'],
                resultados_stacking[barra]['TFT_Precios_Solo']['prediccion_val'],
                resultados_stacking[barra]['Prophet_TFT_Precios']['prediccion_val'],
                resultados_stacking[barra]['TFT_Residuos_Solo']['prediccion_val'],
                resultados_stacking[barra]['Prophet_Residuos_Opt']['prediccion_val']])
            y_val = resultados_stacking[barra]['_datos_split']['y_real_val']

            X_test = np.column_stack([
                resultados_stacking[barra]['Prophet_Solo']['prediccion_test'],
                resultados_stacking[barra]['TFT_Precios_Solo']['prediccion_test'],
                resultados_stacking[barra]['Prophet_TFT_Precios']['prediccion_test'],
                resultados_stacking[barra]['TFT_Residuos_Solo']['prediccion_test'],
                resultados_stacking[barra]['Prophet_Residuos_Opt']['prediccion_test']])
            y_test = np.asarray(resultados_stacking[barra]['_datos_split']['y_real_test'])

            # Estandarizar
            X_val_scaled = scaler.fit_transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            # Entrenar modelo base
            fit_func.fit(X_val_scaled, y_val)
            
            # Predicción puntual 
            y_pred_enbpi = fit_func.predict(X_test_scaled)

            # EnbPI: genera intervalos alrededor de y_pred
            pi = prediction_interval(fit_func, X_val_scaled, X_test_scaled, y_val, y_test)
            bandas = pi.compute_PIs_Ensemble_online(alpha=alpha, 
                                                    B=B, 
                                                    stride=stride, 
                                                    miss_test_idx=miss_test_idx)

            lower = np.asarray(bandas["lower"])
            upper = np.asarray(bandas["upper"])

            # Métricas
            n_test = len(y_test)
            coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
            width = float(np.mean(upper - lower))
            mae = float(np.mean(np.abs(y_test - y_pred_enbpi)))
            rmse = float(np.sqrt(np.mean((y_test - y_pred_enbpi) ** 2)))
            estrategia = estrategia_barras.get(barra, 'Unknown')

            tabla.append({
                "Barra": barra,
                "Estrategia": estrategia,
                "Coverage": coverage,
                "Expected": 1 - alpha,
                "Coverage_Diff": abs(coverage - (1 - alpha)),
                "Width": width,
                "MAE": mae,
                "RMSE": rmse,
                "N_test": n_test,
                "Modelo_Base": modelo_base
            })

            dict_cp[barra] = {
                "y_test": y_test,
                "y_pred": y_pred_enbpi,
                "lower": lower,
                "upper": upper,
                "Estrategia": estrategia,
                "coverage": coverage,
                "width": width,
                "mae": mae,
                "rmse": rmse
            }

        except Exception as e:
            print(f"  ERROR EN {barra}: {str(e)}")
            continue

    df_tabla = pd.DataFrame(tabla).sort_values("Barra").reset_index(drop=True)
    
    return df_tabla, dict_cp

# ========================= CONFORMAL PREDICTION ADAPTIVE OFFLINE ======================

# Función para correr ACP offline: adapta el nivel de confianza alpha_t en cada paso según el error observado
def acp_offline_implementation(resultados_stacking, barras, estrategia_barras,
                              modelo_base='Ridge',
                              alpha=0.05,
                              gamma=0.05):  # Velocidad de adaptación

    tabla = []
    dict_cp = {}
    dict_alphas = {}
    
    # Seleccionar modelo base
    if modelo_base == 'RF':
        fit_func = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        )
    else:  # Ridge
        fit_func = Ridge(alpha=1.0)
    
    scaler = StandardScaler()

    for barra in barras:
        try:
            
            # Construir features
            X_val = np.column_stack([
                resultados_stacking[barra]['Prophet_Solo']['prediccion_val'],
                resultados_stacking[barra]['TFT_Precios_Solo']['prediccion_val'],
                resultados_stacking[barra]['Prophet_TFT_Precios']['prediccion_val'],
                resultados_stacking[barra]['TFT_Residuos_Solo']['prediccion_val'],
                resultados_stacking[barra]['Prophet_Residuos_Opt']['prediccion_val']])
            y_val = resultados_stacking[barra]['_datos_split']['y_real_val']

            X_test = np.column_stack([
                resultados_stacking[barra]['Prophet_Solo']['prediccion_test'],
                resultados_stacking[barra]['TFT_Precios_Solo']['prediccion_test'],
                resultados_stacking[barra]['Prophet_TFT_Precios']['prediccion_test'],
                resultados_stacking[barra]['TFT_Residuos_Solo']['prediccion_test'],
                resultados_stacking[barra]['Prophet_Residuos_Opt']['prediccion_test']])
            y_test = np.asarray(resultados_stacking[barra]['_datos_split']['y_real_test'])

            # Estandarizar
            X_val_scaled = scaler.fit_transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            # Entrenar modelo base
            fit_func.fit(X_val_scaled, y_val)
            y_pred_test = fit_func.predict(X_test_scaled)

            # Residuos de calibración
            y_pred_cal = fit_func.predict(X_val_scaled)
            residuos_cal = np.abs(y_val - y_pred_cal)  # Residuos absolutos (non-conformity scores)

            # ACP (OFFLINE)
            n_test = len(y_test)
            lower = np.empty(n_test)
            upper = np.empty(n_test)
            alphas_t = np.empty(n_test)  # Evolución de alpha
            errors_t = np.empty(n_test)  # Errores observados
            
            alpha_t = alpha  # Inicializar
            
            for t in range(n_test):

                # PASO 1: Calcular quantil con alpha_t actual
                q_level = (1 - alpha_t) * (len(residuos_cal) + 1) / (len(residuos_cal))
                q_t = np.quantile(residuos_cal, np.minimum(q_level, 1.0), method='higher')
                
                # PASO 2: Generar intervalo para t
                lower[t] = y_pred_test[t] - q_t
                upper[t] = y_pred_test[t] + q_t
                
                # PASO 3: Observar y_test[t] y calcular error
                err_t = 1 - int((lower[t] <= y_test[t]) & (y_test[t] <= upper[t]))
                # err_t = 1 si y_test FUERA del intervalo
                # err_t = 0 si y_test DENTRO del intervalo
                
                errors_t[t] = err_t
                alphas_t[t] = alpha_t
                
                # PASO 4: Actualizar alpha para siguiente iteración 
                # Esta es la MAGIA de ACP: adapta dinámicamente
                alpha_t = alpha_t + gamma * (alpha - err_t)
                
                # Limitar alpha_t a rango válido [0, 1]
                alpha_t = np.clip(alpha_t, 0.001, 0.999)

            # Métricas
            coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
            width = float(np.mean(upper - lower))
            mae = float(np.mean(np.abs(y_test - y_pred_test)))
            rmse = float(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
            estrategia = estrategia_barras.get(barra, 'Unknown')
            
            # Estadísticas de adaptación
            alpha_mean = float(np.mean(alphas_t))
            alpha_std = float(np.std(alphas_t))
            error_rate = float(np.mean(errors_t))

            tabla.append({
                "Barra": barra,
                "Estrategia": estrategia,
                "Coverage": coverage,
                "Expected": 1 - alpha,
                "Coverage_Diff": abs(coverage - (1 - alpha)),
                "Width": width,
                "MAE": mae,
                "RMSE": rmse,
                "N_test": n_test,
                "Gamma": gamma,
                "Alpha_Mean": alpha_mean,
                "Alpha_Std": alpha_std,
                "Error_Rate": error_rate,
                "Modelo_Base": modelo_base
            })

            dict_cp[barra] = {
                "y_test": y_test,
                "y_pred": y_pred_test,
                "lower": lower,
                "upper": upper,
                "Estrategia": estrategia,
                "coverage": coverage,
                "width": width,
                "mae": mae,
                "rmse": rmse
            }

            dict_alphas[barra] = {
                "alphas_t": alphas_t,
                "errors_t": errors_t
            }

        except Exception as e:
            print(f"  ERROR EN {barra}: {str(e)}")
            continue

    df_tabla = pd.DataFrame(tabla).sort_values("Barra").reset_index(drop=True)
    
    return df_tabla, dict_cp, dict_alphas

# ========================= CONFORMAL PREDICTION ADAPTIVE ONLINE ======================

# Función 
def acp_online_implementation(resultados_stacking, barras, estrategia_barras,
                              modelo_base='Ridge',
                              alpha=0.05,
                              gamma=0.05):
    """
    ACP ONLINE PURO: Reentrenamiento cada paso (mzaffran style)
    
    Diferencias con ACP Offline:
    ✓ Modelo se retrain cada iteración
    ✓ Residuos se actualizan dinámicamente
    ✓ Máxima adaptación a cambios
    ✗ Más lento: O(n_test × train_time)
    
    Parameters:
    -----------
    gamma : float
        Velocidad de adaptación de alpha
        Recomendado: 0.05
    """
    
    tabla = []
    dict_cp = {}
    dict_alphas = {}
    tiempos_ejecucion = {}
    
    # Seleccionar modelo base
    if modelo_base == 'RF':
        def crear_modelo():
            return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        
    else: 
        def crear_modelo():
            return Ridge(alpha=1.0)
    
    scaler = StandardScaler()

    for barra in barras:
        try:
            tiempo_inicio = time.time()
            
            # Construir features
            X_val = np.column_stack([
                resultados_stacking[barra]['Prophet_Solo']['prediccion_val'],
                resultados_stacking[barra]['TFT_Precios_Solo']['prediccion_val'],
                resultados_stacking[barra]['Prophet_TFT_Precios']['prediccion_val'],
                resultados_stacking[barra]['TFT_Residuos_Solo']['prediccion_val'],
                resultados_stacking[barra]['Prophet_Residuos_Opt']['prediccion_val']])
            y_val = resultados_stacking[barra]['_datos_split']['y_real_val']

            X_test = np.column_stack([
                resultados_stacking[barra]['Prophet_Solo']['prediccion_test'],
                resultados_stacking[barra]['TFT_Precios_Solo']['prediccion_test'],
                resultados_stacking[barra]['Prophet_TFT_Precios']['prediccion_test'],
                resultados_stacking[barra]['TFT_Residuos_Solo']['prediccion_test'],
                resultados_stacking[barra]['Prophet_Residuos_Opt']['prediccion_test']])
            y_test = np.asarray(resultados_stacking[barra]['_datos_split']['y_real_test'])

            # Estandarización global 
            scaler.fit(X_val)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            # Tamaño de datos de entrenamiento
            train_size = len(X_val_scaled)
            n_test = len(y_test)
            
            # Combinar val + test para reentrenamiento progresivo
            X_combined = np.vstack([X_val_scaled, X_test_scaled])
            y_combined = np.hstack([y_val, y_test])

            # ACP Online puro
            lower = np.empty(n_test)
            upper = np.empty(n_test)
            alphas_t = np.empty(n_test)
            errors_t = np.empty(n_test)
            y_pred_test_online = np.empty(n_test)
            
            alpha_t = alpha
            
            for t in range(n_test):

                # PASO 1: RETRAIN modelo con datos hasta t 
                # Índices: [0:train_size] = validación + [train_size:train_size+t] = test hasta t
                idx_train = np.arange(train_size + t)
                
                modelo_t = crear_modelo()
                modelo_t.fit(X_combined[idx_train], y_combined[idx_train])
                
                # PASO 2: Predicción para t
                y_pred_t = modelo_t.predict(X_test_scaled[t:t+1])[0]
                y_pred_test_online[t] = y_pred_t
                
                # PASO 3: Calcular residuos de calibración HASTA t =====
                # Residuos en datos de entrenamiento (hasta t)
                y_pred_cal_t = modelo_t.predict(X_combined[idx_train])
                residuos_cal_t = np.abs(y_combined[idx_train] - y_pred_cal_t)
                
                # PASO 4: Calcular quantil con alpha_t actual =====
                q_level = (1 - alpha_t) * (len(residuos_cal_t) + 1) / (len(residuos_cal_t))
                q_t = np.quantile(residuos_cal_t, np.minimum(q_level, 1.0), method='higher')
                
                # PASO 5: Generar intervalo =====
                lower[t] = y_pred_t - q_t
                upper[t] = y_pred_t + q_t
                
                # PASO 6: Observar error y ADAPTAR alpha =====
                err_t = 1 - int((lower[t] <= y_test[t]) & (y_test[t] <= upper[t]))
                errors_t[t] = err_t
                alphas_t[t] = alpha_t
                
                # Actualizar alpha para siguiente iteración
                alpha_t = alpha_t + gamma * (alpha - err_t)
                alpha_t = np.clip(alpha_t, 0.001, 0.999)

            # Métricas
            coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
            width = float(np.mean(upper - lower))
            mae = float(np.mean(np.abs(y_test - y_pred_test_online)))
            rmse = float(np.sqrt(np.mean((y_test - y_pred_test_online) ** 2)))
            estrategia = estrategia_barras.get(barra, 'Unknown')
            
            alpha_mean = float(np.mean(alphas_t))
            alpha_std = float(np.std(alphas_t))
            error_rate = float(np.mean(errors_t))
            
            tiempo_total = time.time() - tiempo_inicio

            tabla.append({
                "Barra": barra,
                "Estrategia": estrategia,
                "Coverage": coverage,
                "Expected": 1 - alpha,
                "Coverage_Diff": abs(coverage - (1 - alpha)),
                "Width": width,
                "MAE": mae,
                "RMSE": rmse,
                "N_test": n_test,
                "Gamma": gamma,
                "Alpha_Mean": alpha_mean,
                "Alpha_Std": alpha_std,
                "Error_Rate": error_rate,
                "Modelo_Base": modelo_base,
                "Tiempo_Seg": tiempo_total
            })

            dict_cp[barra] = {
                "y_test": y_test,
                "y_pred": y_pred_test_online,
                "lower": lower,
                "upper": upper,
                "Estrategia": estrategia,
                "coverage": coverage,
                "width": width,
                "mae": mae,
                "rmse": rmse
            }

            dict_alphas[barra] = {
                "alphas_t": alphas_t,
                "errors_t": errors_t
            }

            tiempos_ejecucion[barra] = tiempo_total

        except Exception as e:
            print(f"  ERROR EN {barra}: {str(e)}")
            traceback.print_exc()
            continue

    df_tabla = pd.DataFrame(tabla).sort_values("Barra").reset_index(drop=True)
    
    return df_tabla, dict_cp, dict_alphas, tiempos_ejecucion

# ========================= Funciones extras ======================

# Función para graficar los intervalos de conformal prediction de LN y DyT
def plot_cp(lista_barras, cp_ln, cp_dyt, resultados_stacking_ln, resultados_stacking_dyt,
            tipo_cp='clásico', zoom_dias=45, guardar_graficos=False, mostrar_graficos=True):

    N_INSET = 14 * 24  # 2 semanas en horas para el inset

    def _plot_full(ax, fechas, y_true, y_pred, lower, upper, estrategia, res, arq, color_pred, color_band):
        ax.plot(fechas, y_true,  label='Precio Real',    lw=2,   color='black',     alpha=0.9)
        ax.plot(fechas, y_pred,  label=f'{estrategia}',  lw=2,   color=color_pred,  alpha=0.8)
        ax.fill_between(fechas, lower, upper,
                        color=color_band, alpha=0.3,
                        label=f'Banda CP {tipo_cp} {arq}  (Width={res["width"]:.1f})')
        ax.set_ylabel('Precio (USD/MWh)', fontsize=11)
        ax.set_title(f'{ax.get_title()}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # Inset: últimas 2 semanas
        n_in = min(N_INSET, len(fechas))
        ax_in = ax.inset_axes([0.62, 0.55, 0.36, 0.38])
        ax_in.plot(fechas[-n_in:], y_true[-n_in:],  color='black',     linewidth=1.5, alpha=0.9)
        ax_in.plot(fechas[-n_in:], y_pred[-n_in:],  color=color_pred,  linewidth=1.5, alpha=0.8)
        ax_in.fill_between(fechas[-n_in:], lower[-n_in:], upper[-n_in:],
                           color=color_band, alpha=0.3)
        ax_in.tick_params(labelsize=7)
        ax_in.tick_params(axis='x', rotation=30)
        ax_in.grid(True, alpha=0.3, linewidth=0.5)
        ax.indicate_inset_zoom(ax_in, edgecolor='0.4', linewidth=1.2)

    def _plot_zoom(ax, fechas, y_true, y_pred, lower, upper, estrategia, res, arq, color_pred, color_band, zoom_dias):
        n_zoom = min(zoom_dias * 24, len(fechas))
        f_zoom = fechas[-n_zoom:]
        y_zoom = y_true[-n_zoom:]
        p_zoom = y_pred[-n_zoom:]
        l_zoom = lower[-n_zoom:]
        u_zoom = upper[-n_zoom:]

        ax.plot(f_zoom, y_zoom, label='Precio Real',   lw=2.5, color='black',    alpha=0.9, zorder=5)
        ax.plot(f_zoom, p_zoom,
                label=f'{estrategia}  (Coverage={res["coverage"]:.2f}, Width={res["width"]:.1f})',
                lw=2.5, color=color_pred, alpha=0.9)
        ax.fill_between(f_zoom, l_zoom, u_zoom, color=color_band, alpha=0.3,
                        label=f'Banda CP {tipo_cp} {arq}')

        fecha_ini = pd.Timestamp(f_zoom[0]).strftime('%Y-%m-%d')
        fecha_fin = pd.Timestamp(f_zoom[-1]).strftime('%Y-%m-%d')
        ax.set_title(
            f'ZOOM {arq} — Últimos {zoom_dias} días\n{fecha_ini} → {fecha_fin}',
            fontsize=13, fontweight='bold', pad=10
        )
        ax.set_xlabel('Fecha', fontsize=11)
        ax.set_ylabel('Precio (USD/MWh)', fontsize=11)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    for barra in lista_barras:
        fig, axes = plt.subplots(4, 1, figsize=(18, 24))
        graficado = [False, False]

        # ── LN completo (ax 0) + zoom (ax 1) ───────────────────────────────────
        try:
            res_ln      = cp_ln[barra]
            y_true      = np.asarray(res_ln['y_test'])
            y_pred_ln   = np.asarray(res_ln['y_pred'])
            lower_ln    = np.asarray(res_ln['lower'])
            upper_ln    = np.asarray(res_ln['upper'])
            estrategia_ln = res_ln.get('Estrategia', 'Estrategia desconocida')
            fechas      = np.asarray(resultados_stacking_ln[barra]['_datos_split']['fechas_test'])

            axes[0].set_title(f'{barra} - LN | Coverage={res_ln["coverage"]:.2f}', fontsize=14, fontweight='bold')
            _plot_full(axes[0], fechas, y_true, y_pred_ln, lower_ln, upper_ln,
                       estrategia_ln, res_ln, 'LN', 'blue', 'orange')
            _plot_zoom(axes[1], fechas, y_true, y_pred_ln, lower_ln, upper_ln,
                       estrategia_ln, res_ln, 'LN', 'blue', 'orange', zoom_dias)
            graficado[0] = True
        except Exception as e:
            for i in [0, 1]:
                axes[i].set_title(f'{barra} - LN: Sin datos', fontsize=14)
                axes[i].text(0.5, 0.5, 'Sin datos disponibles', ha='center', va='center',
                             fontsize=16, transform=axes[i].transAxes)
                axes[i].axis('off')
            print(f'[WARN] No CP LN para {barra}: {e}')

        # ── DyT completo (ax 2) + zoom (ax 3) ──────────────────────────────────
        try:
            res_dyt       = cp_dyt[barra]
            y_true_dyt    = np.asarray(res_dyt['y_test'] if 'y_test' in res_dyt else y_true)
            y_pred_dyt    = np.asarray(res_dyt['y_pred'])
            lower_dyt     = np.asarray(res_dyt['lower'])
            upper_dyt     = np.asarray(res_dyt['upper'])
            estrategia_dyt = res_dyt.get('Estrategia', 'Estrategia desconocida')
            fechas_dyt    = np.asarray(
                resultados_stacking_dyt[barra]['_datos_split']['fechas_test']
                if barra in resultados_stacking_dyt else fechas
            )

            axes[2].set_title(f'{barra} - DyT | Coverage={res_dyt["coverage"]:.2f}', fontsize=14, fontweight='bold')
            _plot_full(axes[2], fechas_dyt, y_true_dyt, y_pred_dyt, lower_dyt, upper_dyt,
                       estrategia_dyt, res_dyt, 'DyT', 'red', 'cyan')
            _plot_zoom(axes[3], fechas_dyt, y_true_dyt, y_pred_dyt, lower_dyt, upper_dyt,
                       estrategia_dyt, res_dyt, 'DyT', 'red', 'cyan', zoom_dias)
            graficado[1] = True
        except Exception as e:
            for i in [2, 3]:
                axes[i].set_title(f'{barra} - DyT: Sin datos', fontsize=14)
                axes[i].text(0.5, 0.5, 'Sin datos disponibles', ha='center', va='center',
                             fontsize=16, transform=axes[i].transAxes)
                axes[i].axis('off')
            print(f'[WARN] No CP DyT para {barra}: {e}')

        if any(graficado):
            fig.suptitle(f'Intervalos Conformales {tipo_cp.capitalize()} — {barra}',
                         fontsize=18, fontweight='bold')
            plt.tight_layout(rect=[0, 0.01, 1, 0.97])

            if guardar_graficos:
                fname = f'cp_{tipo_cp}_{barra}.png'
                plt.savefig(fname, dpi=300, bbox_inches='tight')
                print(f'Guardado: {fname}')

            if mostrar_graficos:
                plt.show()
            else:
                plt.close(fig)
        else:
            plt.close(fig)
            print(f'[INFO] Ningún resultado para barra {barra}, gráfico no mostrado.')