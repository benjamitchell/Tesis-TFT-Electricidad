import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, QuantileRegressor

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

    lower = np.maximum(yhat_test_clean - q, 0.0)
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
                lower[t] = max(y_pred_test[t] - q_t, 0.0)
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
                lower[t] = max(y_pred_t - q_t, 0.0)
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

# ========================= CONFORMAL PREDICTION LOCALLY WEIGHTED SPLIT CP ======================

def _get_mapping(mapping):
    if mapping == 'LN':
        return {"Prophet Solo": "Prophet_Solo",
                "TFT_LN Precios Solo": "TFT_Precios_Solo",
                "Prophet+TFT_LN Precios": "Prophet_TFT_Precios",
                "TFT Residuos Solo": "TFT_Residuos_Solo",
                "Prophet + Residuos (Directo)": "Prophet_Residuos_Directo",
                "Prophet + Residuos (Opt)": "Prophet_Residuos_Opt",
                "(Prophet + Residuos) + TFT_LN Precios": "Prophet_Residuos_Precios"}
    if mapping == 'DyT':
        return {"Prophet Solo": "Prophet_Solo",
                "TFT_DyT Precios Solo": "TFT_Precios_Solo",
                "Prophet+TFT_DyT Precios": "Prophet_TFT_Precios",
                "TFT_DyT Residuos Solo": "TFT_Residuos_Solo",
                "Prophet + Residuos (Directo)": "Prophet_Residuos_Directo",
                "Prophet + Residuos (Opt)": "Prophet_Residuos_Opt",
                "(Prophet + Residuos) + TFT_DyT Precios": "Prophet_Residuos_Precios"}
    return mapping  # ya es dict


def _estimar_sigma_h(errores_cal, horas_cal, min_muestras=30, eps=1.0):
    """
    sigma_h[h] nunca baja de eps*sigma_global (default eps=1.0: nunca por debajo de la
    varianza global de calibración). Sin este piso, en horas de variabilidad muy baja
    (ej. mediodía/madrugada en barras con curtailment) un outlier puntual normalizado
    por un sigma_h casi nulo infla el cuantil q para TODAS las horas, no solo esa --
    verificado en CHARRUA/QUILLOTA, donde LW Split CP/LW-ACP quedaban peor que Split CP
    plano sin este piso.
    """
    sigma_global = max(float(np.std(errores_cal)), 1e-6)
    piso = eps * sigma_global
    sigma_h = np.empty(24)
    for h in range(24):
        mask_h = horas_cal == h
        if mask_h.sum() >= min_muestras:
            s = float(np.std(errores_cal[mask_h]))
            sigma_h[h] = max(s, piso) if s > 1e-6 else sigma_global
        else:
            sigma_h[h] = sigma_global
    return sigma_h


def aci_from_calibration(y_cal, yhat_cal, y_test, yhat_test, alpha=0.05, gamma=0.05):
    """
    ACI (Gibbs y Candès, 2021) sobre arrays crudos: igual que split_cp_from_calibration
    (score de no conformidad |y - yhat| sobre calibración/VAL) pero con adaptación online
    del nivel de cobertura alpha_t sobre el test, sin meta-modelo ni ponderación horaria.
    Es la version "ACP" del linaje SCP -> ACP -> LW-ACP: mismo yhat crudo que Split CP y
    LW-ACP, para que el ancho de intervalo entre métodos sea comparable (Sección 4.5.7).
    Al no reajustar ningún modelo en cada paso -- solo el escalar alpha_t -- no existe una
    distinción "offline"/"online" como en la formulación con meta-modelo Ridge: es un único
    método.
    """
    y_cal     = np.asarray(y_cal,     dtype=float)
    yhat_cal  = np.asarray(yhat_cal,  dtype=float)
    y_test    = np.asarray(y_test,    dtype=float)
    yhat_test = np.asarray(yhat_test, dtype=float)

    mask_cal  = np.isfinite(y_cal)  & np.isfinite(yhat_cal)
    mask_test = np.isfinite(y_test) & np.isfinite(yhat_test)
    y_cal, yhat_cal = y_cal[mask_cal], yhat_cal[mask_cal]
    y_test, yhat_test = y_test[mask_test], yhat_test[mask_test]

    scores_cal = np.abs(y_cal - yhat_cal)
    n_cal = len(scores_cal)

    n_test   = len(y_test)
    lower    = np.empty(n_test)
    upper    = np.empty(n_test)
    alphas_t = np.empty(n_test)
    errors_t = np.empty(n_test)
    alpha_t  = alpha

    for t in range(n_test):
        q_level  = (1 - alpha_t) * (n_cal + 1) / n_cal
        q_t      = np.quantile(scores_cal, min(q_level, 1.0), method='higher')
        lower[t] = max(yhat_test[t] - q_t, 0.0)
        upper[t] = yhat_test[t] + q_t
        err_t    = 1 - int((lower[t] <= y_test[t]) & (y_test[t] <= upper[t]))
        errors_t[t] = err_t
        alphas_t[t] = alpha_t
        alpha_t  = np.clip(alpha_t + gamma * (alpha - err_t), 0.001, 0.999)

    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    width    = float(np.mean(upper - lower))
    mae      = float(np.mean(np.abs(y_test - yhat_test)))
    rmse     = float(np.sqrt(np.mean((y_test - yhat_test) ** 2)))
    alpha_mean = float(np.mean(alphas_t))

    return {"lower": lower, "upper": upper,
            "coverage": coverage, "expected_coverage": 1 - alpha,
            "diff_coverage": float(abs(coverage - (1 - alpha))),
            "width": width, "mae": mae, "rmse": rmse,
            "alpha_mean": alpha_mean, "alphas_t": alphas_t, "errors_t": errors_t,
            "y_test": y_test, "y_pred": yhat_test}


def lw_split_cp_from_calibration(y_cal, yhat_cal, fechas_cal,
                                  y_test, yhat_test, fechas_test,
                                  alpha=0.05, min_muestras_por_hora=30):
    y_cal     = np.asarray(y_cal,     dtype=float)
    yhat_cal  = np.asarray(yhat_cal,  dtype=float)
    y_test    = np.asarray(y_test,    dtype=float)
    yhat_test = np.asarray(yhat_test, dtype=float)

    horas_cal  = pd.DatetimeIndex(fechas_cal).hour
    horas_test = pd.DatetimeIndex(fechas_test).hour

    mask_cal  = np.isfinite(y_cal)  & np.isfinite(yhat_cal)
    mask_test = np.isfinite(y_test) & np.isfinite(yhat_test)
    y_cal,    yhat_cal,  horas_cal  = y_cal[mask_cal],   yhat_cal[mask_cal],  horas_cal[mask_cal]
    y_test_c, yhat_tc,   horas_tc   = y_test[mask_test], yhat_test[mask_test], horas_test[mask_test]

    if len(y_cal) == 0:
        return {"error": "No hay datos válidos en calibración (VAL)"}
    if len(y_test_c) == 0:
        return {"error": "No hay datos válidos en test (TEST)"}

    errores_cal = np.abs(y_cal - yhat_cal)
    sigma_h     = _estimar_sigma_h(errores_cal, horas_cal, min_muestras_por_hora)
    scores_norm = errores_cal / sigma_h[horas_cal]

    n       = len(scores_norm)
    q_level = np.ceil((1 - alpha) * (n + 1)) / (n + 1)
    q       = np.quantile(scores_norm, q_level, method="higher")

    sigma_tc = sigma_h[horas_tc]
    lower    = np.maximum(yhat_tc - q * sigma_tc, 0.0)
    upper    = yhat_tc + q * sigma_tc

    coverage = float(np.mean((y_test_c >= lower) & (y_test_c <= upper)))
    width    = float(np.mean(upper - lower))
    mae      = float(np.mean(np.abs(y_test_c - yhat_tc)))
    rmse     = float(np.sqrt(np.mean((y_test_c - yhat_tc) ** 2)))

    return {"lower": lower, "upper": upper,
            "coverage": coverage, "expected_coverage": 1 - alpha,
            "diff_coverage": float(abs(coverage - (1 - alpha))),
            "width": width, "mae": mae, "rmse": rmse,
            "q": float(q), "sigma_h": sigma_h,
            "y_test": y_test_c, "y_pred": yhat_tc,
            "horas_test": np.asarray(horas_tc),
            "mask_test": mask_test}


def lw_acp_online_from_calibration(y_cal, yhat_cal, fechas_cal,
                                    y_test, yhat_test, fechas_test,
                                    alpha=0.05, gamma=0.05, min_muestras_por_hora=30):
    """LW-ACP sobre arrays crudos: igual que lw_split_cp_from_calibration (score
    normalizado por sigma_h de calibración/VAL) pero con adaptación online del nivel
    alpha_t sobre el test, como en aci_offline_solar_implementation. Combina la
    ponderación horaria con la adaptación online (gamma fijo, sin barrer/seleccionar
    sobre test)."""
    y_cal     = np.asarray(y_cal,     dtype=float)
    yhat_cal  = np.asarray(yhat_cal,  dtype=float)
    y_test    = np.asarray(y_test,    dtype=float)
    yhat_test = np.asarray(yhat_test, dtype=float)

    horas_cal  = pd.DatetimeIndex(fechas_cal).hour
    horas_test = pd.DatetimeIndex(fechas_test).hour

    mask_cal  = np.isfinite(y_cal)  & np.isfinite(yhat_cal)
    mask_test = np.isfinite(y_test) & np.isfinite(yhat_test)
    y_cal, yhat_cal, horas_cal = y_cal[mask_cal], yhat_cal[mask_cal], horas_cal[mask_cal]
    y_test, yhat_test, horas_test = y_test[mask_test], yhat_test[mask_test], horas_test[mask_test]

    errores_cal = np.abs(y_cal - yhat_cal)
    sigma_h     = _estimar_sigma_h(errores_cal, horas_cal, min_muestras_por_hora)
    scores_cal  = errores_cal / sigma_h[horas_cal]

    n_test   = len(y_test)
    lower    = np.empty(n_test)
    upper    = np.empty(n_test)
    alphas_t = np.empty(n_test)
    errors_t = np.empty(n_test)
    alpha_t  = alpha

    n_cal = len(scores_cal)
    for t in range(n_test):
        q_level  = (1 - alpha_t) * (n_cal + 1) / n_cal
        q_t      = np.quantile(scores_cal, min(q_level, 1.0), method='higher')
        h_t      = horas_test[t]
        lower[t] = max(yhat_test[t] - q_t * sigma_h[h_t], 0.0)
        upper[t] = yhat_test[t] + q_t * sigma_h[h_t]
        err_t    = 1 - int((lower[t] <= y_test[t]) & (y_test[t] <= upper[t]))
        errors_t[t] = err_t
        alphas_t[t] = alpha_t
        alpha_t  = np.clip(alpha_t + gamma * (alpha - err_t), 0.001, 0.999)

    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    width    = float(np.mean(upper - lower))
    mae      = float(np.mean(np.abs(y_test - yhat_test)))
    rmse     = float(np.sqrt(np.mean((y_test - yhat_test) ** 2)))
    alpha_mean = float(np.mean(alphas_t))

    return {"lower": lower, "upper": upper,
            "coverage": coverage, "expected_coverage": 1 - alpha,
            "diff_coverage": float(abs(coverage - (1 - alpha))),
            "width": width, "mae": mae, "rmse": rmse,
            "alpha_mean": alpha_mean, "alphas_t": alphas_t, "errors_t": errors_t,
            "sigma_h": sigma_h, "y_test": y_test, "y_pred": yhat_test,
            "horas_test": np.asarray(horas_test)}


def lw_split_cp_implementation(resultados_stacking, df_mejores,
                                alpha=0.05, min_muestras_por_hora=30,
                                split_key="_datos_split",
                                y_val_key="y_real_val", y_test_key="y_real_test",
                                fechas_val_key="fechas_val", fechas_test_key="fechas_test",
                                pred_val_key="prediccion_val", pred_test_key="prediccion_test",
                                mapping='LN', verbose=True):

    mapping = _get_mapping(mapping)
    best_by_barra = df_mejores.set_index("Barra")["Mejor Estrategia"].to_dict()
    cp_best = {}
    rows    = []

    for barra, data in resultados_stacking.items():
        best_pretty = best_by_barra.get(barra)
        if best_pretty is None:
            if verbose:
                print(f"[SKIP] {barra}: no aparece en df_mejores")
            continue

        best = mapping.get(best_pretty)
        if best is None:
            raise ValueError(f"{barra}: estrategia '{best_pretty}' no está en el mapping")
        if best not in data:
            raise ValueError(f"{barra}: '{best}' no existe en resultados_stacking")

        res = lw_split_cp_from_calibration(
            y_cal=data[split_key][y_val_key],
            yhat_cal=data[best][pred_val_key],
            fechas_cal=data[split_key][fechas_val_key],
            y_test=data[split_key][y_test_key],
            yhat_test=data[best][pred_test_key],
            fechas_test=data[split_key][fechas_test_key],
            alpha=alpha,
            min_muestras_por_hora=min_muestras_por_hora,
        )
        cp_best[barra] = {"Estrategia": best_pretty, **res}

        if verbose:
            if "error" in res:
                print(f"{barra:10s} | {best_pretty:30s} | ERROR: {res['error']}")
            else:
                picos = ", ".join(f"h{h}:{res['sigma_h'][h]:.2f}" for h in [7, 8, 19, 20, 21])
                print(f"{barra:10s} | {best_pretty:30s} | coverage={res['coverage']:.3f} | "
                      f"width={res['width']:.2f} | q={res['q']:.4f} | sigma[7,8,19-21]={picos}")

        if "error" not in res:
            rows.append({"Barra": barra, "Estrategia": best_pretty,
                         "Coverage": res["coverage"],
                         "Expected": res["expected_coverage"],
                         "Coverage_Diff": res["diff_coverage"],
                         "Width": res["width"],
                         "MAE": res["mae"], "RMSE": res["rmse"],
                         "q": res["q"], "N_test": len(res["y_test"])})

    df_cp = pd.DataFrame(rows).sort_values("Barra").reset_index(drop=True)
    return cp_best, df_cp


# ========================= CONFORMAL PREDICTION LW-ACP ONLINE ======================

def lw_acp_online_implementation(resultados_stacking, df_mejores,
                                  alpha=0.05, gamma=0.01,
                                  min_muestras_por_hora=30,
                                  split_key="_datos_split",
                                  y_val_key="y_real_val", y_test_key="y_real_test",
                                  fechas_val_key="fechas_val", fechas_test_key="fechas_test",
                                  pred_val_key="prediccion_val", pred_test_key="prediccion_test",
                                  mapping='LN', verbose=True):

    mapping   = _get_mapping(mapping)
    best_by_barra = df_mejores.set_index("Barra")["Mejor Estrategia"].to_dict()
    tabla      = []
    dict_cp    = {}
    dict_alphas = {}

    for barra, data in resultados_stacking.items():
        try:
            best_pretty = best_by_barra.get(barra)
            if best_pretty is None:
                if verbose:
                    print(f"[SKIP] {barra}: no aparece en df_mejores")
                continue

            best = mapping.get(best_pretty)
            if best is None:
                raise ValueError(f"estrategia '{best_pretty}' no está en el mapping")
            if best not in data:
                raise ValueError(f"'{best}' no existe en resultados_stacking[{barra}]")

            y_val       = np.asarray(data[split_key][y_val_key],  dtype=float)
            y_test      = np.asarray(data[split_key][y_test_key], dtype=float)
            fechas_val  = data[split_key][fechas_val_key]
            fechas_test = data[split_key][fechas_test_key]
            yhat_val    = np.asarray(data[best][pred_val_key],  dtype=float)
            yhat_test   = np.asarray(data[best][pred_test_key], dtype=float)

            horas_val  = pd.DatetimeIndex(fechas_val).hour
            horas_test = pd.DatetimeIndex(fechas_test).hour

            # Estimar sigma_h desde calibración (VAL) — sin data leakage
            errores_cal = np.abs(y_val - yhat_val)
            sigma_h     = _estimar_sigma_h(errores_cal, horas_val, min_muestras_por_hora)
            scores_cal  = errores_cal / sigma_h[horas_val]

            # Loop ACP Online sobre score normalizado
            n_test   = len(y_test)
            lower    = np.empty(n_test)
            upper    = np.empty(n_test)
            alphas_t = np.empty(n_test)
            errors_t = np.empty(n_test)
            alpha_t  = alpha

            n_cal = len(scores_cal)
            for t in range(n_test):
                q_level    = (1 - alpha_t) * (n_cal + 1) / n_cal
                q_t        = np.quantile(scores_cal, min(q_level, 1.0), method='higher')
                h_t        = horas_test[t]
                lower[t]   = max(yhat_test[t] - q_t * sigma_h[h_t], 0.0)
                upper[t]   = yhat_test[t] + q_t * sigma_h[h_t]
                err_t      = 1 - int((lower[t] <= y_test[t]) & (y_test[t] <= upper[t]))
                errors_t[t] = err_t
                alphas_t[t] = alpha_t
                alpha_t    = np.clip(alpha_t + gamma * (alpha - err_t), 0.001, 0.999)

            coverage   = float(np.mean((y_test >= lower) & (y_test <= upper)))
            width      = float(np.mean(upper - lower))
            mae        = float(np.mean(np.abs(y_test - yhat_test)))
            rmse       = float(np.sqrt(np.mean((y_test - yhat_test) ** 2)))
            alpha_mean = float(np.mean(alphas_t))
            alpha_std  = float(np.std(alphas_t))
            error_rate = float(np.mean(errors_t))

            if verbose:
                print(f"{barra:10s} | {best_pretty:30s} | coverage={coverage:.3f} | "
                      f"width={width:.2f} | alpha_mean={alpha_mean:.4f}")

            tabla.append({"Barra": barra, "Estrategia": best_pretty,
                          "Coverage": coverage, "Expected": 1 - alpha,
                          "Coverage_Diff": abs(coverage - (1 - alpha)),
                          "Width": width, "MAE": mae, "RMSE": rmse,
                          "N_test": n_test, "Gamma": gamma,
                          "Alpha_Mean": alpha_mean, "Alpha_Std": alpha_std,
                          "Error_Rate": error_rate})

            dict_cp[barra] = {"y_test": y_test, "y_pred": yhat_test,
                              "lower": lower, "upper": upper,
                              "Estrategia": best_pretty,
                              "coverage": coverage, "width": width,
                              "mae": mae, "rmse": rmse,
                              "sigma_h": sigma_h,
                              "horas_test": np.asarray(horas_test)}

            dict_alphas[barra] = {"alphas_t": alphas_t, "errors_t": errors_t}

        except Exception as e:
            print(f"  ERROR EN {barra}: {str(e)}")
            traceback.print_exc()
            continue

    df_tabla = pd.DataFrame(tabla).sort_values("Barra").reset_index(drop=True)
    return df_tabla, dict_cp, dict_alphas


# ========================= ACI OFFLINE / ONLINE SOBRE S+C (Ridge + posición solar) ======================

# Estas dos variantes reemplazan, para S+C, al meta-modelo Ridge de 5 predicciones de
# Stacking (P+C) usado por acp_offline_implementation/acp_online_implementation: en vez de
# combinar múltiples predicciones candidatas, el Ridge combina la predicción única de S+C
# con las dos variables solares dominantes según el análisis de interpretabilidad
# (elevacion_solar, cos_elevacion), ya que S+C no pasa por Stacking Optimization.

COORDENADAS_BARRAS_SC = {
    'ATACAMA':  (-28.57617, -70.75938),
    'CARDONES': (-27.36737, -70.33219),
    'CHARRUA':  (-36.82699, -73.04977),
    'CRUCERO':  (-23.65094, -70.39752),
    'P.AZUCAR': (-29.90591, -71.25014),
    'P.MONTT':  (-41.4693,  -72.94237),
    'QUILLOTA': (-33.036,   -71.62963),
    'TARAPACA': (-20.21326, -70.15027),
}


def _features_solares(fechas, lat, lon):
    """elevacion_solar y cos_elevacion vía pvlib, para un arreglo de fechas locales (America/Santiago)."""
    import pvlib
    ds_local = pd.to_datetime(fechas)
    ds_utc = ds_local.tz_localize(
        'America/Santiago', ambiguous='NaT', nonexistent='shift_forward'
    ).tz_convert('UTC')
    if ds_utc.isna().any():
        ds_utc = pd.DatetimeIndex(pd.Series(ds_utc).ffill() + pd.Timedelta(hours=1))
    sol = pvlib.solarposition.get_solarposition(ds_utc, lat, lon)
    elevacion = sol['elevation'].values
    return elevacion, np.cos(np.radians(elevacion))


def construir_resultados_stacking_sc(datos_eval, barras):
    """Shim minimo con la forma de resultados_stacking[barra]['_datos_split'] (fechas/y_real
    val y test), construido a partir de un dict eval_TFT_{arq}.pkl, para poder reusar
    plot_cp/plot_cp_por_hora (pensados para el pkl de Stacking P+C) con resultados de S+C."""
    shim = {}
    for barra in barras:
        val = datos_eval[barra]['val']['Precios']['datos']
        test = datos_eval[barra]['test']['Precios']['datos']
        orden_val = np.argsort(np.asarray(val['fechas']))
        orden_test = np.argsort(np.asarray(test['fechas']))
        shim[barra] = {
            '_datos_split': {
                'fechas_val': np.asarray(val['fechas'])[orden_val],
                'y_real_val': np.asarray(val['y_real'])[orden_val],
                'fechas_test': np.asarray(test['fechas'])[orden_test],
                'y_real_test': np.asarray(test['y_real'])[orden_test],
            }
        }
    return shim


def _datos_barra_sc(datos_eval, barra, coordenadas=COORDENADAS_BARRAS_SC):
    """Extrae y ordena por fecha val/test de un dict eval_TFT_{arq}.pkl (datos_eval = pickle.load(...)[0]),
    agregando las features solares [yhat, elevacion_solar, cos_elevacion]."""
    lat, lon = coordenadas[barra]
    val = datos_eval[barra]['val']['Precios']['datos']
    test = datos_eval[barra]['test']['Precios']['datos']

    orden_val = np.argsort(np.asarray(val['fechas']))
    orden_test = np.argsort(np.asarray(test['fechas']))
    y_val = np.asarray(val['y_real'])[orden_val]
    yhat_val = np.asarray(val['y_pred'])[orden_val]
    fechas_val = np.asarray(val['fechas'])[orden_val]
    y_test = np.asarray(test['y_real'])[orden_test]
    yhat_test = np.asarray(test['y_pred'])[orden_test]
    fechas_test = np.asarray(test['fechas'])[orden_test]

    elev_val, cos_val = _features_solares(fechas_val, lat, lon)
    elev_test, cos_test = _features_solares(fechas_test, lat, lon)

    X_val = np.column_stack([yhat_val, elev_val, cos_val])
    X_test = np.column_stack([yhat_test, elev_test, cos_test])
    return X_val, y_val, X_test, y_test


def aci_offline_solar_implementation(datos_eval, barras, alpha=0.05, gamma=0.05,
                                      coordenadas=COORDENADAS_BARRAS_SC, verbose=True):
    """ACI offline sobre S+C: Ridge(yhat_S+C, elevacion_solar, cos_elevacion) fijo desde
    calibración; solo alpha_t/el cuantil se adaptan en el test. Ver acp_offline_implementation
    para la variante P+C-Stacking equivalente."""
    tabla = []
    dict_cp = {}
    dict_alphas = {}

    for barra in barras:
        try:
            X_val, y_val, X_test, y_test = _datos_barra_sc(datos_eval, barra, coordenadas)

            scaler = StandardScaler()
            X_val_s = scaler.fit_transform(X_val)
            X_test_s = scaler.transform(X_test)

            modelo = Ridge(alpha=1.0)
            modelo.fit(X_val_s, y_val)
            y_pred_test = modelo.predict(X_test_s)
            residuos_cal = np.abs(y_val - modelo.predict(X_val_s))

            n_test = len(y_test)
            lower = np.empty(n_test)
            upper = np.empty(n_test)
            alphas_t = np.empty(n_test)
            errors_t = np.empty(n_test)
            alpha_t = alpha

            for t in range(n_test):
                q_level = (1 - alpha_t) * (len(residuos_cal) + 1) / (len(residuos_cal))
                q_t = np.quantile(residuos_cal, np.minimum(q_level, 1.0), method='higher')
                lower[t] = max(y_pred_test[t] - q_t, 0.0)
                upper[t] = y_pred_test[t] + q_t
                err_t = 1 - int((lower[t] <= y_test[t]) & (y_test[t] <= upper[t]))
                errors_t[t] = err_t
                alphas_t[t] = alpha_t
                alpha_t = np.clip(alpha_t + gamma * (alpha - err_t), 0.001, 0.999)

            coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
            width = float(np.mean(upper - lower))
            mae = float(np.mean(np.abs(y_test - y_pred_test)))
            rmse = float(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
            alpha_mean = float(np.mean(alphas_t))
            alpha_std = float(np.std(alphas_t))
            error_rate = float(np.mean(errors_t))

            if verbose:
                print(f"{barra:10s} | gamma={gamma} | coverage={coverage:.4f} | width={width:.2f} | mae={mae:.2f}")

            tabla.append({
                "Barra": barra, "Coverage": coverage, "Expected": 1 - alpha,
                "Coverage_Diff": abs(coverage - (1 - alpha)), "Width": width,
                "MAE": mae, "RMSE": rmse, "N_test": n_test, "Gamma": gamma,
                "Alpha_Mean": alpha_mean, "Alpha_Std": alpha_std, "Error_Rate": error_rate,
                "Modelo_Base": "Ridge_solar",
            })
            dict_cp[barra] = {"y_test": y_test, "y_pred": y_pred_test, "lower": lower, "upper": upper,
                               "coverage": coverage, "width": width, "mae": mae, "rmse": rmse}
            dict_alphas[barra] = {"alphas_t": alphas_t, "errors_t": errors_t}

        except Exception as e:
            print(f"  ERROR EN {barra}: {str(e)}")
            traceback.print_exc()
            continue

    df_tabla = pd.DataFrame(tabla).sort_values("Barra").reset_index(drop=True)
    return df_tabla, dict_cp, dict_alphas


def aci_online_solar_implementation(datos_eval, barras, alpha=0.05, gamma=0.05,
                                     coordenadas=COORDENADAS_BARRAS_SC, verbose=True):
    """ACI online sobre S+C: igual que aci_offline_solar_implementation, pero reentrena
    el Ridge en cada paso del test con todos los datos observados hasta ese instante.
    Ver acp_online_implementation para la variante P+C-Stacking equivalente."""
    tabla = []
    dict_cp = {}
    dict_alphas = {}
    tiempos_ejecucion = {}

    for barra in barras:
        try:
            tiempo_inicio = time.time()
            X_val, y_val, X_test, y_test = _datos_barra_sc(datos_eval, barra, coordenadas)

            train_size = len(X_val)
            n_test = len(y_test)
            X_combined = np.vstack([X_val, X_test])
            y_combined = np.hstack([y_val, y_test])

            scaler = StandardScaler()
            scaler.fit(X_val)
            X_combined_s = scaler.transform(X_combined)

            lower = np.empty(n_test)
            upper = np.empty(n_test)
            alphas_t = np.empty(n_test)
            errors_t = np.empty(n_test)
            y_pred_online = np.empty(n_test)
            alpha_t = alpha

            for t in range(n_test):
                idx_train = np.arange(train_size + t)
                modelo_t = Ridge(alpha=1.0)
                modelo_t.fit(X_combined_s[idx_train], y_combined[idx_train])

                y_pred_t = modelo_t.predict(X_combined_s[train_size + t:train_size + t + 1])[0]
                y_pred_online[t] = y_pred_t

                residuos_cal_t = np.abs(y_combined[idx_train] - modelo_t.predict(X_combined_s[idx_train]))
                q_level = (1 - alpha_t) * (len(residuos_cal_t) + 1) / (len(residuos_cal_t))
                q_t = np.quantile(residuos_cal_t, np.minimum(q_level, 1.0), method='higher')

                lower[t] = max(y_pred_t - q_t, 0.0)
                upper[t] = y_pred_t + q_t
                err_t = 1 - int((lower[t] <= y_test[t]) & (y_test[t] <= upper[t]))
                errors_t[t] = err_t
                alphas_t[t] = alpha_t
                alpha_t = np.clip(alpha_t + gamma * (alpha - err_t), 0.001, 0.999)

            tiempo_total = time.time() - tiempo_inicio

            coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
            width = float(np.mean(upper - lower))
            mae = float(np.mean(np.abs(y_test - y_pred_online)))
            rmse = float(np.sqrt(np.mean((y_test - y_pred_online) ** 2)))
            alpha_mean = float(np.mean(alphas_t))
            alpha_std = float(np.std(alphas_t))
            error_rate = float(np.mean(errors_t))

            if verbose:
                print(f"{barra:10s} | coverage={coverage:.4f} | width={width:.2f} | mae={mae:.2f} | t={tiempo_total:.1f}s")

            tabla.append({
                "Barra": barra, "Coverage": coverage, "Expected": 1 - alpha,
                "Coverage_Diff": abs(coverage - (1 - alpha)), "Width": width,
                "MAE": mae, "RMSE": rmse, "N_test": n_test, "Gamma": gamma,
                "Alpha_Mean": alpha_mean, "Alpha_Std": alpha_std, "Error_Rate": error_rate,
                "Modelo_Base": "Ridge_solar", "Tiempo_Seg": tiempo_total,
            })
            dict_cp[barra] = {"y_test": y_test, "y_pred": y_pred_online, "lower": lower, "upper": upper,
                               "coverage": coverage, "width": width, "mae": mae, "rmse": rmse}
            dict_alphas[barra] = {"alphas_t": alphas_t, "errors_t": errors_t}
            tiempos_ejecucion[barra] = tiempo_total

        except Exception as e:
            print(f"  ERROR EN {barra}: {str(e)}")
            traceback.print_exc()
            continue

    df_tabla = pd.DataFrame(tabla).sort_values("Barra").reset_index(drop=True)
    return df_tabla, dict_cp, dict_alphas, tiempos_ejecucion


# ========================= CONFORMAL QUANTILE REGRESSION (CQR) ======================

def cqr_implementation(resultados_stacking, df_mejores,
                       alpha=0.05,
                       n_folds=5,
                       qr_l1=0.01,
                       split_key="_datos_split",
                       y_val_key="y_real_val", y_test_key="y_real_test",
                       fechas_val_key="fechas_val", fechas_test_key="fechas_test",
                       pred_val_key="prediccion_val", pred_test_key="prediccion_test",
                       mapping='LN', verbose=True):
    """
    CQR: Conformalized Quantile Regression (Romano et al. 2019) con cross-conformal.

    Features = [yhat, sin(2π·h/24), cos(2π·h/24)]  → captura heteroscedasticidad intradiaria.

    Calibración via K-fold cross-conformal (sin shuffle, respeta orden temporal):
      - Cada fold entrena QR en K-1 folds y calcula scores en el fold restante
      - Se acumulan scores de todos los folds → q_cp más estable y sin desfase temporal
      - QR final entrenado en val completo → mejor generalización al test

    Esto evita el problema del split 70/30: el desfase de distribución temporal
    entre la primera y segunda mitad de val inflaba q_cp artificialmente.
    """
    from sklearn.model_selection import KFold

    mapping       = _get_mapping(mapping)
    best_by_barra = df_mejores.set_index("Barra")["Mejor Estrategia"].to_dict()
    tabla         = []
    dict_cp       = {}

    def _make_X(yhat, fechas):
        h = pd.DatetimeIndex(fechas).hour
        r = 2 * np.pi * h / 24
        return np.column_stack([yhat, np.sin(r), np.cos(r)])

    for barra, data in resultados_stacking.items():
        try:
            best_pretty = best_by_barra.get(barra)
            if best_pretty is None:
                if verbose:
                    print(f"[SKIP] {barra}: no aparece en df_mejores")
                continue

            best = mapping.get(best_pretty)
            if best is None:
                raise ValueError(f"estrategia '{best_pretty}' no está en el mapping")
            if best not in data:
                raise ValueError(f"'{best}' no existe en resultados_stacking[{barra}]")

            y_val       = np.asarray(data[split_key][y_val_key],  dtype=float)
            y_test      = np.asarray(data[split_key][y_test_key], dtype=float)
            fechas_val  = data[split_key][fechas_val_key]
            fechas_test = data[split_key][fechas_test_key]
            yhat_val    = np.asarray(data[best][pred_val_key],  dtype=float)
            yhat_test   = np.asarray(data[best][pred_test_key], dtype=float)

            X_val  = _make_X(yhat_val,  fechas_val)
            X_test = _make_X(yhat_test, fechas_test)

            # Paso 1: Cross-conformal — acumular scores de todos los folds
            kf       = KFold(n_splits=n_folds, shuffle=False)
            s_list   = []
            for train_idx, calib_idx in kf.split(X_val):
                qrl = QuantileRegressor(quantile=alpha / 2,     alpha=qr_l1, solver='highs')
                qrh = QuantileRegressor(quantile=1 - alpha / 2, alpha=qr_l1, solver='highs')
                qrl.fit(X_val[train_idx], y_val[train_idx])
                qrh.fit(X_val[train_idx], y_val[train_idx])
                s_fold = np.maximum(
                    qrl.predict(X_val[calib_idx]) - y_val[calib_idx],
                    y_val[calib_idx] - qrh.predict(X_val[calib_idx])
                )
                s_list.append(s_fold)

            s_cal = np.concatenate(s_list)
            n_c   = len(s_cal)
            q_lev = np.ceil((1 - alpha) * (n_c + 1)) / (n_c + 1)
            q_cp  = float(np.quantile(s_cal, q_lev, method='higher'))

            # Paso 2: QR final entrenado en val completo
            qr_low  = QuantileRegressor(quantile=alpha / 2,     alpha=qr_l1, solver='highs')
            qr_high = QuantileRegressor(quantile=1 - alpha / 2, alpha=qr_l1, solver='highs')
            qr_low.fit(X_val,  y_val)
            qr_high.fit(X_val, y_val)

            # Paso 3: Intervalos en test
            lower = qr_low.predict(X_test)  - q_cp
            upper = qr_high.predict(X_test) + q_cp
            lower = np.maximum(lower, 0.0)

            swap = lower > upper
            if swap.any():
                lower[swap], upper[swap] = upper[swap], lower[swap]

            horas_test = pd.DatetimeIndex(fechas_test).hour
            coverage   = float(np.mean((y_test >= lower) & (y_test <= upper)))
            width      = float(np.mean(upper - lower))
            mae        = float(np.mean(np.abs(y_test - yhat_test)))
            rmse       = float(np.sqrt(np.mean((y_test - yhat_test) ** 2)))

            if verbose:
                print(f"{barra:10s} | {best_pretty:30s} | coverage={coverage:.3f} | "
                      f"width={width:.2f} | q_cp={q_cp:.2f}")

            tabla.append({"Barra": barra, "Estrategia": best_pretty,
                          "Coverage": coverage, "Expected": 1 - alpha,
                          "Coverage_Diff": abs(coverage - (1 - alpha)),
                          "Width": width, "MAE": mae, "RMSE": rmse,
                          "q_cp": q_cp, "N_test": len(y_test)})

            dict_cp[barra] = {"y_test": y_test, "y_pred": yhat_test,
                              "lower": lower, "upper": upper,
                              "Estrategia": best_pretty,
                              "coverage": coverage, "width": width,
                              "mae": mae, "rmse": rmse,
                              "q_cp": q_cp,
                              "horas_test": np.asarray(horas_test)}

        except Exception as e:
            print(f"  ERROR EN {barra}: {str(e)}")
            traceback.print_exc()
            continue

    df_tabla = pd.DataFrame(tabla).sort_values("Barra").reset_index(drop=True)
    return df_tabla, dict_cp


# ========================= FIGURA COMPARATIVA POR HORA ======================

def plot_cp_por_hora(metodos, resultados_stacking, barra, alpha=0.05, figsize=(16, 7)):
    """
    Compara coverage y width por hora del día para múltiples métodos CP.

    metodos : dict { nombre: cp_dict }
        cp_dict : { barra: {'y_test', 'lower', 'upper', ...} }
        Si el resultado tiene 'horas_test', se usa directamente.
        Si tiene 'mask_test', se aplica sobre fechas_test del stacking.
        Si no tiene ninguno, se asume alineación completa con fechas_test.
    """
    horas_full = pd.DatetimeIndex(
        resultados_stacking[barra]['_datos_split']['fechas_test']
    ).hour

    colores  = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63',
                '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']
    marcadores = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Análisis por hora del día — {barra}', fontsize=14, fontweight='bold')

    for (nombre, cp_dict), color, marker in zip(metodos.items(), colores, marcadores):
        if barra not in cp_dict:
            continue
        res    = cp_dict[barra]
        y_test = np.asarray(res['y_test'])
        lower  = np.asarray(res['lower'])
        upper  = np.asarray(res['upper'])

        # Obtener array de horas alineado con y_test / lower / upper
        if 'horas_test' in res:
            horas = np.asarray(res['horas_test'])
        elif 'mask_test' in res:
            horas = horas_full[np.asarray(res['mask_test'])]
        else:
            horas = horas_full[:len(y_test)]

        n = min(len(y_test), len(lower), len(upper), len(horas))
        y_test, lower, upper, horas = y_test[:n], lower[:n], upper[:n], horas[:n]

        cov_h   = []
        width_h = []
        for h in range(24):
            mask_h = horas == h
            if mask_h.sum() == 0:
                cov_h.append(np.nan)
                width_h.append(np.nan)
            else:
                cov_h.append(float(np.mean((y_test[mask_h] >= lower[mask_h]) &
                                           (y_test[mask_h] <= upper[mask_h]))))
                width_h.append(float(np.mean(upper[mask_h] - lower[mask_h])))

        x = np.arange(24)
        axes[0].plot(x, cov_h,   marker=marker, color=color, lw=2, label=nombre)
        axes[1].plot(x, width_h, marker=marker, color=color, lw=2, label=nombre)

    # Panel coverage
    axes[0].axhline(1 - alpha, color='red', ls='--', lw=1.5,
                    label=f'Objetivo ({1 - alpha:.0%})')
    axes[0].set_xlabel('Hora del día', fontsize=11)
    axes[0].set_ylabel('Coverage', fontsize=11)
    axes[0].set_title('Coverage por hora', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(24))
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel width
    axes[1].set_xlabel('Hora del día', fontsize=11)
    axes[1].set_ylabel('Width (USD/MWh)', fontsize=11)
    axes[1].set_title('Width promedio por hora', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(24))
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Sombrear horas de transición solar
    for ax in axes:
        for h in [7, 8, 19, 20, 21]:
            ax.axvspan(h - 0.5, h + 0.5, alpha=0.10, color='gold', zorder=0)

    plt.tight_layout()
    plt.show()


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


# ========================= COMPARATIVA GENERAL DE MÉTODOS CP ======================

def comparativa_metodos_cp(todos_metodos, lista_barras, alpha=0.05):
    """
    Genera tablas y gráficas comparativas para múltiples métodos de CP.

    Parámetros
    ----------
    todos_metodos : dict
        Diccionario con la estructura:
        {
          'Nombre Método': {
              'ln':     df_ln,     # DataFrame con columnas Barra, Coverage, Width, MAE, RMSE
              'dyt':    df_dyt,
              'cp_ln':  cp_ln,     # dict barra -> resultados (no usado en tablas, solo referencia)
              'cp_dyt': cp_dyt,
              'color':  '#RRGGBB',
              'marker': 'o'
          }, ...
        }
    lista_barras : list[str]
        Barras a incluir en el análisis.
    alpha : float
        Nivel de significancia objetivo (default 0.05 → cobertura objetivo 95%).

    Retorna
    -------
    df_tabla_general, df_tabla_por_barra : pd.DataFrame
    """

    from IPython.display import display

    target_cov = 1 - alpha

    # ── 1. TABLA GENERAL ──────────────────────────────────────────────────────
    print("\n" + "="*150)
    print("TABLA 1: COMPARATIVA GENERAL - TODOS LOS MÉTODOS CP")
    print("="*150 + "\n")

    tabla_general = []
    for metodo_nombre, md in todos_metodos.items():
        cov_ln   = md['ln']['Coverage'].mean()
        cov_dyt  = md['dyt']['Coverage'].mean()
        cov_prom = (cov_ln + cov_dyt) / 2

        w_ln   = md['ln']['Width'].mean()
        w_dyt  = md['dyt']['Width'].mean()
        w_prom = (w_ln + w_dyt) / 2

        mae_ln   = md['ln']['MAE'].mean()
        mae_dyt  = md['dyt']['MAE'].mean()
        mae_prom = (mae_ln + mae_dyt) / 2

        rmse_ln   = md['ln']['RMSE'].mean()
        rmse_dyt  = md['dyt']['RMSE'].mean()
        rmse_prom = (rmse_ln + rmse_dyt) / 2

        tabla_general.append({
            'Método':             metodo_nombre,
            'Coverage_LN':        f"{cov_ln:.4f}",
            'Coverage_DyT':       f"{cov_dyt:.4f}",
            'Coverage_Promedio':  f"{cov_prom:.4f}",
            'Coverage_Error':     f"{abs(cov_prom - target_cov):.4f}",
            'Width_LN':           f"{w_ln:.2f}",
            'Width_DyT':          f"{w_dyt:.2f}",
            'Width_Promedio':     f"{w_prom:.2f}",
            'MAE_LN':             f"{mae_ln:.4f}",
            'MAE_DyT':            f"{mae_dyt:.4f}",
            'MAE_Promedio':       f"{mae_prom:.4f}",
            'RMSE_Promedio':      f"{rmse_prom:.4f}",
        })

    df_tabla_general = pd.DataFrame(tabla_general)
    display(df_tabla_general)

    # ── 2. TABLA POR BARRA ────────────────────────────────────────────────────
    print("\n" + "="*200)
    print("TABLA 2: COMPARATIVA POR BARRA")
    print("="*200 + "\n")

    tabla_por_barra = []
    for barra in lista_barras:
        for metodo_nombre, md in todos_metodos.items():
            df_ln  = md['ln']
            df_dyt = md['dyt']

            rows_ln  = df_ln[df_ln['Barra'] == barra]
            rows_dyt = df_dyt[df_dyt['Barra'] == barra]
            if rows_ln.empty or rows_dyt.empty:
                continue

            r_ln  = rows_ln.iloc[0]
            r_dyt = rows_dyt.iloc[0]

            cov_prom  = (r_ln['Coverage'] + r_dyt['Coverage']) / 2
            w_prom    = (r_ln['Width']    + r_dyt['Width'])    / 2
            mae_prom  = (r_ln['MAE']      + r_dyt['MAE'])      / 2
            rmse_prom = (r_ln['RMSE']     + r_dyt['RMSE'])     / 2

            tabla_por_barra.append({
                'Barra':             barra,
                'Método':            metodo_nombre,
                'Coverage_LN':       f"{r_ln['Coverage']:.4f}",
                'Coverage_DyT':      f"{r_dyt['Coverage']:.4f}",
                'Coverage_Promedio': f"{cov_prom:.4f}",
                'Width_Promedio':    f"{w_prom:.2f}",
                'MAE_Promedio':      f"{mae_prom:.4f}",
                'RMSE_Promedio':     f"{rmse_prom:.4f}",
            })

    df_tabla_por_barra = pd.DataFrame(tabla_por_barra)
    display(df_tabla_por_barra)

    # ── 3. RANKINGS ───────────────────────────────────────────────────────────
    print("\n" + "="*150)
    print("TABLA 3: RANKINGS - MEJOR MÉTODO POR MÉTRICA")
    print("="*150 + "\n")

    def _ranking_barra(metrica_fn, reverse=True):
        rows = []
        for barra in lista_barras:
            scores = []
            for metodo_nombre, md in todos_metodos.items():
                r_ln  = md['ln'][md['ln']['Barra'] == barra]
                r_dyt = md['dyt'][md['dyt']['Barra'] == barra]
                if r_ln.empty or r_dyt.empty:
                    continue
                val = metrica_fn(r_ln.iloc[0], r_dyt.iloc[0])
                scores.append((metodo_nombre, val))
            scores.sort(key=lambda x: x[1], reverse=reverse)
            row = {'Barra': barra}
            labels = ['1er Lugar', '2do Lugar', '3er Lugar', '4to Lugar']
            for i, (nombre, val) in enumerate(scores[:4]):
                fmt = f"{nombre} ({val:.4f})"
                row[labels[i]] = fmt
            rows.append(row)
        return pd.DataFrame(rows)

    print("COVERAGE (más alto = mejor):")
    df_rank_cov = _ranking_barra(
        lambda ln, dyt: (ln['Coverage'] + dyt['Coverage']) / 2,
        reverse=True
    )
    print(df_rank_cov.to_string(index=False))

    print("\n\nWIDTH (más bajo = mejor):")
    df_rank_w = _ranking_barra(
        lambda ln, dyt: (ln['Width'] + dyt['Width']) / 2,
        reverse=False
    )
    print(df_rank_w.to_string(index=False))

    # ── 4. GRÁFICAS ───────────────────────────────────────────────────────────
    print("\n\n" + "="*150)
    print("CREANDO GRÁFICAS COMPARATIVAS...")
    print("="*150)

    n_cols = 2
    n_rows = (len(lista_barras) + n_cols - 1) // n_cols

    def _bar_chart(metrica_fn, ylabel, titulo_fig, fmt='.2f', hline=None):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        axes = np.array(axes).flatten()
        fig.suptitle(titulo_fig, fontsize=16, fontweight='bold')

        for idx, barra in enumerate(lista_barras):
            ax = axes[idx]
            vals, nombres, colores = [], [], []
            for metodo_nombre, md in todos_metodos.items():
                r_ln  = md['ln'][md['ln']['Barra'] == barra]
                r_dyt = md['dyt'][md['dyt']['Barra'] == barra]
                if r_ln.empty or r_dyt.empty:
                    continue
                vals.append(metrica_fn(r_ln.iloc[0], r_dyt.iloc[0]))
                nombres.append(metodo_nombre)
                colores.append(md['color'])

            bars = ax.bar(nombres, vals, color=colores, alpha=0.7, edgecolor='black', linewidth=2)
            if hline is not None:
                ax.axhline(hline, color='red', linestyle='--', linewidth=2, label=f'Objetivo ({hline:.0%})')
                ax.legend()
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(barra, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars, vals):
                label = f'{val:{fmt}}' if fmt != '.2%' else f'{val:.2%}'
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        label, ha='center', va='bottom', fontweight='bold', fontsize=9)

        for j in range(len(lista_barras), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()

    _bar_chart(
        lambda ln, dyt: (ln['Coverage'] + dyt['Coverage']) / 2,
        'Coverage', f'Comparativa Coverage — {len(todos_metodos)} Métodos CP',
        fmt='.2%', hline=target_cov
    )
    _bar_chart(
        lambda ln, dyt: (ln['Width'] + dyt['Width']) / 2,
        'Width (USD/MWh)', f'Comparativa Width — {len(todos_metodos)} Métodos CP',
        fmt='.1f'
    )
    _bar_chart(
        lambda ln, dyt: (ln['MAE'] + dyt['MAE']) / 2,
        'MAE (USD/MWh)', f'Comparativa MAE — {len(todos_metodos)} Métodos CP',
        fmt='.2f'
    )

    # Scatter Coverage vs Width
    fig, ax = plt.subplots(figsize=(15, 9))
    for metodo_nombre, md in todos_metodos.items():
        coverages, widths = [], []
        for barra in lista_barras:
            r_ln  = md['ln'][md['ln']['Barra'] == barra]
            r_dyt = md['dyt'][md['dyt']['Barra'] == barra]
            if r_ln.empty or r_dyt.empty:
                continue
            coverages.append((r_ln.iloc[0]['Coverage'] + r_dyt.iloc[0]['Coverage']) / 2)
            widths.append((r_ln.iloc[0]['Width'] + r_dyt.iloc[0]['Width']) / 2)
        ax.scatter(widths, coverages, s=350, marker=md['marker'],
                   label=metodo_nombre, color=md['color'], alpha=0.7,
                   edgecolors='black', linewidth=2)
        for i, barra in enumerate(lista_barras):
            if i < len(widths):
                ax.annotate(barra, (widths[i], coverages[i]), fontsize=8,
                            ha='center', fontweight='bold')
    ax.axhline(target_cov, color='red', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Objetivo Coverage ({target_cov:.0%})')
    ax.set_xlabel('Width (USD/MWh)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coverage', fontsize=12, fontweight='bold')
    ax.set_title(f'Trade-off: Coverage vs Width — {len(todos_metodos)} Métodos CP',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.show()

    # Heatmaps Coverage LN / DyT
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, split in zip(axes, ['ln', 'dyt']):
        data = []
        for md in todos_metodos.values():
            row = []
            for barra in lista_barras:
                r = md[split][md[split]['Barra'] == barra]
                row.append(r.iloc[0]['Coverage'] if not r.empty else 0)
            data.append(row)
        sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlGn',
                    center=target_cov, vmin=0.88, vmax=0.96,
                    xticklabels=lista_barras,
                    yticklabels=list(todos_metodos.keys()),
                    ax=ax, cbar_kws={'label': 'Coverage'})
        ax.set_title(f'Heatmap Coverage — {split.upper()} Strategy',
                     fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ── 5. RESUMEN EJECUTIVO ──────────────────────────────────────────────────
    print("\n" + "="*180)
    print("TABLA 4: RESUMEN EJECUTIVO")
    print("="*180 + "\n")

    resumen_rows = []
    for i, row in df_tabla_general.iterrows():
        resumen_rows.append({
            'Método':            row['Método'],
            'Coverage Promedio': row['Coverage_Promedio'],
            'Coverage Error':    row['Coverage_Error'],
            'Width Promedio':    f"{row['Width_Promedio']} USD/MWh",
            'MAE Promedio':      f"{row['MAE_Promedio']} USD/MWh",
            'RMSE Promedio':     f"{row['RMSE_Promedio']} USD/MWh",
        })
    df_resumen = pd.DataFrame(resumen_rows)
    print(df_resumen.to_string(index=False))

    print("\n" + "="*180)
    print("COMPARATIVA COMPLETADA")
    print("="*180)

    return df_tabla_general, df_tabla_por_barra