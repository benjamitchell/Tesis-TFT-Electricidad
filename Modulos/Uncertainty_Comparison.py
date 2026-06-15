import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════════
# QUANTILE REGRESSION
# ════════════════════════════════════════════════════════════════════════════════

def quantile_regression(y_calib, yhat_calib, y_test, yhat_test, alpha=0.05, verbose=False):
    """
    Quantile regression para intervalos de predicción.
    Entrena dos modelos (α/2 y 1-α/2) y garantiza lower ≥ 0.
    """
    y_calib = np.asarray(y_calib).flatten()
    yhat_calib = np.asarray(yhat_calib).flatten()
    y_test = np.asarray(y_test).flatten()
    yhat_test = np.asarray(yhat_test).flatten()

    X_calib = yhat_calib.reshape(-1, 1)
    X_test = yhat_test.reshape(-1, 1)

    try:
        # Lower quantile (α/2)
        qr_lower = QuantileRegressor(quantile=alpha/2, alpha=0.01, solver='highs')
        qr_lower.fit(X_calib, y_calib)
        lower = qr_lower.predict(X_test)

        # Upper quantile (1-α/2)
        qr_upper = QuantileRegressor(quantile=1-alpha/2, alpha=0.01, solver='highs')
        qr_upper.fit(X_calib, y_calib)
        upper = qr_upper.predict(X_test)

        # Garantizar lower ≥ 0
        lower = np.maximum(lower, 0)

        # Validar intervalos válidos
        invalid = upper < lower
        if np.any(invalid):
            upper[invalid] = lower[invalid] + 1.0

        # Métricas
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        width = np.mean(upper - lower)
        mae = np.mean(np.abs(y_test - yhat_test))
        rmse = np.sqrt(np.mean((y_test - yhat_test)**2))

        if verbose:
            print(f"  QR: coverage={coverage:.3f}, width={width:.2f}, mae={mae:.3f}")

        return {
            "lower": lower,
            "upper": upper,
            "coverage": float(coverage),
            "expected_coverage": float(1 - alpha),
            "width": float(width),
            "mae": float(mae),
            "rmse": float(rmse),
            "y_test": y_test,
            "y_pred": yhat_test
        }
    except Exception as e:
        print(f"  ERROR QR: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════════
# NETWORK LOGNORMAL
# ════════════════════════════════════════════════════════════════════════════════

def network_lognormal(y_calib, yhat_calib, y_test, yhat_test, alpha=0.05,
                      epochs=100, verbose=False):
    """
    Red neuronal que predice parámetros (μ, σ) de Lognormal.
    Arquitectura: 1 → 128 → 64 → 32 → 16 → 2 (softplus)
    """
    if not TF_AVAILABLE:
        print("ERROR: TensorFlow no disponible. Instala con: pip install tensorflow")
        return None

    y_calib = np.asarray(y_calib).flatten()
    yhat_calib = np.asarray(yhat_calib).flatten()
    y_test = np.asarray(y_test).flatten()
    yhat_test = np.asarray(yhat_test).flatten()

    try:
        # Normalizar entrada
        scaler = StandardScaler()
        X_calib = scaler.fit_transform(yhat_calib.reshape(-1, 1))
        X_test = scaler.transform(yhat_test.reshape(-1, 1))

        # Build model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(1,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(2, activation='softplus')  # [μ_log, σ_log], σ ≥ ~0.01
        ])

        # Custom loss: Negative log-likelihood of Lognormal
        def nll_lognormal(y_true, y_pred):
            mu = y_pred[:, 0]
            sigma = y_pred[:, 1] + 0.01  # Asegurar σ > 0
            log_y = tf.math.log(tf.maximum(y_true, 1e-6))
            nll = -tf.math.log(1.0 / (sigma * y_true * tf.sqrt(2 * np.pi))) \
                  - 0.5 * ((log_y - mu) / sigma)**2
            return -tf.reduce_mean(nll)

        model.compile(optimizer='adam', loss=nll_lognormal)

        # Entrenar
        with tf.device('/CPU:0'):
            hist = model.fit(X_calib, y_calib,
                           epochs=epochs,
                           batch_size=16,
                           verbose=0,
                           validation_split=0.2)

        # Predecir parámetros
        params = model.predict(X_test, verbose=0)
        mu_pred = params[:, 0]
        sigma_pred = np.maximum(params[:, 1], 0.01)

        # Calcular intervalos de Lognormal
        z = stats.norm.ppf(1 - alpha/2)
        lower = np.exp(mu_pred - z * sigma_pred)
        upper = np.exp(mu_pred + z * sigma_pred)

        # Garantizar lower ≥ 0 (debe ser automático pero por seguridad)
        lower = np.maximum(lower, 0)

        # Métricas
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        width = np.mean(upper - lower)
        mae = np.mean(np.abs(y_test - yhat_test))
        rmse = np.sqrt(np.mean((y_test - yhat_test)**2))

        if verbose:
            print(f"  Net-Lognormal: coverage={coverage:.3f}, width={width:.2f}, mae={mae:.3f}")

        return {
            "lower": lower,
            "upper": upper,
            "coverage": float(coverage),
            "expected_coverage": float(1 - alpha),
            "width": float(width),
            "mae": float(mae),
            "rmse": float(rmse),
            "y_test": y_test,
            "y_pred": yhat_test,
            "mu": mu_pred,
            "sigma": sigma_pred,
            "model": model
        }
    except Exception as e:
        print(f"  ERROR Network: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════════
# CONFORMAL PREDICTION BASELINE
# ════════════════════════════════════════════════════════════════════════════════

def cp_baseline(y_calib, yhat_calib, y_test, yhat_test, alpha=0.05, verbose=False):
    """
    Wrapper para CP clásico (split conformal prediction).
    """
    y_calib = np.asarray(y_calib).flatten()
    yhat_calib = np.asarray(yhat_calib).flatten()
    y_test = np.asarray(y_test).flatten()
    yhat_test = np.asarray(yhat_test).flatten()

    # Non-conformity scores (residuos absolutos)
    scores = np.abs(y_calib - yhat_calib)

    # Quantil
    n = len(scores)
    q_level = np.ceil((1 - alpha) * (n + 1)) / (n + 1)
    q = np.quantile(scores, q_level, method='higher')

    # Intervalos
    lower = yhat_test - q
    upper = yhat_test + q

    # Métricas
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    width = np.mean(upper - lower)
    mae = np.mean(np.abs(y_test - yhat_test))
    rmse = np.sqrt(np.mean((y_test - yhat_test)**2))

    if verbose:
        print(f"  CP: coverage={coverage:.3f}, width={width:.2f}, mae={mae:.3f}")

    return {
        "lower": lower,
        "upper": upper,
        "coverage": float(coverage),
        "expected_coverage": float(1 - alpha),
        "width": float(width),
        "mae": float(mae),
        "rmse": float(rmse),
        "y_test": y_test,
        "y_pred": yhat_test,
        "q": float(q)
    }

# ════════════════════════════════════════════════════════════════════════════════
# COMPARISON ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════════

def compare_uncertainty_methods(barra, resultados_stacking, df_mejores,
                                mapping='LN',
                                alpha=0.05,
                                epochs=100,
                                verbose=True):
    """
    Compara CP vs Quantile Regression vs Network Lognormal en una barra.

    Parameters:
    -----------
    barra : str
        Nombre de la barra (e.g., "ATACAMA")
    resultados_stacking : dict
        Resultados del stacking con estructura: resultados_stacking[barra][estrategia][...]
    df_mejores : pd.DataFrame
        DataFrame con Barra, Mejor Estrategia, ...
    mapping : str
        'LN' o 'DyT' para mapear nombres de estrategias
    alpha : float
        Nivel de cobertura: 1-alpha (default 0.95)
    epochs : int
        Epochs para la red neuronal
    verbose : bool
        Imprimir mensajes de progreso

    Returns:
    --------
    dict : {"CP": {...}, "QR": {...}, "Network": {...}, "df_summary": pd.DataFrame}
    """

    if mapping == 'LN':
        mapping_dict = {
            "Prophet Solo": "Prophet_Solo",
            "TFT_LN Precios Solo": "TFT_Precios_Solo",
            "Prophet+TFT_LN Precios": "Prophet_TFT_Precios",
            "TFT Residuos Solo": "TFT_Residuos_Solo",
            "Prophet + Residuos (Directo)": "Prophet_Residuos_Directo",
            "Prophet + Residuos (Opt)": "Prophet_Residuos_Opt",
            "(Prophet + Residuos) + TFT_LN Precios": "Prophet_Residuos_Precios"
        }
    else:  # DyT
        mapping_dict = {
            "Prophet Solo": "Prophet_Solo",
            "TFT_DyT Precios Solo": "TFT_Precios_Solo",
            "Prophet+TFT_DyT Precios": "Prophet_TFT_Precios",
            "TFT_DyT Residuos Solo": "TFT_Residuos_Solo",
            "Prophet + Residuos (Directo)": "Prophet_Residuos_Directo",
            "Prophet + Residuos (Opt)": "Prophet_Residuos_Opt",
            "(Prophet + Residuos) + TFT_DyT Precios": "Prophet_Residuos_Precios"
        }

    try:
        # Obtener mejor estrategia
        best_pretty = df_mejores[df_mejores['Barra'] == barra]['Mejor Estrategia'].values[0]
        best_key = mapping_dict.get(best_pretty)

        if best_key is None or best_key not in resultados_stacking[barra]:
            print(f"[ERROR] {barra}: estrategia no encontrada ({best_pretty} → {best_key})")
            return None

        # Extraer datos
        y_val = resultados_stacking[barra]["_datos_split"]["y_real_val"]
        y_test = resultados_stacking[barra]["_datos_split"]["y_real_test"]
        yhat_val = resultados_stacking[barra][best_key]["prediccion_val"]
        yhat_test = resultados_stacking[barra][best_key]["prediccion_test"]

        # Limpiar NaNs
        mask_val = np.isfinite(y_val) & np.isfinite(yhat_val)
        mask_test = np.isfinite(y_test) & np.isfinite(yhat_test)

        y_val_clean = y_val[mask_val]
        yhat_val_clean = yhat_val[mask_val]
        y_test_clean = y_test[mask_test]
        yhat_test_clean = yhat_test[mask_test]

        if len(y_val_clean) < 10 or len(y_test_clean) < 10:
            print(f"[ERROR] {barra}: datos insuficientes después de limpiar NaNs")
            return None

        if verbose:
            print(f"\n{barra} - {best_pretty}")
            print(f"  Datos calib: {len(y_val_clean)}, test: {len(y_test_clean)}")

        # Correr métodos
        results = {}

        cp_res = cp_baseline(y_val_clean, yhat_val_clean, y_test_clean, yhat_test_clean,
                             alpha=alpha, verbose=verbose)
        results["CP"] = cp_res

        qr_res = quantile_regression(y_val_clean, yhat_val_clean, y_test_clean, yhat_test_clean,
                                     alpha=alpha, verbose=verbose)
        results["QuantileReg"] = qr_res

        net_res = network_lognormal(y_val_clean, yhat_val_clean, y_test_clean, yhat_test_clean,
                                    alpha=alpha, epochs=epochs, verbose=verbose)
        results["Network"] = net_res

        # Crear DataFrame de comparación
        summary_rows = []
        for method, res in results.items():
            if res is not None:
                summary_rows.append({
                    "Barra": barra,
                    "Metodo": method,
                    "Coverage": res["coverage"],
                    "Expected": res["expected_coverage"],
                    "Width": res["width"],
                    "MAE": res["mae"],
                    "RMSE": res["rmse"]
                })

        df_summary = pd.DataFrame(summary_rows)

        if verbose:
            print("\nRESUMEN:")
            print(df_summary.to_string(index=False))

        results["df_summary"] = df_summary
        results["barra"] = barra
        results["estrategia"] = best_pretty

        return results

    except Exception as e:
        print(f"[ERROR] {barra}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════════

def plot_comparison(results, save_path=None, show=True):
    """
    Visualiza la comparación de 3 métodos para una barra.
    4 subplots: serie completa, coverage, width, zoom últimos 30 días
    """
    if results is None:
        print("Error: resultados vacíos")
        return

    barra = results["barra"]
    fechas = np.arange(len(results["CP"]["y_test"]))
    y_test = results["CP"]["y_test"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    colors = {"CP": "orange", "QuantileReg": "green", "Network": "purple"}

    # ── SUBPLOT 0: Serie completa con intervalos ─────────────────────────────────
    ax = axes[0, 0]
    ax.plot(fechas, y_test, label="Real", lw=2.5, color="black", alpha=0.9, zorder=5)

    for method, color in colors.items():
        res = results[method]
        if res is not None:
            ax.fill_between(fechas, res["lower"], res["upper"],
                           color=color, alpha=0.2, label=f"{method}")

    ax.set_title(f"{barra} - Serie completa", fontsize=13, fontweight="bold")
    ax.set_xlabel("Índice temporal")
    ax.set_ylabel("Precio (USD/MWh)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── SUBPLOT 1: Comparación Coverage ────────────────────────────────────────
    ax = axes[0, 1]
    methods = []
    coverages = []
    for method, color in colors.items():
        res = results[method]
        if res is not None:
            methods.append(method)
            coverages.append(res["coverage"])

    bars = ax.bar(methods, coverages, color=[colors[m] for m in methods], alpha=0.7, edgecolor="black")
    ax.axhline(y=0.95, color="red", linestyle="--", lw=2, label="Target (0.95)")
    ax.set_ylabel("Coverage", fontsize=11)
    ax.set_title("Coverage (debe ≈ 0.95)", fontsize=13, fontweight="bold")
    ax.set_ylim([0.85, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar, cov in zip(bars, coverages):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f'{cov:.3f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)

    # ── SUBPLOT 2: Comparación Width ──────────────────────────────────────────
    ax = axes[1, 0]
    widths = []
    for method in methods:
        res = results[method]
        widths.append(res["width"])

    bars = ax.bar(methods, widths, color=[colors[m] for m in methods], alpha=0.7, edgecolor="black")
    ax.set_ylabel("Width (USD/MWh)", fontsize=11)
    ax.set_title("Ancho promedio de intervalos (menor es mejor)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, w in zip(bars, widths):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f'{w:.1f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)

    # ── SUBPLOT 3: Zoom últimos 30 días ────────────────────────────────────────
    ax = axes[1, 1]
    n_zoom = min(30 * 24, len(y_test))  # 30 días = 720 horas
    f_zoom = fechas[-n_zoom:]
    y_zoom = y_test[-n_zoom:]

    ax.plot(f_zoom, y_zoom, label="Real", lw=2.5, color="black", alpha=0.9, zorder=5)
    for method, color in colors.items():
        res = results[method]
        if res is not None:
            ax.fill_between(f_zoom, res["lower"][-n_zoom:], res["upper"][-n_zoom:],
                           color=color, alpha=0.25, label=f"{method}")

    ax.set_title(f"ZOOM - Últimos 30 días", fontsize=13, fontweight="bold")
    ax.set_xlabel("Índice temporal")
    ax.set_ylabel("Precio (USD/MWh)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Comparación métodos de incertidumbre — {barra}",
                fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Guardado: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
