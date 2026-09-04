#!/usr/bin/env python3
"""
eval_tft.py - Evalúa modelos TFT (LN o DyT) desde terminal sin necesidad del notebook.

Carga modelos entrenados desde checkpoints, ejecuta inferencia y guarda los
resultados en un archivo .pkl que puede ser cargado en el notebook.

Uso:
    python eval_tft.py --tipo-modelo LN  --carpeta-modelos Multi-Modelos_TFT/D/LN/...  --barras ATACAMA CARDONES CHARRUA
    python eval_tft.py --tipo-modelo DyT --carpeta-modelos Multi-Modelos_TFT/D/DyT/... --barras ATACAMA CARDONES CHARRUA

Archivos requeridos en --carpeta-datos (por defecto = --carpeta-modelos):
    - dataloaders_precios.pkl
    - dataloaders_residuos.pkl
    - datasets_norm.pkl       ← exportar desde el notebook (ver instrucciones abajo)
    - scalers.pkl             ← exportar desde el notebook (ver instrucciones abajo)

Cómo exportar desde el notebook:
    import pickle, os
    ruta = "Multi-Modelos_TFT/D/LN/pred_1_90_Features_10"  # tu carpeta
    with open(os.path.join(ruta, 'datasets_norm.pkl'), 'wb') as f:
        pickle.dump(datasets_norm, f)
    with open(os.path.join(ruta, 'scalers.pkl'), 'wb') as f:
        pickle.dump(diccionario_scalers, f)

Cómo cargar resultados en el notebook:
    import pickle
    with open('resultados/eval_TFT_LN.pkl', 'rb') as f:
        datos_ln, metricas_ln = pickle.load(f)
"""

import argparse
import os
import pickle
import sys
import time
import traceback

from tqdm import tqdm

# ─── Resolver sys.path para encontrar Modulos/ ───────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_search_dir = _script_dir
while True:
    if os.path.isdir(os.path.join(_search_dir, "Modulos")):
        break
    _parent = os.path.dirname(_search_dir)
    if _parent == _search_dir:
        print(
            "ERROR: No se encontró la carpeta 'Modulos/' partiendo desde"
            f" '{_script_dir}'."
        )
        sys.exit(1)
    _search_dir = _parent
sys.path.insert(0, _search_dir)

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para terminal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa modelos TFT (LN o DyT) y guarda resultados en .pkl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--tipo-modelo",
        choices=["LN", "DyT"],
        required=True,
        help="Tipo de modelo a evaluar: LN (LayerNorm) o DyT (Dynamic Tanh).",
    )
    parser.add_argument(
        "--carpeta-modelos",
        required=True,
        help="Carpeta raíz donde están los checkpoints (ej. Multi-Modelos_TFT/D/LN/...).",
    )
    parser.add_argument(
        "--carpeta-datos",
        default=None,
        help=(
            "Carpeta con los pkl de datos (dataloaders, datasets_norm, scalers)."
            " Por defecto = --carpeta-modelos."
        ),
    )
    parser.add_argument(
        "--carpeta-resultados",
        default="resultados",
        help="Carpeta donde se guardarán el .pkl de resultados y los gráficos PNG (default: resultados/).",
    )
    parser.add_argument(
        "--nombre-modelo",
        default=None,
        help=(
            "Nombre para etiquetas y archivos de salida (ej. TFT_LN, TFT_DyT)."
            " Por defecto: TFT_<tipo-modelo>."
        ),
    )
    parser.add_argument(
        "--barras",
        nargs="+",
        default=None,
        metavar="BARRA",
        help="Barras a evaluar. Por defecto: todas las disponibles en los dataloaders.",
    )
    parser.add_argument(
        "--horizonte",
        type=int,
        default=1,
        help="Horizonte de predicción a evaluar (default: 1).",
    )
    parser.add_argument(
        "--umbral-desfase",
        type=float,
        default=0.001,
        help="Umbral para detección de desfase temporal (default: 0.001).",
    )
    parser.add_argument(
        "--solo-precios",
        action="store_true",
        help="Evaluar solo modelos de precios.",
    )
    parser.add_argument(
        "--solo-residuos",
        action="store_true",
        help="Evaluar solo modelos de residuos.",
    )
    parser.add_argument(
        "--carpeta-graficos",
        default=None,
        help="Carpeta donde se guardarán los gráficos PNG.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar métricas detalladas por barra.",
    )

    return parser.parse_args()


def _cargar_pickle(ruta, descripcion="archivo"):
    print(f"Cargando {descripcion} desde: {ruta}")
    with open(ruta, "rb") as f:
        return pickle.load(f)


def _verificar_archivos(*rutas):
    faltantes = [r for r in rutas if not os.path.exists(r)]
    if faltantes:
        print("\nERROR: Archivos requeridos no encontrados:")
        for r in faltantes:
            print(f"  - {r}")
        sys.exit(1)


def _cargar_modelo_ln(barra, experimento, carpeta_modelos):
    from Modulos.TFT_Model import cargar_modelo_entrenado
    modelo, config = cargar_modelo_entrenado(barra, experimento, carpeta_modelos)
    if modelo is None:
        raise FileNotFoundError(f"No se encontró checkpoint para {barra} ({experimento})")
    return modelo


def _cargar_modelo_dyt(barra, experimento, carpeta_modelos):
    import json
    import torch
    from Modulos.TFT_Model import TFTBridge
    from Modulos.Parche import activar_dyt_mode, desactivar_dyt_mode

    carpeta_exp = os.path.join(carpeta_modelos, experimento, barra)
    ruta_config = os.path.join(carpeta_exp, f"{experimento}_{barra}_config.json")

    if not os.path.exists(ruta_config):
        raise FileNotFoundError(f"No se encontró config: {ruta_config}")

    try:
        with open(ruta_config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except UnicodeDecodeError:
        with open(ruta_config, "r", encoding="latin-1") as f:
            config = json.load(f)

    ruta_checkpoint = config.get("model_path")
    if not os.path.exists(ruta_checkpoint):
        raise FileNotFoundError(f"No se encontró checkpoint: {ruta_checkpoint}")

    # Patch para redirigir tensores CUDA a CPU al cargar en máquina sin GPU
    _orig_zeros = torch.zeros
    def _safe_zeros(*args, **kwargs):
        if "cuda" in str(kwargs.get("device", "")):
            kwargs["device"] = "cpu"
        return _orig_zeros(*args, **kwargs)

    activar_dyt_mode()
    torch.zeros = _safe_zeros
    try:
        modelo = TFTBridge.load_from_checkpoint(
            ruta_checkpoint,
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
    finally:
        torch.zeros = _orig_zeros
        desactivar_dyt_mode()

    modelo.eval()
    print(f"  Modelo DyT para {barra} cargado desde: {ruta_checkpoint}")
    return modelo


def main():
    args = parse_args()

    if args.solo_precios and args.solo_residuos:
        print("ERROR: --solo-precios y --solo-residuos no pueden usarse juntos.")
        sys.exit(1)

    tipo        = args.tipo_modelo
    carpeta_mod = args.carpeta_modelos
    carpeta_dat = args.carpeta_datos or carpeta_mod
    carpeta_res = args.carpeta_resultados
    nombre_mod  = args.nombre_modelo or f"TFT_{tipo}"

    os.makedirs(carpeta_res, exist_ok=True)

    # Nombres de experimento según tipo
    if tipo == "LN":
        exp_precios  = "Multi-TFT_Precios"
        exp_residuos = "Multi-TFT_Residuos"
    else:  # DyT
        exp_precios  = "Precios"
        exp_residuos = "Residuos"

    # ─── Validar y cargar archivos de datos ──────────────────────────────────
    ruta_dl_p   = os.path.join(carpeta_dat, "dataloaders_precios.pkl")
    ruta_dl_r   = os.path.join(carpeta_dat, "dataloaders_residuos.pkl")
    ruta_dsn    = os.path.join(carpeta_dat, "datasets_norm.pkl")
    ruta_sc     = os.path.join(carpeta_dat, "scalers.pkl")

    _verificar_archivos(ruta_dl_p, ruta_dl_r, ruta_dsn, ruta_sc)

    print("\n" + "=" * 80)
    print(f"EVALUACIÓN {nombre_mod}")
    print("=" * 80)

    dataloaders_precios  = _cargar_pickle(ruta_dl_p,  "dataloaders_precios.pkl")
    dataloaders_residuos = _cargar_pickle(ruta_dl_r,  "dataloaders_residuos.pkl")
    datasets_norm        = _cargar_pickle(ruta_dsn,   "datasets_norm.pkl")
    diccionario_scalers  = _cargar_pickle(ruta_sc,    "scalers.pkl")

    lista_barras = args.barras or list(dataloaders_precios.keys())
    print(f"\nBarras a evaluar ({len(lista_barras)}): {lista_barras}")

    for barra in lista_barras:
        for nombre_dl, dl in [("precios", dataloaders_precios), ("residuos", dataloaders_residuos)]:
            if barra not in dl:
                print(f"ERROR: '{barra}' no encontrada en dataloaders_{nombre_dl}.")
                sys.exit(1)

    cargar = _cargar_modelo_ln if tipo == "LN" else _cargar_modelo_dyt

    # ─── Verificar checkpoints antes de empezar ───────────────────────────────
    print("\n" + "=" * 80)
    print("VERIFICANDO CHECKPOINTS")
    print("=" * 80)
    for barra in lista_barras:
        print(f"\n[{barra}]")
        try:
            if not args.solo_residuos:
                m = cargar(barra, exp_precios, carpeta_mod); del m
            if not args.solo_precios:
                m = cargar(barra, exp_residuos, carpeta_mod); del m
        except Exception as e:
            print(f"  ERROR verificando {barra}: {e}")
            traceback.print_exc()
            sys.exit(1)
    import gc; gc.collect()
    print("\nTodos los checkpoints verificados OK.")

    # ─── Evaluar barra a barra (un modelo a la vez para no saturar la RAM) ───
    from Modulos.Evaluacion_TFT import (
        extraer_todas_predicciones_modelo,
        calcular_metricas_predicciones,
        crear_grafico_evaluacion,
        crear_tabla_resumen,
    )

    print("\n" + "=" * 80)
    print("INICIANDO EVALUACIÓN")
    print("=" * 80)

    t_inicio = time.time()
    datos_evaluacion = {}

    for barra in tqdm(lista_barras, desc="Evaluando barras", unit="barra"):
        if args.verbose:
            tqdm.write(f"\nEvaluando: {barra}")

        # Cargar → inferencia → liberar: modelo de precios
        todas_precios = None
        if not args.solo_residuos:
            modelo_p = cargar(barra, exp_precios, carpeta_mod)
            todas_precios = extraer_todas_predicciones_modelo(
                barra, modelo_p, dataloaders_precios[barra]['test'],
                datasets_norm, diccionario_scalers,
                'y_real', None, args.horizonte, args.umbral_desfase,
            )
            del modelo_p; gc.collect()

        # Cargar → inferencia → liberar: modelo de residuos
        todas_residuos = None
        if not args.solo_precios:
            modelo_r = cargar(barra, exp_residuos, carpeta_mod)
            todas_residuos = extraer_todas_predicciones_modelo(
                barra, modelo_r, dataloaders_residuos[barra]['test'],
                datasets_norm, diccionario_scalers,
                'residuo', None, args.horizonte, args.umbral_desfase,
            )
            del modelo_r; gc.collect()

        datos_evaluacion[barra] = {}
        for conjunto in ['train', 'val', 'test']:
            datos_evaluacion[barra][conjunto] = {}
            if todas_precios is not None:
                dp = todas_precios[conjunto]
                mp = calcular_metricas_predicciones(
                    dp['y_real'], dp['y_pred'],
                    y_tft_reales=dp['y_tft_reales'],
                    umbral_desfase=args.umbral_desfase,
                    verbose=args.verbose,
                )
                datos_evaluacion[barra][conjunto]['Precios'] = {
                    'datos': dp, 'metricas': mp,
                    'n_validos': dp['n_validos'], 'n_total': dp['n_total'],
                }
            if todas_residuos is not None:
                dr = todas_residuos[conjunto]
                mr = calcular_metricas_predicciones(
                    dr['y_real'], dr['y_pred'],
                    y_tft_reales=dr['y_tft_reales'],
                    umbral_desfase=args.umbral_desfase,
                    verbose=args.verbose,
                )
                datos_evaluacion[barra][conjunto]['Residuos'] = {
                    'datos': dr, 'metricas': mr,
                    'n_validos': dr['n_validos'], 'n_total': dr['n_total'],
                }

        if todas_precios is not None and todas_residuos is not None:
            crear_grafico_evaluacion(
                barra,
                datos_evaluacion[barra]['test']['Precios']['datos'],
                datos_evaluacion[barra]['test']['Residuos']['datos'],
                datos_evaluacion[barra]['test']['Precios']['metricas'],
                datos_evaluacion[barra]['test']['Residuos']['metricas'],
                nombre_modelo=nombre_mod,
                horizonte=args.horizonte,
                carpeta=args.carpeta_graficos,
                mostrar=False,
                verbose=args.verbose,
            )
        gc.collect()

    import torch; torch.cuda.empty_cache()
    df_metricas = crear_tabla_resumen(datos_evaluacion)

    duracion = (time.time() - t_inicio) / 60

    # ─── Guardar resultados ───────────────────────────────────────────────────
    ruta_pkl = os.path.join(carpeta_res, f"eval_{nombre_mod}.pkl")
    with open(ruta_pkl, "wb") as f:
        pickle.dump((datos_evaluacion, df_metricas), f)

    ruta_csv = os.path.join(carpeta_res, f"metricas_{nombre_mod}.csv")
    df_metricas.to_csv(ruta_csv, index=False)

    print("\n" + "=" * 80)
    print("EVALUACIÓN COMPLETADA")
    print("=" * 80)
    print(f"\n  Barras evaluadas: {len(lista_barras)}")
    print(f"  Duración:         {duracion:.1f} min")
    print(f"\n  Resultados pkl:   {ruta_pkl}")
    print(f"  Métricas CSV:     {ruta_csv}")
    if args.carpeta_graficos:
        print(f"  Gráficos PNG:     {args.carpeta_graficos}/")
    print()
    print(df_metricas.to_string(index=False))


if __name__ == "__main__":
    main()
