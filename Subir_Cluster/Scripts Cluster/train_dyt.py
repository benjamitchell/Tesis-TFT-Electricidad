#!/usr/bin/env python3
"""
train_tft_dyt.py - Script para entrenar modelos TFT-DyT en el cluster del CMM (leftaru).

Carga dataloaders y configuraciones preprocesadas y ejecuta el entrenamiento
de modelos TFT con Dynamic Tanh (DyT) de precios y residuos sin necesidad
del Jupyter Notebook.

Uso:
    python train_tft_dyt.py [opciones]

Ejemplo:
    python train_tft_dyt.py \\
        --carpeta-modelos Multi-Modelos_TFT/D/DyT/pred_1_90_Features_Prophet \\
        --carpeta-logs Logs_TFT/D/DyT/pred_1_90_Features_Prophet \\
        --barras ATACAMA CARDONES CHARRUA \\
        --verbose

Archivos requeridos (generados en el notebook local):
    - dataloaders_precios.pkl
    - dataloaders_residuos.pkl
    - tft_config.json
    - feature_config.json  (opcional)
    - scalers.pkl          (opcional, no usado durante el entrenamiento)
"""

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
_search_dir = _script_dir
while True:
    if os.path.isdir(os.path.join(_search_dir, "Modulos")):
        break
    _parent = os.path.dirname(_search_dir)
    if _parent == _search_dir:
        print(
            "ERROR: No se encontró la carpeta 'Modulos/' partiendo desde"
            f" '{_script_dir}'.\n"
            "Asegúrate de ejecutar el script desde la carpeta 'src/' del"
            " proyecto, o agrega su ruta al PYTHONPATH."
        )
        sys.exit(1)
    _search_dir = _parent
sys.path.insert(0, _search_dir)

from Modulos.Parche import crear_tft_con_dyt  # noqa: E402
from Modulos.DyT import entrenar_y_guardar_tft_dyt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena modelos TFT-DyT de precios y residuos para el cluster del CMM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Carpetas
    parser.add_argument(
        "--carpeta-modelos",
        default="Multi-Modelos_TFT/D/DyT/pred_1_90_Features_Prophet",
        help=(
            "Carpeta donde están los archivos de configuración y donde se guardarán"
            " los modelos (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--carpeta-logs",
        default=None,
        help=(
            "Carpeta para guardar los logs. Por defecto se deriva de --carpeta-modelos"
            " reemplazando 'Multi-Modelos_TFT' por 'Logs_TFT'."
        ),
    )

    # Rutas de archivos de entrada
    parser.add_argument(
        "--dataloaders-precios",
        default=None,
        help=(
            "Ruta al archivo dataloaders_precios.pkl"
            " (default: <carpeta-modelos>/dataloaders_precios.pkl)"
        ),
    )
    parser.add_argument(
        "--dataloaders-residuos",
        default=None,
        help=(
            "Ruta al archivo dataloaders_residuos.pkl"
            " (default: <carpeta-modelos>/dataloaders_residuos.pkl)"
        ),
    )
    parser.add_argument(
        "--tft-config",
        default=None,
        help=(
            "Ruta al archivo tft_config.json"
            " (default: <carpeta-modelos>/tft_config.json)"
        ),
    )
    parser.add_argument(
        "--feature-config",
        default=None,
        help=(
            "Ruta al archivo feature_config.json"
            " (default: <carpeta-modelos>/feature_config.json)"
        ),
    )

    # Selección de barras y modelos
    parser.add_argument(
        "--barras",
        nargs="+",
        default=None,
        metavar="BARRA",
        help=(
            "Barras a entrenar (p.ej. --barras ATACAMA CARDONES)."
            " Por defecto: todas las barras en los dataloaders."
        ),
    )
    parser.add_argument(
        "--solo-precios",
        action="store_true",
        help="Entrenar solo modelos de precios (omite residuos).",
    )
    parser.add_argument(
        "--solo-residuos",
        action="store_true",
        help="Entrenar solo modelos de residuos (omite precios).",
    )

    # Reproducibilidad
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Semilla para pytorch_lightning.seed_everything (afecta init de pesos,"
            " shuffling de train, etc.). Si no se entrega, no se fija ninguna semilla"
            " (comportamiento por defecto de antes, no determinista)."
        ),
    )

    # Salida
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar progreso detallado durante el entrenamiento.",
    )

    return parser.parse_args()


def _cargar_pickle(ruta, descripcion="archivo"):
    """Carga un archivo pickle y devuelve su contenido."""
    print(f"Cargando {descripcion} desde: {ruta}")
    with open(ruta, "rb") as f:
        contenido = pickle.load(f)
    return contenido


def _cargar_json(ruta, descripcion="archivo"):
    """Carga un archivo JSON y devuelve su contenido."""
    print(f"Cargando {descripcion} desde: {ruta}")
    with open(ruta, "r") as f:
        contenido = json.load(f)
    return contenido


def _verificar_archivos(*rutas):
    """Verifica que todos los archivos requeridos existan."""
    faltantes = [r for r in rutas if not os.path.exists(r)]
    if faltantes:
        print("\nERROR: No se encontraron los siguientes archivos requeridos:")
        for r in faltantes:
            print(f"  - {r}")
        sys.exit(1)


def _derivar_carpeta_logs(carpeta_modelos):
    """Deriva la carpeta de logs a partir de la carpeta de modelos."""
    return carpeta_modelos.replace("Multi-Modelos_TFT", "Logs_TFT", 1)


def _entrenar_multiples_dyt(
    datasets_dict,
    dataloaders_dict,
    barras,
    tft_config,
    early_stop_config,
    tipo_modelo,
    epochs,
    timestamp,
    carpeta_modelos,
    carpeta_logs,
    verbose,
):
    """
    Crea y entrena un modelo TFT-DyT para cada barra.

    Equivalente a ``entrenar_multiple_tft`` de TFT_Model.py pero usando
    monkey-patching DyT (Parche.py) para la creación del modelo y
    ``entrenar_y_guardar_tft_dyt`` (DyT.py) para el entrenamiento.
    """
    modelos = {}
    n_barras = len(barras)

    print(f"\n{'='*80}")
    print(f"ENTRENAMIENTO MÚLTIPLE TFT-DyT: {tipo_modelo}")
    print(f"Barras: {n_barras}")
    print(f"{'='*80}\n")

    for i, barra in enumerate(barras, 1):
        print(f"\n{'─'*80}")
        print(f"[{i}/{n_barras}] Entrenando: {barra}")
        print(f"{'─'*80}")

        try:
            training_dataset = datasets_dict[barra]
            train_dl = dataloaders_dict[barra]["train"]
            val_dl = dataloaders_dict[barra]["val"]

            # Crear modelo TFT con DyT en lugar de LayerNorm
            modelo = crear_tft_con_dyt(training_dataset, tft_config, verbose=verbose)

            # Entrenar y guardar
            modelo, trainer, ruta_ckpt = entrenar_y_guardar_tft_dyt(
                modelo=modelo,
                train_dl=train_dl,
                val_dl=val_dl,
                barra=barra,
                tipo_modelo=tipo_modelo,
                tft_config=tft_config,
                early_stop_config=early_stop_config,
                epochs=epochs,
                timestamp=timestamp,
                carpeta_modelos=carpeta_modelos,
                carpeta_logs=carpeta_logs,
                verbose=verbose,
            )

            modelos[barra] = (modelo, trainer, ruta_ckpt)
            print(f"{barra} completado\n")

        except Exception as e:
            print(f"Error en {barra}: {e}\n")
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Modelos entrenados: {len(modelos)}/{n_barras}")
    print(f"{'='*80}\n")

    return modelos


def main():
    args = parse_args()

    if args.solo_precios and args.solo_residuos:
        print("ERROR: No puedes usar --solo-precios y --solo-residuos al mismo tiempo.")
        sys.exit(1)

    if args.seed is not None:
        import pytorch_lightning as pl
        pl.seed_everything(args.seed, workers=True)
        print(f"\nSemilla fijada: {args.seed}")

    carpeta_modelos = args.carpeta_modelos
    carpeta_logs = args.carpeta_logs or _derivar_carpeta_logs(carpeta_modelos)

    # Rutas de archivos con valores por defecto
    ruta_dl_precios = args.dataloaders_precios or os.path.join(
        carpeta_modelos, "dataloaders_precios.pkl"
    )
    ruta_dl_residuos = args.dataloaders_residuos or os.path.join(
        carpeta_modelos, "dataloaders_residuos.pkl"
    )
    ruta_tft_config = args.tft_config or os.path.join(carpeta_modelos, "tft_config.json")
    ruta_feature_config = args.feature_config or os.path.join(
        carpeta_modelos, "feature_config.json"
    )

    # ─── Validar archivos requeridos ─────────────────────────────────────────
    archivos_requeridos = [ruta_dl_precios, ruta_dl_residuos, ruta_tft_config]
    _verificar_archivos(*archivos_requeridos)

    # ─── Cargar configuraciones ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CARGANDO CONFIGURACIONES")
    print("=" * 80)

    tft_config_total = _cargar_json(ruta_tft_config, "tft_config.json")
    tft_config = tft_config_total["tft_config"]
    early_stop_config = tft_config_total["early_stop_config"]
    epochs = tft_config_total["epochs"]

    print(f"\nConfiguración TFT:")
    for k, v in tft_config.items():
        print(f"  {k}: {v}")
    print(f"\nEarly stopping:")
    for k, v in early_stop_config.items():
        print(f"  {k}: {v}")
    print(f"\nÉpocas máximas: {epochs}")

    # Cargar feature_config si existe (opcional)
    lista_barras_config = None
    if os.path.exists(ruta_feature_config):
        feature_config = _cargar_json(ruta_feature_config, "feature_config.json")
        lista_barras_config = feature_config.get("lista_barras")
    else:
        print(f"\nAVISO: No se encontró feature_config.json en {ruta_feature_config}")

    # ─── Cargar dataloaders ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CARGANDO DATALOADERS")
    print("=" * 80)

    dataloaders_precios = _cargar_pickle(ruta_dl_precios, "dataloaders_precios.pkl")
    print(f"  Barras disponibles (precios): {list(dataloaders_precios.keys())}")

    dataloaders_residuos = _cargar_pickle(ruta_dl_residuos, "dataloaders_residuos.pkl")
    print(f"  Barras disponibles (residuos): {list(dataloaders_residuos.keys())}")

    # Determinar barras a entrenar
    lista_barras = (
        args.barras
        or lista_barras_config
        or list(dataloaders_precios.keys())
    )
    print(f"\nBarras a entrenar ({len(lista_barras)}): {lista_barras}")

    # Verificar que todas las barras existen en los dataloaders y contienen 'train'/'val'
    for barra in lista_barras:
        if barra not in dataloaders_precios:
            print(f"ERROR: La barra '{barra}' no se encuentra en dataloaders_precios.")
            sys.exit(1)
        for clave in ("train", "val"):
            if clave not in dataloaders_precios[barra]:
                print(
                    f"ERROR: El dataloader de precios para la barra '{barra}' no contiene"
                    f" la clave '{clave}'. Claves disponibles:"
                    f" {list(dataloaders_precios[barra].keys())}"
                )
                sys.exit(1)
        if barra not in dataloaders_residuos:
            print(f"ERROR: La barra '{barra}' no se encuentra en dataloaders_residuos.")
            sys.exit(1)
        for clave in ("train", "val"):
            if clave not in dataloaders_residuos[barra]:
                print(
                    f"ERROR: El dataloader de residuos para la barra '{barra}' no contiene"
                    f" la clave '{clave}'. Claves disponibles:"
                    f" {list(dataloaders_residuos[barra].keys())}"
                )
                sys.exit(1)

    # Extraer datasets (TimeSeriesDataSet) desde los dataloaders
    datasets_dict_precios = {
        barra: dataloaders_precios[barra]["train"].dataset for barra in lista_barras
    }
    datasets_dict_residuos = {
        barra: dataloaders_residuos[barra]["train"].dataset for barra in lista_barras
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tiempo_total_inicio = time.time()
    modelos_precios = {}
    modelos_residuos = {}
    duracion_precios = 0.0
    duracion_residuos = 0.0

    # ─── Entrenamiento de precios ─────────────────────────────────────────────
    if not args.solo_residuos:
        print("\n" + "=" * 80)
        print("INICIANDO ENTRENAMIENTO DE PRECIOS (DyT)")
        print("=" * 80)

        tiempo_inicio_precios = time.time()
        modelos_precios = _entrenar_multiples_dyt(
            datasets_dict=datasets_dict_precios,
            dataloaders_dict=dataloaders_precios,
            barras=lista_barras,
            tft_config=tft_config,
            early_stop_config=early_stop_config,
            tipo_modelo="Precios",
            epochs=epochs,
            timestamp=timestamp,
            carpeta_modelos=carpeta_modelos,
            carpeta_logs=carpeta_logs,
            verbose=args.verbose,
        )
        duracion_precios = (time.time() - tiempo_inicio_precios) / 60
        print(
            f"ENTRENAMIENTO MODELO DE PRECIOS (DyT) | DURACIÓN: {duracion_precios:.2f} minutos"
        )

    # ─── Entrenamiento de residuos ────────────────────────────────────────────
    if not args.solo_precios:
        print("\n" + "=" * 80)
        print("INICIANDO ENTRENAMIENTO DE RESIDUOS (DyT)")
        print("=" * 80)

        tiempo_inicio_residuos = time.time()
        modelos_residuos = _entrenar_multiples_dyt(
            datasets_dict=datasets_dict_residuos,
            dataloaders_dict=dataloaders_residuos,
            barras=lista_barras,
            tft_config=tft_config,
            early_stop_config=early_stop_config,
            tipo_modelo="Residuos",
            epochs=epochs,
            timestamp=timestamp,
            carpeta_modelos=carpeta_modelos,
            carpeta_logs=carpeta_logs,
            verbose=args.verbose,
        )
        duracion_residuos = (time.time() - tiempo_inicio_residuos) / 60
        print(
            f"ENTRENAMIENTO MODELO DE RESIDUOS (DyT) | DURACIÓN: {duracion_residuos:.2f} minutos"
        )

    # ─── Resumen final ────────────────────────────────────────────────────────
    duracion_total = (time.time() - tiempo_total_inicio) / 60

    print("\n" + "=" * 80)
    print("ENTRENAMIENTO DyT COMPLETADO")
    print("=" * 80)
    print(f"\nRESUMEN:")
    print(f"  Barras entrenadas:    {len(lista_barras)}")
    print(f"  Modelos de precios:   {len(modelos_precios)}")
    print(f"  Modelos de residuos:  {len(modelos_residuos)}")
    print(f"\n  TIEMPOS:")
    print(f"    Precios:   {duracion_precios:.2f} min")
    print(f"    Residuos:  {duracion_residuos:.2f} min")
    print(f"    Total:     {duracion_total:.2f} min ({duracion_total / 60:.2f} horas)")
    print(f"\nARCHIVOS GUARDADOS:")
    print(f"  Modelos: {carpeta_modelos}/")
    print(f"  Logs:    {carpeta_logs}/")


if __name__ == "__main__":
    main()