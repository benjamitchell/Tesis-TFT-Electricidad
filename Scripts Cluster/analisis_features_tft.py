"""
Análisis de importancia de variables para modelos TFT (LN y DyT), a partir de la
Variable Selection Network: interpret_output(reduction="mean") sobre el conjunto de
prueba, en porcentaje del total del encoder y del decoder.

Sirve para cualquier carpeta de modelos (S+C o P+C, LN o DyT, 1 hora o 24 horas), en el
cluster o en local: si el checkpoint no está en la ruta guardada al entrenar, se busca
junto a su config (Modulos/TFT_Model.cargar_modelo_entrenado).

USO:
    # Sin argumentos: configuración original del cluster (P+C, LN, 8 barras, con gráficos)
    python analisis_features_tft.py

    # S+C a 24 horas, solo el modelo de precios, sin gráficos
    python "Scripts Cluster/analisis_features_tft.py" \\
        --carpeta-modelos Multi-Modelos_TFT/h/LN/pred_sol_clima_h24 \\
        --experimentos Multi-TFT_Precios --barras ATACAMA CHARRUA --sin-graficos

    # DyT (sus carpetas de experimento se llaman "Precios"/"Residuos")
    python "Scripts Cluster/analisis_features_tft.py" --arquitectura DyT \\
        --carpeta-modelos Multi-Modelos_TFT/h/DyT/pred_sol_clima --experimentos Precios

ARGUMENTOS:
    --carpeta-modelos  carpeta con dataloaders_*.pkl y las subcarpetas de experimento
    --arquitectura     LN o DyT (DyT reconstruye el modelo con Dynamic Tanh)
    --experimentos     subcarpetas de modelos a analizar (precios usa dataloaders_precios.pkl,
                       el resto dataloaders_residuos.pkl)
    --barras           barras a analizar
    --sin-graficos     solo CSV y consola, sin los PNG de plot_interpretation (más rápido)

ORDEN DE PROCESAMIENTO: experimento → barra (un modelo cargado a la vez).

SALIDA (por experimento y barra), en <carpeta-modelos>/Resultados/Features/{experimento}/{barra}/:
    - Consola: importancia numérica por variable
    - CSV:     {arquitectura}_{experimento}_{barra}_importancia.csv (tipo, variable, importancia_pct)
    - PNG:     gráficos de importancia y atención (salvo --sin-graficos)
"""

import argparse
import pickle
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sys
import os
import gc

# Modulos/ está junto al script en el cluster y un nivel arriba en el repositorio
_aqui = os.path.dirname(os.path.abspath(__file__))
sys.path += [_aqui, os.path.dirname(_aqui)]
from Modulos.TFT_Model import cargar_modelo_entrenado

# ── Configuración (por defecto, la original del cluster) ───────────────────
BARRAS = ['ATACAMA', 'CARDONES', 'CHARRUA', 'CRUCERO',
          'P.AZUCAR', 'P.MONTT', 'QUILLOTA', 'TARAPACA']

parser = argparse.ArgumentParser(description="Importancia de variables (VSN) de modelos TFT")
parser.add_argument("--carpeta-modelos",
                    default="/home/minas01/BMitchell/Multi-Modelos_TFT/h/LN/pred_MAE_prophet_clima/")
parser.add_argument("--arquitectura", choices=["LN", "DyT"], default="LN")
parser.add_argument("--experimentos", nargs="+", default=["Multi-TFT_Precios", "Multi-TFT_Residuos"])
parser.add_argument("--barras", nargs="+", default=BARRAS)
parser.add_argument("--sin-graficos", action="store_true")
args = parser.parse_args()

carpeta_modelos = args.carpeta_modelos
arquitectura = args.arquitectura
guardar_graficos = not args.sin_graficos
carpeta_salida_base = os.path.join(carpeta_modelos, "Resultados", "Features")

# ── Análisis: experimento → barra ──────────────────────────────────────────
for experimento in args.experimentos:
    print(f"\n{'#'*60}")
    print(f"#  {experimento}")
    print(f"{'#'*60}")

    # Cargar dataloader correcto según experimento
    dl_file = "dataloaders_precios.pkl" if "Precios" in experimento else "dataloaders_residuos.pkl"
    with open(os.path.join(carpeta_modelos, dl_file), 'rb') as f:
        dataloaders = pickle.load(f)

    for barra in args.barras:
        print(f"\n{'='*60}")
        print(f"  {barra} — {experimento}")
        print(f"{'='*60}")

        carpeta_salida = os.path.join(carpeta_salida_base, experimento, barra)
        os.makedirs(carpeta_salida, exist_ok=True)

        modelo, _ = cargar_modelo_entrenado(
            barra, experimento, carpeta_modelos=carpeta_modelos, dyt=arquitectura == "DyT"
        )
        if modelo is None:
            print(f"  [SKIP] No se encontró modelo para {barra}")
            continue

        # ── Importancia por lotes ───────────────────────────────────────────
        # interpret_output suma los pesos de la VSN sobre las muestras (también con
        # reduction="mean"), así que sumar lote a lote da el mismo resultado sin cargar
        # todo el test a la vez (con 24 pasos, la atención completa no cabe en memoria)
        enc_vals, dec_vals = 0, 0
        with torch.no_grad():
            for x, _ in dataloaders[barra]['test']:
                parcial = modelo.interpret_output(modelo(x), reduction="sum")
                enc_vals = enc_vals + parcial["encoder_variables"]
                dec_vals = dec_vals + parcial["decoder_variables"]

        # ── Extraer nombres y valores ──────────────────────────────────────
        # El orden del tensor sigue: unknown_reals + known_reals (encoder)
        # y known_reals (decoder) — debe coincidir con plot_interpretation
        enc_names = list(modelo.hparams.get("time_varying_reals_encoder", []))
        dec_names = list(modelo.hparams.get("time_varying_reals_decoder", []))

        enc_names = enc_names[:len(enc_vals)]
        dec_names = dec_names[:len(dec_vals)]

        enc_total = enc_vals.sum()
        dec_total = dec_vals.sum()

        # ── Imprimir en consola ─────────────────────────────────────────────
        print("\nEncoder variables:")
        for n, v in sorted(zip(enc_names, enc_vals.tolist()), key=lambda x: -x[1]):
            print(f"  {n}: {v/enc_total.item()*100:.2f}%")

        print("\nDecoder variables:")
        for n, v in sorted(zip(dec_names, dec_vals.tolist()), key=lambda x: -x[1]):
            print(f"  {n}: {v/dec_total.item()*100:.2f}%")

        # ── Guardar CSV ──────────────────────────────────────────────────────
        rows = []
        for n, v in zip(enc_names, enc_vals.tolist()):
            rows.append({
                "tipo": "encoder",
                "variable": n,
                "importancia_pct": round(v / enc_total.item() * 100, 2)
            })
        for n, v in zip(dec_names, dec_vals.tolist()):
            rows.append({
                "tipo": "decoder",
                "variable": n,
                "importancia_pct": round(v / dec_total.item() * 100, 2)
            })

        df_importancia = pd.DataFrame(rows).sort_values(
            ["tipo", "importancia_pct"], ascending=[True, False]
        )
        csv_fname = os.path.join(
            carpeta_salida,
            f"{arquitectura}_{experimento}_{barra}_importancia.csv"
        )
        df_importancia.to_csv(csv_fname, index=False)
        print(f"\n  CSV guardado: {csv_fname}")

        # ── Guardar gráficos ─────────────────────────────────────────────────
        # Los gráficos (incluida la atención) sí requieren la interpretación completa
        if guardar_graficos:
            raw_predictions = modelo.predict(dataloaders[barra]['test'], mode="raw", return_x=True)
            interpretation = modelo.interpret_output(raw_predictions.output, reduction="mean")
            figs = modelo.plot_interpretation(interpretation)
            for name, fig in figs.items():
                fig.suptitle(
                    f"TFT_{arquitectura} — {barra} — {experimento} — {name}",
                    fontsize=13, fontweight='bold'
                )
                plt.tight_layout()
                fname = os.path.join(
                    carpeta_salida,
                    f"{arquitectura}_{experimento}_{barra}_{name}.png"
                )
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                print(f"  Guardado: {fname}")
                plt.close(fig)
            del raw_predictions, interpretation

        # ── Liberar memoria ──────────────────────────────────────────────────
        del modelo
        gc.collect()

print("\nAnálisis completado.")
