"""
Análisis de importancia de variables para modelos TFT (LN y DyT).

CONFIGURACIÓN:
    - arquitectura:       "LN" o "DyT"
    - lista_experimentos: ["Multi-TFT_Precios", "Multi-TFT_Residuos"]
    - lista_barras:       ["ATACAMA"], ["CHARRUA"], etc.

ORDEN DE PROCESAMIENTO: experimento → barra
    (recorre todas las barras de Precios, luego todas las de Residuos)

SALIDA (por experimento y barra):
    - Consola: importancia numérica por variable
    - PNG:     gráficos de importancia y atención
    - CSV:     valores numéricos guardados en archivo
    - Carpeta: Resultados/Features/{experimento}/{barra}/
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import gc

sys.path.append(r'/home/minas01/BMitchell/')
from Modulos.TFT_Model import cargar_modelo_entrenado
from Modulos.Parche import activar_dyt_mode, desactivar_dyt_mode

# ── Configuración ──────────────────────────────────────────────────────────
freq               = "h"
max_pred           = 1
max_encoder        = 168
arquitectura       = "LN"               # "LN" o "DyT"
guardar_graficos   = True
lista_experimentos = ["Multi-TFT_Precios", "Multi-TFT_Residuos"]
lista_barras       = ['ATACAMA', 'CARDONES', 'CHARRUA', 'CRUCERO', 
                    'P.AZUCAR', 'P.MONTT', 'QUILLOTA', 'TARAPACA']

carpeta_modelos = (f"/home/minas01/BMitchell/"
    f"Multi-Modelos_TFT/{freq}/{arquitectura}/"
    f"pred_MAE_prophet_clima/"
)

carpeta_salida_base = os.path.join(carpeta_modelos, "Resultados", "Features")

# ── Análisis: experimento → barra ──────────────────────────────────────────
for experimento in lista_experimentos:
    print(f"\n{'#'*60}")
    print(f"#  {experimento}")
    print(f"{'#'*60}")

    # Cargar dataloader correcto según experimento
    dl_file = "dataloaders_precios.pkl" if "Precios" in experimento else "dataloaders_residuos.pkl"
    with open(f'{carpeta_modelos}/{dl_file}', 'rb') as f:
        dataloaders = pickle.load(f)

    for barra in lista_barras:
        print(f"\n{'='*60}")
        print(f"  {barra} — {experimento}")
        print(f"{'='*60}")

        carpeta_salida = os.path.join(carpeta_salida_base, experimento, barra)
        if guardar_graficos:
            os.makedirs(carpeta_salida, exist_ok=True)

        if arquitectura == "DyT":
            activar_dyt_mode()
        try:
            modelo, _ = cargar_modelo_entrenado(
                barra, experimento, carpeta_modelos=carpeta_modelos
            )
        finally:
            if arquitectura == "DyT":
                desactivar_dyt_mode()
        if modelo is None:
            print(f"  [SKIP] No se encontró modelo para {barra}")
            continue
        modelo.eval()

        raw_predictions = modelo.predict(
            dataloaders[barra]['test'],
            mode="raw",
            return_x=True
        )

        interpretation = modelo.interpret_output(
            raw_predictions.output,
            reduction="mean"
        )

        # ── Extraer nombres y valores ──────────────────────────────────────
        # El orden del tensor sigue: unknown_reals + known_reals (encoder)
        # y known_reals (decoder) — debe coincidir con plot_interpretation
        enc_names = list(modelo.hparams.get("time_varying_reals_encoder", []))
        dec_names = list(modelo.hparams.get("time_varying_reals_decoder", []))

        enc_vals = interpretation["encoder_variables"]
        dec_vals = interpretation["decoder_variables"]

        if enc_vals.dim() > 1:
            enc_vals = enc_vals.mean(0)
        if dec_vals.dim() > 1:
            dec_vals = dec_vals.mean(0)

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
        figs = modelo.plot_interpretation(interpretation)
        for name, fig in figs.items():
            fig.suptitle(
                f"TFT_{arquitectura} — {barra} — {experimento} — {name}",
                fontsize=13, fontweight='bold'
            )
            plt.tight_layout()
            if guardar_graficos:
                fname = os.path.join(
                    carpeta_salida,
                    f"{arquitectura}_{experimento}_{barra}_{name}.png"
                )
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                print(f"  Guardado: {fname}")
            plt.close(fig)

        # ── Liberar memoria ──────────────────────────────────────────────────
        del modelo, raw_predictions, interpretation
        gc.collect()

print("\nAnálisis completado.")