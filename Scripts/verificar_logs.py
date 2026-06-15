"""
Verifica que los logs de TensorBoard descargados (Logs_TFT) correspondan
a los checkpoints guardados (Multi-Modelos_TFT).

Para cada barra/arquitectura/experimento compara:
  - best_val_loss y epochs_trained del *_config.json (guardado al terminar
    el entrenamiento que generó el checkpoint)
  - min(val_loss) y nº de épocas del log de TensorBoard
  - timestamp del config.json vs timestamp embebido en el nombre del tfevents

Si min(val_loss) y nº de épocas del log coinciden con el config -> [OK]
Si no coinciden (o no hay log) -> [??] / [!!] -> ese log NO corresponde
al checkpoint guardado (o falta descargarlo).
"""
import os
import json
import glob
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = r"C:\Users\56977\OneDrive\Escritorio\Tesis - copia"

# (arquitectura, carpeta_modelos, carpeta_logs, {carpeta_exp_modelos: carpeta_exp_logs})
CONFIGURACIONES = [
    ("LN", "Multi-Modelos_TFT/h/LN/pred_1_168_cluster_con_clima",
            "Logs_TFT/h/LN/pred_1_168_cluster_con_clima",
            {"Multi-TFT_Precios": "Multi-TFT_Precios",
             "Multi-TFT_Residuos": "Multi-TFT_Residuos"}),
    ("DyT", "Multi-Modelos_TFT/h/DyT/pred_1_168_cluster_con_clima",
            "Logs_TFT/h/DyT/pred_1_168_cluster_con_clima",
            {"Precios": "Precios",
             "Residuos": "Residuos"}),
]

lista_barras = ['ATACAMA', 'CARDONES', 'CHARRUA', 'CRUCERO',
                'P.AZUCAR', 'P.MONTT', 'QUILLOTA', 'TARAPACA']

TOL_LOSS = 1e-4


def cargar_val_loss(eventos_path):
    ea = EventAccumulator(eventos_path)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    tag_val = next((t for t in tags if 'val_loss' in t.lower()), None)
    if not tag_val:
        return None, None
    vals = [e.value for e in ea.Scalars(tag_val)]
    return min(vals), len(vals)


for arquitectura, carpeta_modelos_rel, carpeta_logs_rel, experimentos in CONFIGURACIONES:
    carpeta_modelos = os.path.join(BASE, carpeta_modelos_rel)
    carpeta_logs = os.path.join(BASE, carpeta_logs_rel)

    for exp_modelos, exp_logs in experimentos.items():
        print(f"\n{'#'*70}")
        print(f"# {arquitectura} - {exp_modelos}")
        print(f"{'#'*70}")

        for barra in lista_barras:
            config_path = os.path.join(
                carpeta_modelos, exp_modelos, barra, f"{exp_modelos}_{barra}_config.json"
            )
            log_dir = os.path.join(carpeta_logs, exp_logs, barra)

            if not os.path.exists(config_path):
                continue  # esta barra no fue entrenada con esta config

            with open(config_path, encoding='utf-8') as f:
                config = json.load(f)

            ts_config = datetime.strptime(config['timestamp'], "%Y%m%d_%H%M%S")
            best_val_config = config['best_val_loss']
            epochs_config = config['epochs_trained']

            print(f"\n{barra}:")
            print(f"  Checkpoint -> timestamp: {ts_config} | best_val_loss: {best_val_config:.6f} | epochs: {epochs_config}")

            archivos = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
            if not archivos:
                print("  [!!] No hay ningún log descargado para esta barra")
                continue

            for archivo in archivos:
                nombre = os.path.basename(archivo)
                try:
                    ts_log = int(nombre.split('.')[3])
                    ts_log_dt = datetime.fromtimestamp(ts_log)
                except (IndexError, ValueError):
                    ts_log_dt = None

                best_val_log, n_epochs_log = cargar_val_loss(archivo)

                if best_val_log is None:
                    print(f"  [!!] {nombre}")
                    print(f"        inicio log: {ts_log_dt} -- sin tag 'val_loss'")
                    continue

                match_loss = abs(best_val_log - best_val_config) < TOL_LOSS
                match_epochs = abs(n_epochs_log - epochs_config) <= 1

                estado = "OK" if (match_loss and match_epochs) else "??"
                print(f"  [{estado}] {nombre}")
                print(f"        inicio log: {ts_log_dt}")
                print(f"        min(val_loss) log: {best_val_log:.6f} (checkpoint: {best_val_config:.6f})")
                print(f"        nº épocas log:     {n_epochs_log} (checkpoint: {epochs_config})")

print("\nListo. [OK] = el log corresponde al checkpoint guardado.")
print("       [??] = el log NO coincide (otro run / falta descargar el correcto).")
print("       [!!] = no se encontró log o no tiene la métrica val_loss.")
