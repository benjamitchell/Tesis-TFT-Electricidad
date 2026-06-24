"""
Verificación completa: todos los modelos en Multi-Modelos_TFT tienen su
log de TensorBoard correspondiente en Logs_TFT.

Criterio de correspondencia: al menos un segmento tfevents del log
contiene un min(val_loss) que coincide con el best_val_loss del
config.json (tolerancia 1e-4). Esto maneja correctamente los
entrenamientos reanudados (varios tfevents en la misma carpeta).

Salida:
  [OK]  -> log correcto encontrado
  [!!]  -> no existe ningún archivo tfevents
  [??]  -> hay logs pero ninguno coincide con best_val_loss
"""
import json
import pathlib
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = pathlib.Path(r"C:\Users\56977\OneDrive\Escritorio\Tesis - copia")
TOL  = 1e-4

EXCLUIR = {"_NO_SIRVE", "_LISTO"}


def min_val_loss(events_path):
    ea = EventAccumulator(str(events_path))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    tag  = next((t for t in tags if "val_loss" in t.lower()), None)
    if not tag:
        return None
    return min(e.value for e in ea.Scalars(tag))


configs = sorted(
    p for p in (BASE / "Multi-Modelos_TFT").rglob("*_config.json")
    if p.name not in ("feature_config.json", "tft_config.json")
    and not any(ex in str(p) for ex in EXCLUIR)
)

# contadores globales
total = ok = sin_logs = no_match = 0

grupo_actual = ""
for cfg_path in configs:
    parts = cfg_path.parts
    # estructura esperada:
    # ...\Multi-Modelos_TFT \ freq \ arch \ exp_config \ exp_folder \ barra \ file
    try:
        idx = parts.index("Multi-Modelos_TFT")
        freq, arch, exp_config, barra = parts[idx+1], parts[idx+2], parts[idx+3], parts[idx+5]
    except (ValueError, IndexError):
        continue

    grupo = f"{freq}/{arch}/{exp_config}"

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    experimento = cfg.get("experimento") or cfg.get("tipo_modelo", "")
    best_val    = cfg.get("best_val_loss")
    epochs      = cfg.get("epochs_trained")
    ts_str      = cfg.get("timestamp", "")

    if not experimento or best_val is None:
        continue

    total += 1

    if grupo != grupo_actual:
        grupo_actual = grupo
        print(f"\n{'#'*70}")
        print(f"# {grupo}")
        print(f"{'#'*70}")

    log_dir  = BASE / "Logs_TFT" / freq / arch / exp_config / experimento / barra
    archivos = sorted(log_dir.glob("events.out.tfevents.*")) if log_dir.exists() else []

    if not archivos:
        sin_logs += 1
        print(f"  {barra:12s} [!!] SIN LOGS   best_val={best_val:.6f}  epochs={epochs}  ts={ts_str}")
        continue

    match = False
    mins  = []
    for a in archivos:
        mv = min_val_loss(a)
        if mv is not None:
            mins.append(round(mv, 6))
            if abs(mv - best_val) < TOL:
                match = True

    if match:
        ok += 1
        print(f"  {barra:12s} [OK]  best_val={best_val:.6f}  mins={mins}")
    else:
        no_match += 1
        print(f"  {barra:12s} [??]  best_val={best_val:.6f}  mins={mins}")

print(f"\n{'='*70}")
print(f"RESUMEN: {total} modelos revisados")
print(f"  [OK] {ok:3d}  — log correcto")
print(f"  [!!] {sin_logs:3d}  — sin logs descargados")
print(f"  [??] {no_match:3d}  — logs no coinciden con checkpoint")
print(f"{'='*70}")
