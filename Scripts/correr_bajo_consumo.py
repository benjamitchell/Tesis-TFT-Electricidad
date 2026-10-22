#!/usr/bin/env python3
"""
correr_bajo_consumo.py - Corre en secuencia los scripts de inferencia a 24 horas con
prioridad baja de Windows y pocos hilos, para no saturar el computador mientras se usa.

Uso:
    python Scripts/correr_bajo_consumo.py
"""
import os
import subprocess
import sys
import time

HILOS = "2"
PASOS = [
    ["Scripts/extraer_predicciones_h24.py"],
    ["Scripts/evaluar_h24.py", "--solo-resto", "--arq=DyT"],
    ["Scripts/h24_ln_vs_dyt.py"],
    ["Scripts/h24_pc_vs_sc.py"],
]
LOG = "Scripts/correr_bajo_consumo.log"


def main():
    env = dict(os.environ, OMP_NUM_THREADS=HILOS, MKL_NUM_THREADS=HILOS, PYTHONIOENCODING="utf-8")
    prioridad = getattr(subprocess, "IDLE_PRIORITY_CLASS", 0)
    with open(LOG, "w", encoding="utf-8") as log:
        for paso in PASOS:
            t0 = time.time()
            log.write(f"\n##### {' '.join(paso)}\n")
            log.flush()
            r = subprocess.run([sys.executable] + paso, env=env, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=prioridad)
            log.write(f"##### fin {paso[0]}: codigo {r.returncode}, {(time.time() - t0) / 60:.1f} min\n")
            log.flush()
            if r.returncode != 0:
                break


if __name__ == "__main__":
    main()
