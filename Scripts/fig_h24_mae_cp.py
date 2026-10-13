#!/usr/bin/env python3
"""Genera fig_h24_mae_ancho.pdf: MAE y ancho de Split CP (90%) promedio (5 barras)
por paso del horizonte day-ahead, desde Scripts/resultado_h24_errores_cp.csv (ya
calculado, no requiere reinferencia)."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

df = pd.read_csv("Scripts/resultado_h24_errores_cp.csv")
agg = df.groupby("h")[["MAE", "Ancho_90%", "Cobertura_90%"]].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(agg.index, agg["MAE"], marker="o", markersize=3, color="tab:red")
ax1.set_xlabel("Paso del horizonte ($h$)"); ax1.set_ylabel("MAE (USD/MWh)")
ax1.set_title("MAE por paso"); ax1.set_xticks(range(1, 25, 3))

ax2b = ax2.twinx()
ax2.plot(agg.index, agg["Ancho_90%"], marker="o", markersize=3, color="tab:blue", label="Ancho CP (90\\%)")
ax2b.plot(agg.index, agg["Cobertura_90%"], marker="s", markersize=3, color="tab:green", linestyle="--", label="Cobertura")
ax2b.axhline(0.90, color="black", linestyle=":", linewidth=1)
ax2.set_xlabel("Paso del horizonte ($h$)"); ax2.set_ylabel("Ancho medio Split CP, 90\\% (USD/MWh)", color="tab:blue")
ax2b.set_ylabel("Cobertura empírica", color="tab:green")
ax2b.set_ylim(0.85, 0.97)
ax2.set_title("Ancho y cobertura CP por paso"); ax2.set_xticks(range(1, 25, 3))
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

fig.tight_layout()
fig.savefig("Escrito/imagenes/fig_h24_mae_ancho.pdf", bbox_inches="tight")
print("Guardado: Escrito/imagenes/fig_h24_mae_ancho.pdf")
