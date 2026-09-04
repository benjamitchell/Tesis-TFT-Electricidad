import pickle

with open("C:\\Users\\56977\\OneDrive\\Escritorio\\Tesis - copia\\Multi-Modelos_TFT\\h\\LN\\pred_1_168_con_sol\\dataloaders_precios.pkl", "rb") as f:
    dl = pickle.load(f)

# Ver las barras disponibles
print("Barras:", list(dl.keys()))

# Ver features de la primera barra
barra = list(dl.keys())[0]
dataset = dl[barra]["train"].dataset

print(f"\nFeatures de '{barra}':")
print("  time_varying_known_reals:  ", dataset.time_varying_known_reals)
print("  time_varying_unknown_reals:", dataset.time_varying_unknown_reals)
print("  static_reals:              ", dataset.static_reals)
print("  time_varying_known_cats:   ", dataset.time_varying_known_categoricals)
print("  static_cats:               ", dataset.static_categoricals)