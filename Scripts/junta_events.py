# Para juntar los .events 
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

# Ajusta estas rutas
archivo1 = r"C:\Users\56977\OneDrive\Escritorio\Tesis - copia\Logs_TFT\h\LN\pred_1_168_sin_features\Multi-TFT_Residuos\TARAPACA\events.out.tfevents.1782519163.gn004.1816987.0"
archivo2 = r"C:\Users\56977\OneDrive\Escritorio\Tesis - copia\Logs_TFT\h\LN\pred_1_168_sin_features\Multi-TFT_Residuos\TARAPACA\events.out.tfevents.1782534194.gn005.2334035.0"
carpeta_salida = r"C:\Users\56977\OneDrive\Escritorio\Tesis - copia\Logs_TFT\h\LN\pred_1_168_sin_features\Multi-TFT_Residuos\TARAPACA\eventos_combinados"

ea1 = EventAccumulator(archivo1)
ea1.Reload()
ea2 = EventAccumulator(archivo2)
ea2.Reload()

writer = SummaryWriter(carpeta_salida)

for tag in ea1.Tags()["scalars"]:
    for event in ea1.Scalars(tag):
        writer.add_scalar(tag, event.value, event.step)

for tag in ea2.Tags()["scalars"]:
    for event in ea2.Scalars(tag):
        writer.add_scalar(tag, event.value, event.step)

writer.close()
print(f"Listo! Logs combinados en: {carpeta_salida}")