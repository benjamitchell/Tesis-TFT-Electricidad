# Para juntar los .events 
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

# Ajusta estas rutas
archivo1 = r"C:\Users\56977\OneDrive\Escritorio\Tesis - copia\Logs_TFT\h\DyT\pred_1_168_cluster_con_clima\Residuos\P.MONTT\events.out.tfevents.1781016391.gn004.1820853.0"
archivo2 = r"C:\Users\56977\OneDrive\Escritorio\Tesis - copia\Logs_TFT\h\DyT\pred_1_168_cluster_con_clima\Residuos\P.MONTT\events.out.tfevents.1781032145.gn004.2394978.0"
carpeta_salida = r"C:\Users\56977\OneDrive\Escritorio\Tesis - copia\Logs_TFT\h\DyT\pred_1_168_cluster_con_clima\Residuos\P.MONTT\eventos_combinados"

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