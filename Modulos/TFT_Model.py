# Al inicio de TFT_Model.py, ANTES de import pytorch_lightning as pl

import os
import warnings
import logging

# Configurar variables de entorno ANTES de importar PL
os.environ['PL_DISABLE_FORK_WARNING'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configurar logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# AHORA importar PyTorch Lightning
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from datetime import datetime
import matplotlib.pyplot as plt
import torch, os, logging, warnings, json

# ============================ ENTRENAMIENTO DEL MODELO TFT ============================

# Función para crear, configurar y entrenar el modelo TFT
def tft_model(dataset, barra, train_dl, val_dl, 
              tft_config, early_stop_config,
              nombre_experimento='TFT',
              epochs=100,
              gradient_clip_val=0.1,
              carpeta_modelos='Modelos_TFT',
              carpeta_logs='Logs_TFT',
              guardar_mejor=True,
              guardar_ultimo=False,
              guardar_top_k=1,
              mostrar_summary=True,
              verbose=True):
    
    # Creamos el modelo
    model = TemporalFusionTransformer.from_dataset(dataset, 
                                                   learning_rate=tft_config.get('lr', 0.001),
                                                   hidden_size=tft_config.get('hidden_size', 64),
                                                   attention_head_size=tft_config.get('heads', 4),
                                                   dropout=tft_config.get('dropout', 0.2),
                                                   hidden_continuous_size=tft_config.get('cont_size', 16),
                                                   loss=QuantileLoss(),
                                                   log_interval=10,
                                                   reduce_on_plateau_patience=tft_config.get('patience_lr', 5))            
    
    # Parche de identidad (evita RuntimeError)
    if not isinstance(model, LightningModule):
        class TFTBridge(TemporalFusionTransformer, LightningModule):
            configure_sharded_model = None 
        model.__class__ = TFTBridge

    # Configuramos carpetas
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Carpeta para este experimento
    carpeta_exp = os.path.join(carpeta_modelos, nombre_experimento, barra)
    os.makedirs(carpeta_exp, exist_ok=True)

    # Carpeta para logs
    carpeta_logs_exp = os.path.join(carpeta_logs, nombre_experimento, barra)
    os.makedirs(carpeta_logs_exp, exist_ok=True)

    callbacks = []
    
    # Early Stopping
    early_stop = EarlyStopping(monitor="val_loss",
                               min_delta=early_stop_config.get('min_delta', 1e-4),
                               patience=early_stop_config.get('patience', 10),
                               verbose=verbose,
                               mode="min")
    callbacks.append(early_stop)

    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Model Checkpoint - Mejor modelo
    ruta_mejor_modelo = None
    if guardar_mejor or guardar_top_k > 0:
        nombre_checkpoint = f"{nombre_experimento}_{barra}_best"
        
        checkpoint_callback = ModelCheckpoint(dirpath=carpeta_exp,
                                              filename=nombre_checkpoint,
                                              monitor='val_loss',
                                              save_top_k=guardar_top_k if guardar_top_k > 0 else 1,
                                              mode='min',
                                              save_last=guardar_ultimo,
                                              verbose=verbose)
        callbacks.append(checkpoint_callback)
        ruta_mejor_modelo = os.path.join(carpeta_exp, f"{nombre_checkpoint}.ckpt")
    
    # Crear ambos loggers
    logger_tb = TensorBoardLogger(save_dir=carpeta_logs, name=nombre_experimento, version=barra)
    #logger_csv = CSVLogger(save_dir=carpeta_logs, name=nombre_experimento, version=barra)

    # Configuramos el trainer
    trainer = pl.Trainer(max_epochs=epochs,
                         accelerator="auto",
                         gradient_clip_val=gradient_clip_val,
                         callbacks=callbacks,
                         logger=logger_tb,
                         default_root_dir=carpeta_logs_exp,
                         enable_model_summary=mostrar_summary,
                         enable_progress_bar=verbose)

    # Detectar si existe checkpoint previo para reanudar
    ckpt_para_reanudar = None
    if ruta_mejor_modelo and os.path.exists(ruta_mejor_modelo):
        ckpt_para_reanudar = ruta_mejor_modelo
        print(f"\nCheckpoint encontrado, reanudando desde: {ruta_mejor_modelo}")
    else:
        print(f"\nNo se encontró checkpoint previo, entrenando desde cero.")

    # Entrenamos
    print(f"\nIniciando entrenamiento: {nombre_experimento} - {barra}\n")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl,
                ckpt_path=ckpt_para_reanudar)
 
    # Guardamos configuración y detalles del entrenamiento
    if guardar_mejor or guardar_top_k > 0:

        # Guardar configuración
        config_info = {'barra': barra,
                       'experimento': nombre_experimento,
                       'timestamp': timestamp,
                       'tft_config': tft_config,
                       'early_stop_config': early_stop_config,
                       'epochs_trained': trainer.current_epoch,
                       'best_val_loss': float(trainer.checkpoint_callback.best_model_score) if hasattr(trainer, 'checkpoint_callback') else None,
                       'model_path': ruta_mejor_modelo}
        
        ruta_config = os.path.join(carpeta_exp, f"{nombre_experimento}_{barra}_config.json")
        with open(ruta_config, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=4, ensure_ascii=False)
        
        print(f"\n{'─'*80}")
        print(f"ENTRENAMIENTO COMPLETADO: {nombre_experimento} - {barra}")
        print(f"Épocas de entrenamiento: {trainer.current_epoch}")
        if hasattr(trainer, 'checkpoint_callback'):
            print(f"Mejor val_loss: {trainer.checkpoint_callback.best_model_score:.6f}")
        print(f"Modelo guardado: {ruta_mejor_modelo}")
        print(f"Configuración guardada: {ruta_config}")
        print(f"{'─'*80}\n")
    
    return model, trainer, ruta_mejor_modelo

# Función para entrenar múltiples modelos TFT
def entrenar_multiple_tft(datasets_dict, dataloaders_dict, barras, 
                         tft_config, early_stop_config,
                         nombre_experimento='TFT',
                         epochs=100,
                         carpeta_modelos='Modelos_TFT',
                         carpeta_logs='Logs_TFT',
                         verbose=True):
    
    modelos = {}
    n_barras = len(barras)
    
    print(f"\n{'='*80}")
    print(f"ENTRENAMIENTO MÚLTIPLE TFT: {nombre_experimento}")
    print(f"Barras: {n_barras}")
    print(f"{'='*80}\n")
    
    for i, barra in enumerate(barras, 1):
        print(f"\n{'─'*80}")
        print(f"[{i}/{n_barras}] Entrenando: {barra}")
        print(f"{'─'*80}")
        
        try:
            dataset = datasets_dict[barra]
            train_dl = dataloaders_dict[barra]['train']
            val_dl = dataloaders_dict[barra]['val']
            
            model, trainer, ruta = tft_model(
                dataset=dataset,
                barra=barra,
                train_dl=train_dl,
                val_dl=val_dl,
                tft_config=tft_config,
                early_stop_config=early_stop_config,
                nombre_experimento=nombre_experimento,
                epochs=epochs,
                carpeta_modelos=carpeta_modelos,
                carpeta_logs=carpeta_logs,
                verbose=verbose
            )
            
            modelos[barra] = (model, trainer, ruta)
            print(f"{barra} completado\n")
            
        except Exception as e:
            print(f"Error en {barra}: {e}\n")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Modelos entrenados: {len(modelos)}/{n_barras}")
    print(f"{'='*80}\n")
    
    return modelos

# ============================ CARGA Y CURVA DE APRENDIZAJE ======================

# Redefinimos el parche de identidad
class TFTBridge(TemporalFusionTransformer, LightningModule):
    configure_sharded_model = None 

# Función para cargar un modelo TFT entrenado desde un checkpoint
def cargar_modelo_entrenado(barra, nombre_experimento, carpeta_modelos='Modelos_TFT', device='cpu'):

    # Construimos las rutas
    carpeta_exp = os.path.join(carpeta_modelos, nombre_experimento, barra)
    ruta_config = os.path.join(carpeta_exp, f"{nombre_experimento}_{barra}_config.json")
    
    # Cargamos la configuración
    if not os.path.exists(ruta_config):
        print(f"No se encontró el archivo de configuración: {ruta_config}")
        return None, None
    
    try:
        with open(ruta_config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except UnicodeDecodeError:
        print("Advertencia: Error de codificación UTF-8, intentando con 'latin-1'...")
        with open(ruta_config, 'r', encoding='latin-1') as f:
            config = json.load(f)
    
    ruta_checkpoint = config.get('model_path')
    
    # Cargamos el modelo desde el checkpoint
    if not os.path.exists(ruta_checkpoint):
        print(f"No se encontró el checkpoint: {ruta_checkpoint}")
        return None, None

    # Al cargar checkpoints de GPU en máquinas CPU-only, torchmetrics intenta crear tensores
    # en 'cuda:0' (su device guardado) para detectar cambios de dispositivo, lo que lanza
    # AssertionError. Parcheamos torch.zeros para redirigir esas llamadas a CPU.
    _orig_zeros = torch.zeros
    def _safe_zeros(*args, **kwargs):
        d = kwargs.get('device')
        if d is not None and 'cuda' in str(d):
            kwargs['device'] = 'cpu'
        return _orig_zeros(*args, **kwargs)

    torch.zeros = _safe_zeros
    try:
        best_tft = TFTBridge.load_from_checkpoint(
            ruta_checkpoint,
            map_location=lambda storage, loc: storage,
            weights_only=False
        )
    finally:
        torch.zeros = _orig_zeros
    
    # Ponemos el modelo en modo evaluación
    best_tft.eval()
    
    print(f"Modelo para {barra} cargado desde: {ruta_checkpoint}")
    return best_tft, config

# Función
def buscar_tfevents(carpeta_base, max_profundidad=3):
    
    # Buscamos el archivo tfevents más reciente en la carpeta y subcarpetas
    def buscar_recursivo(carpeta, profundidad):
        if profundidad > max_profundidad or not os.path.exists(carpeta):
            return []
        
        archivos_encontrados = []
        
        try:
            for item in os.listdir(carpeta):
                ruta = os.path.join(carpeta, item)
                
                if os.path.isdir(ruta):
                    archivos_encontrados.extend(buscar_recursivo(ruta, profundidad + 1))
                elif item.startswith('events.out.tfevents'):
                    archivos_encontrados.append(ruta)
        except PermissionError:
            pass
        
        return archivos_encontrados
    
    archivos = buscar_recursivo(carpeta_base, 0)
    
    if not archivos:
        return None
    
    # Extraer timestamp del nombre y retornar el más reciente
    def extraer_timestamp(ruta):
        
        nombre = os.path.basename(ruta)
        partes = nombre.split('.')
        try:
            return int(partes[3])
        except (IndexError, ValueError):
            
            # Fallback a fecha de modificación si no se puede parsear
            return os.path.getmtime(ruta)
    
    return max(archivos, key=extraer_timestamp)


def cargar_losses_tensorboard(carpeta_log, verbose=False):
    
    if not os.path.exists(carpeta_log):
        if verbose:
            print(f"No existe: {carpeta_log}")
        return [], [], [], []
    
    # Buscar archivo tfevents más reciente
    eventos_path = buscar_tfevents(carpeta_log)
    
    if not eventos_path:
        if verbose:
            print(f"No se encontraron archivos tfevents")
        return [], [], [], []
    
    if verbose:
        archivo = os.path.basename(eventos_path)
        tamaño_mb = os.path.getsize(eventos_path) / (1024 * 1024)
        print(f"    Leyendo: {archivo} ({tamaño_mb:.1f} MB)")
    
    try:
        # Cargar eventos de TensorBoard
        ea = EventAccumulator(eventos_path)
        ea.Reload()
        
        tags = ea.Tags().get('scalars', [])
        
        # Buscar train_loss_epoch o train_loss
        tag_train = None
        for buscar in ['train_loss_epoch', 'train_loss']:
            for t in tags:
                if buscar in t.lower():
                    tag_train = t
                    break
            if tag_train:
                break
        
        # Si no encontró ninguno, buscar cualquier tag con 'train' y 'loss'
        if not tag_train:
            for t in tags:
                if 'train' in t.lower() and 'loss' in t.lower():
                    tag_train = t
                    break
        
        # Buscar val_loss
        tag_val = None
        for t in tags:
            if 'val_loss' in t.lower():
                tag_val = t
                break
        
        # Extraer train_loss y convertir steps a épocas
        epochs_train, train_loss = [], []
        if tag_train:
            events = ea.Scalars(tag_train)

            # Usar índice secuencial en lugar de steps
            epochs_train = list(range(len(events)))
            train_loss = [e.value for e in events]
            
            if verbose:
                print(f"    Train: {tag_train} | {len(train_loss)} épocas")
        
        # Extraer val_loss y convertir steps a épocas
        epochs_val, val_loss = [], []
        if tag_val:
            events = ea.Scalars(tag_val)

            # Usar índice secuencial en lugar de steps
            epochs_val = list(range(len(events)))
            val_loss = [e.value for e in events]
            
            if verbose:
                print(f"    Val: {tag_val} | {len(val_loss)} épocas")
        
        return epochs_train, train_loss, epochs_val, val_loss
        
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        return [], [], [], []
    
# Función para graficar curvas de aprendizaje
def grafico_losses(lista_barras, logs_dir, 
                   experimento_precios='Multi-TFT_Precios', 
                   experimento_residuos='Multi-TFT_Residuos',
                   nombre_modelo='TFT'):

    
    n_barras = len(lista_barras)
    n_filas = 4 #n_barras // 2

    fig, axs = plt.subplots(n_filas, 2, figsize=(16, 5 * n_filas), sharey=True)
    
    # Si solo hay 1 fila, axs es array 1D, sino es 2D
    if n_filas == 1:
        axs = axs.reshape(1, -1)
    
    for idx, barra in enumerate(lista_barras):
        
        print(f"{'─'*70}")
        print(f"{barra}")
        print('─'*70)
        
        # Cargar datos de Precios
        carpeta_precios = os.path.join(logs_dir, experimento_precios, barra)
        print(f"Precios:")
        epochs_train_p, train_loss_p, epochs_val_p, val_loss_p = cargar_losses_tensorboard(carpeta_precios, verbose=True)
        
        # Cargar datos de Residuos
        carpeta_residuos = os.path.join(logs_dir, experimento_residuos, barra)
        print(f"Residuos:")
        epochs_train_r, train_loss_r, epochs_val_r, val_loss_r = cargar_losses_tensorboard(carpeta_residuos, verbose=True)
        
        # Calcular posición en grid 2D
        fila = idx // 2
        col = idx % 2
        ax = axs[fila, col]
        
        # Train losses
        if train_loss_p:
            ax.plot(epochs_train_p, train_loss_p, label="Train Precios", 
                   color='blue', lw=2, linestyle='--', alpha=0.6)
        
        if train_loss_r:
            ax.plot(epochs_train_r, train_loss_r, label="Train Residuos", 
                   color='green', lw=1.8, linestyle='--', alpha=0.6)
        
        # Val losses con marcadores de mejor época
        if val_loss_p:
            ax.plot(epochs_val_p, val_loss_p, label="Val Precios", 
                   color='orange', lw=2.5)
            
            # Marcar mejor val_loss
            best_val_p = min(val_loss_p)
            best_epoch_p = epochs_val_p[val_loss_p.index(best_val_p)]
            ax.axvline(best_epoch_p, color='orange', linestyle=':', alpha=0.5, lw=1)
            ax.text(best_epoch_p, ax.get_ylim()[1] * 0.95, 
                   f'Best P: {best_val_p:.3f} (ep {best_epoch_p})', 
                   fontsize=9, ha='center', color='orange',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        if val_loss_r:
            ax.plot(epochs_val_r, val_loss_r, label="Val Residuos", 
                   color='red', lw=2)
            
            # Marcar mejor val_loss
            best_val_r = min(val_loss_r)
            best_epoch_r = epochs_val_r[val_loss_r.index(best_val_r)]
            ax.axvline(best_epoch_r, color='red', linestyle=':', alpha=0.5, lw=1)
            ax.text(best_epoch_r, ax.get_ylim()[1] * 0.85, 
                   f'Best R: {best_val_r:.3f} (ep {best_epoch_r})', 
                   fontsize=9, ha='center', color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Formato
        ax.set_title(f"{barra}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Época", fontsize=11)
        if col == 0:
            ax.set_ylabel("Loss", fontsize=11)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.suptitle(f"Curvas de Aprendizaje {nombre_modelo}", 
                fontsize=16, fontweight='bold', y=1.00)
    plt.show()