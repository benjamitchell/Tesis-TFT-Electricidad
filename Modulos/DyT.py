import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from torch import nn

# Implementamos DyT extraido desde github
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

# Función para contar LayerNorm y DynamicTanh
def count_layers(module):
    ln_count = 0
    dyt_count = 0
    
    # Recorrer el módulo de forma recursiva
    for child in module.children():
        ln_sub, dyt_sub = count_layers(child)
        ln_count += ln_sub
        dyt_count += dyt_sub
    
    # Verificar la capa actual
    if isinstance(module, nn.LayerNorm):
        ln_count += 1
    elif isinstance(module, DynamicTanh):
        dyt_count += 1
        
    return ln_count, dyt_count


def entrenar_y_guardar_tft_dyt(
    modelo,
    train_dl,
    val_dl,
    barra,
    tipo_modelo,
    tft_config,
    early_stop_config,
    epochs,
    timestamp,
    carpeta_modelos,
    carpeta_logs,
    verbose=True,
):
    carpeta_exp = os.path.join(carpeta_modelos, tipo_modelo, barra)
    os.makedirs(carpeta_exp, exist_ok=True)

    carpeta_logs_exp = os.path.join(carpeta_logs, tipo_modelo, barra)
    os.makedirs(carpeta_logs_exp, exist_ok=True)

    nombre_checkpoint = f"{tipo_modelo}_{barra}_best"
    checkpoint_callback = ModelCheckpoint(
        dirpath=carpeta_exp,
        filename=nombre_checkpoint,
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=False,
        verbose=verbose,
    )
    ruta_ckpt = os.path.join(carpeta_exp, f"{nombre_checkpoint}.ckpt")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=early_stop_config.get("min_delta", 1e-4),
            patience=early_stop_config.get("patience", 10),
            verbose=verbose,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        checkpoint_callback,
    ]

    logger_tb = TensorBoardLogger(save_dir=carpeta_logs, name=tipo_modelo, version=barra)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        gradient_clip_val=tft_config.get("gradient_clip_val", 0.1),
        callbacks=callbacks,
        logger=logger_tb,
        default_root_dir=carpeta_logs_exp,
        enable_model_summary=False,
        enable_progress_bar=verbose,
    )

    if not isinstance(modelo, LightningModule):
        class TFTBridge(TemporalFusionTransformer, LightningModule):
            configure_sharded_model = None
        modelo.__class__ = TFTBridge

    print(f"\nIniciando entrenamiento DyT: {tipo_modelo} - {barra}\n")
    trainer.fit(modelo, train_dataloaders=train_dl, val_dataloaders=val_dl)

    config_info = {
        "barra": barra,
        "tipo_modelo": tipo_modelo,
        "timestamp": timestamp,
        "tft_config": tft_config,
        "early_stop_config": early_stop_config,
        "epochs_trained": trainer.current_epoch,
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score)
        if hasattr(trainer, "checkpoint_callback")
        else None,
        "model_path": ruta_ckpt,
    }

    ruta_config = os.path.join(carpeta_exp, f"{tipo_modelo}_{barra}_config.json")
    with open(ruta_config, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=4, ensure_ascii=False)

    print(f"\n{'─'*80}")
    print(f"ENTRENAMIENTO DyT COMPLETADO: {tipo_modelo} - {barra}")
    print(f"Épocas entrenadas: {trainer.current_epoch}")
    if hasattr(trainer, "checkpoint_callback"):
        print(f"Mejor val_loss: {trainer.checkpoint_callback.best_model_score:.6f}")
    print(f"Modelo guardado: {ruta_ckpt}")
    print(f"Configuración guardada: {ruta_config}")
    print(f"{'─'*80}\n")

    return modelo, trainer, ruta_ckpt

# Función para analizar los valores de alpha aprendidos en el modelo DyT
def analizar_alpha_dyt(modelo_dyt, barra, verbose=True):
    
    alphas = {}
    for name, param in modelo_dyt.named_parameters():
        if 'alpha' in name.lower():
            alphas[name] = param.detach().cpu().item()
    if verbose:
        print(f"\nValores de alpha aprendidos - {barra}:")
        print(f"{'Capa':<70} {'Alpha':>8}")
        print("-" * 80)
        for name, val in alphas.items():
            print(f"{name:<70} {val:>8.4f}")
    
    return alphas

# Función para graficar los valores de alpha agrupados por módulo
def graficar_alphas_agrupados(alphas_dict, barra, figsize=(10, 6)):
    
    # Agrupar por módulo principal
    grupos = {
        'Static VSN': [],
        'Encoder VSN': [],
        'Decoder VSN': [],
        'Static Context': [],
        'Post-LSTM': [],
        'Static Enrichment': [],
        'Post-Attention': [],
        'Feed-Forward': [],
        'Pre-Output': []
    }
    
    mapeo = {
        'static_variable_selection': 'Static VSN',
        'encoder_variable_selection': 'Encoder VSN',
        'decoder_variable_selection': 'Decoder VSN',
        'static_context': 'Static Context',
        'post_lstm': 'Post-LSTM',
        'static_enrichment': 'Static Enrichment',
        'post_attn': 'Post-Attention',
        'pos_wise': 'Feed-Forward',
        'pre_output': 'Pre-Output'
    }
    
    for name, val in alphas_dict.items():
        for key, grupo in mapeo.items():
            if name.startswith(key):
                grupos[grupo].append(val)
                break
    
    # Calcular estadísticas por grupo
    nombres = []
    medias = []
    stds = []
    mins = []
    maxs = []
    
    for grupo, vals in grupos.items():
        if vals:
            nombres.append(grupo)
            medias.append(np.mean(vals))
            stds.append(np.std(vals))
            mins.append(np.min(vals))
            maxs.append(np.max(vals))
    
    # Ordenar por media
    indices = np.argsort(medias)
    nombres = [nombres[i] for i in indices]
    medias = [medias[i] for i in indices]
    stds = [stds[i] for i in indices]
    mins = [mins[i] for i in indices]
    maxs = [maxs[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = range(len(nombres))
    
    # Barras con error
    bars = ax.barh(y_pos, medias, xerr=stds,
                   color='steelblue', alpha=0.7, 
                   edgecolor='black', linewidth=0.8,
                   error_kw={'linewidth': 2, 'capsize': 5, 'capthick': 2})
    
    # Marcar min y max con puntos
    ax.scatter(mins, y_pos, color='red', zorder=5, s=50, label='Mín/Máx', marker='|')
    ax.scatter(maxs, y_pos, color='red', zorder=5, s=50, marker='|')
    
    # Referencias
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, 
               alpha=0.7, label='$\\alpha_0 = 0.5$')
    ax.axvline(x=1.0, color='blue', linestyle=':', linewidth=1.5,
               alpha=0.7, label='$\\alpha = 1.0$')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nombres, fontsize=11)
    ax.set_xlabel('Valor de $\\alpha$ aprendido', fontsize=12)
    ax.set_title(f'Parámetros $\\alpha$ de DyT agrupados por módulo\nBarra {barra}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'alpha_dyt_agrupado_{barra}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return fig

# Función para graficar los valores de alpha por barra
def graficar_alphas_combinado(alphas_por_barra, lista_barras, titulo="Parámetros α de DyT por módulo", out_path=None):
    mapeo = {
        'static_variable_selection': 'Static VSN',
        'encoder_variable_selection': 'Encoder VSN',
        'decoder_variable_selection': 'Decoder VSN',
        'static_context': 'Static Context',
        'post_lstm': 'Post-LSTM',
        'static_enrichment': 'Static Enrichment',
        'post_attn': 'Post-Attention',
        'pos_wise': 'Feed-Forward',
        'pre_output': 'Pre-Output'
    }

    n_cols = 2
    n_rows = (len(lista_barras) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5*n_rows))
    axes = axes.flatten()

    for idx, barra in enumerate(lista_barras):
        ax = axes[idx]

        if barra not in alphas_por_barra:
            ax.set_visible(False)
            continue

        # ── Agrupar por módulo (igual que graficar_alphas_agrupados) ───────
        grupos = {g: [] for g in set(mapeo.values())}
        for name, val in alphas_por_barra[barra].items():
            for key, grupo in mapeo.items():
                if name.startswith(key):
                    grupos[grupo].append(val)
                    break

        nombres, medias, stds, mins, maxs = [], [], [], [], []
        for grupo, vals in grupos.items():
            if vals:
                nombres.append(grupo)
                medias.append(np.mean(vals))
                stds.append(np.std(vals))
                mins.append(np.min(vals))
                maxs.append(np.max(vals))

        orden  = np.argsort(medias)
        nombres = [nombres[i] for i in orden]
        medias  = [medias[i] for i in orden]
        stds    = [stds[i] for i in orden]
        mins    = [mins[i] for i in orden]
        maxs    = [maxs[i] for i in orden]

        y_pos = range(len(nombres))
        ax.barh(y_pos, medias, xerr=stds, color='steelblue', alpha=0.7,
                edgecolor='black', linewidth=0.8,
                error_kw={'linewidth': 1.5, 'capsize': 4, 'capthick': 1.5})
        ax.scatter(mins, y_pos, color='red', zorder=5, s=30, marker='|')
        ax.scatter(maxs, y_pos, color='red', zorder=5, s=30, marker='|')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='$\\alpha_0=0.5$')
        ax.axvline(x=1.0, color='blue', linestyle=':', linewidth=1.2, alpha=0.7, label='$\\alpha=1.0$')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(nombres, fontsize=8)
        ax.set_xlabel('$\\alpha$ aprendido', fontsize=9)
        ax.set_title(f'Barra {barra}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    axes[0].legend(fontsize=8, loc='lower right')

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(titulo, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"✅ Guardado: {out_path}")
    plt.show()