from Modulos.DyT import DynamicTanh, count_layers
import torch.nn as nn
from lightning.pytorch import LightningModule
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss

# Guardamos la clase LayerNorm original
_OriginalLayerNorm = nn.LayerNorm

class DyTLayerNorm(nn.Module):
    """
    Wrapper que se comporta como LayerNorm pero usa DynamicTanh (DyT).
    """
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, 
                 device=None, dtype=None, alpha_init_value=0.5):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        elif isinstance(normalized_shape, list):
            normalized_shape = tuple(normalized_shape)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        self.dyt = DynamicTanh(
            normalized_shape=normalized_shape[0] if len(normalized_shape) == 1 else normalized_shape,
            channels_last=True,
            alpha_init_value=alpha_init_value
        )
    
    def forward(self, x):
        return self.dyt(x)
    
    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, eps={self.eps} (DyT wrapper)"


# Activar Monkey Patching: nn.LayerNorm → DyTLayerNorm
def activar_dyt_mode():
    nn.LayerNorm = DyTLayerNorm

# Desactivar Monkey Patching: restaura nn.LayerNorm original
def desactivar_dyt_mode():
    nn.LayerNorm = _OriginalLayerNorm

# Función para crear un TFT con DyT en vez de LayerNorm
def crear_tft_con_dyt(training_dataset, tft_config, verbose=True):
    
    # Activar monkey patching
    activar_dyt_mode()
    
    try:
        # Instanciar TFT
        tft_dyt = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=tft_config['lr'],
            hidden_size=tft_config['hidden_size'],
            attention_head_size=tft_config['heads'],
            dropout=tft_config['dropout'],
            hidden_continuous_size=tft_config['cont_size'],
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=tft_config['patience_lr'])
        
        if verbose:
            print(f"Modelo instanciado")
        
    except Exception as e:
        print(f"\nERROR al crear TFT: {e}")
        raise
    
    finally:
        # Desactivar Monkey Patching
        desactivar_dyt_mode()
    
    # Verificar resultados
    is_lightning = isinstance(tft_dyt, LightningModule)
    ln_count, dyt_count = count_layers(tft_dyt)
    params = sum(p.numel() for p in tft_dyt.parameters())
    

    if verbose:
        print(f"\nVerificación:")
        print(f"   LightningModule: {is_lightning}")
        print(f"   LayerNorm: {ln_count}")
        print(f"   DyT: {dyt_count}")
        print(f"   Parámetros: {params:,}")
    
    # Validación
    if not is_lightning:
        print(f"\n   ERROR: NO es LightningModule")
        raise TypeError("El modelo no es LightningModule")
    
    if ln_count > 0:
        print(f"\n   ADVERTENCIA: Quedan {ln_count} LayerNorm sin convertir")
    
    if dyt_count == 0:
        print(f"\n   ERROR: No se crearon capas DyT")
        raise ValueError("No se crearon capas DyT")
    
    if ln_count == 0 and dyt_count > 0:
        print(f"\n   Conversión exitosa: {dyt_count} capas DyT creadas")
    
    return tft_dyt