import sys, os, pickle, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('.')
import warnings
warnings.filterwarnings('ignore')

from Modulos.Preprocesamiento_Transformer import crear_dataloaders, guardar_experimento

BARRAS = ['ATACAMA', 'CHARRUA', 'P.MONTT']

for variante in ['LN', 'DyT']:
    origen = f'Multi-Modelos_TFT/h/{variante}/pred_sol_clima'
    destino = f'Multi-Modelos_TFT/h/{variante}/pred_sol_clima_h24'

    with open(f'{origen}/datasets_norm.pkl', 'rb') as f:
        datasets_norm = pickle.load(f)
    with open(f'{origen}/scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    with open(f'{origen}/feature_config.json', encoding='utf-8') as f:
        feature_config = json.load(f)
    with open(f'{origen}/tft_config.json', encoding='utf-8') as f:
        tft_config_total = json.load(f)

    print(f'=== {variante} ===')
    print('known_reals:', feature_config['known_reals'])
    print('unknown_reals_price:', feature_config['unknown_reals_price'])
    print('unknown_reals_resid:', feature_config['unknown_reals_resid'])

    dataloaders_precios, dataloaders_residuos, _, _ = crear_dataloaders(
        lista_barras=BARRAS,
        datasets_norm=datasets_norm,
        known_reals=feature_config['known_reals'],
        unknown_reals_price=feature_config['unknown_reals_price'],
        unknown_reals_resid=feature_config['unknown_reals_resid'],
        max_encoder_length=feature_config['max_encoder_length'],
        max_prediction_length=24,
        batch_size=feature_config['batch_size'],
    )

    scalers_3barras = {b: scalers[b] for b in BARRAS}
    feature_config_h24 = dict(feature_config)
    feature_config_h24['max_prediction_length'] = 24
    feature_config_h24['lista_barras'] = BARRAS
    tft_config_h24 = dict(tft_config_total)
    tft_config_h24['max_prediction_length'] = 24

    guardar_experimento(destino, dataloaders_precios, dataloaders_residuos,
                         scalers_3barras, feature_config_h24, tft_config_h24)
    print(f'Guardado en: {destino}\n')

print('LISTO')
