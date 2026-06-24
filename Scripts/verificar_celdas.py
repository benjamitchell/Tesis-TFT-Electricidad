import json
nb = json.load(open(r'C:\Users\56977\OneDrive\Escritorio\Tesis - copia\Multi_modelo TFT Horario.ipynb', encoding='utf-8'))
for c in nb['cells'][-7:]:
    src = c['source']
    preview = src[:90].replace('\n', ' ') if src else '(vacio)'
    print(f"  [{c['cell_type'][:4]}] id={c['id']}  -> {preview}")
