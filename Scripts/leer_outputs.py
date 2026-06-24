import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open(r'C:\Users\56977\OneDrive\Escritorio\Tesis - copia\Multi_modelo TFT Horario.ipynb', encoding='utf-8'))

# Mostrar outputs de las ultimas 8 celdas
for i, cell in enumerate(nb['cells'][-8:], start=len(nb['cells'])-7):
    src_raw = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    src_preview = src_raw[:80].replace('\n', ' ')
    outputs = cell.get('outputs', [])
    print(f"\n=== Celda {i} [{cell['cell_type']}] ===")
    print(f"  Source: {src_preview}...")
    if not outputs:
        print("  (sin outputs)")
    for out in outputs:
        otype = out.get('output_type', '')
        if otype == 'stream':
            text = ''.join(out.get('text', []))
            print(f"  [stream] {text[:500]}")
        elif otype in ('display_data', 'execute_result'):
            data = out.get('data', {})
            if 'text/plain' in data:
                txt = ''.join(data['text/plain'])
                print(f"  [display] {txt[:800]}")
            if 'text/html' in data:
                html = ''.join(data['text/html'])
                # extraer texto plano del html de forma simple
                import re
                plain = re.sub(r'<[^>]+>', ' ', html)
                plain = re.sub(r'\s+', ' ', plain).strip()
                print(f"  [html] {plain[:800]}")
        elif otype == 'error':
            print(f"  [ERROR] {out.get('ename','')}: {out.get('evalue','')}")
