import os
import re
import pandas as pd

def parse_consenso_txt(path):
    comunidades = []
    with open(path, 'r') as f:
        content = f.read()
    bloques = re.split(r'comunidad_\d+:', content)[1:]
    for idx, bloque in enumerate(bloques):
        core = re.search(r'Núcleo: \[([^\]]*)\]', bloque)
        overlap = re.search(r'Solapados: \[([^\]]*)\]', bloque)
        core_list = [int(x) for x in core.group(1).split(',') if x.strip()] if core else []
        overlap_list = [int(x) for x in overlap.group(1).split(',') if x.strip()] if overlap else []
        comunidades.append({'comunidad': idx, 'core': core_list, 'overlap': overlap_list})
    return comunidades

def parse_superpuestos_txt(path):
    nodos = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('Nodo'):
                parts = re.findall(r'\d+', line)
                nodo = int(parts[0])
                comunidades = [int(x) for x in parts[1:]]
                nodos[nodo] = comunidades
    return nodos

def extract_params_from_folder(folder_path):
    # folder_path: output_experiments_signed_100/gamma_0.5_alpha_-0.25
    base = os.path.basename(os.path.dirname(folder_path))
    m_gran = re.search(r'output_experiments_signed_(\d+)', base)
    granularidad = int(m_gran.group(1)) if m_gran else None

    m = re.search(r'gamma_([\d\.]+)_alpha_(-?[\d\.]+)', os.path.basename(folder_path))
    gamma = float(m.group(1)) if m else None
    alpha = float(m.group(2)) if m else None

    return granularidad, gamma, alpha

def resumen_consenso(consenso_path, superpuestos_path, granularidad, gamma, alpha, total_nodos):
    comunidades = parse_consenso_txt(consenso_path)
    nodos_solapados = parse_superpuestos_txt(superpuestos_path)
    k_grupos = len(comunidades)
    vertices_superior = set()
    for c in comunidades:
        vertices_superior.update(c['core'])
        vertices_superior.update(c['overlap'])
    n_superior = len(vertices_superior)
    n_solapados = len(nodos_solapados)
    porc_superior = n_superior / total_nodos if total_nodos else 0
    porc_solapados = n_solapados / total_nodos if total_nodos else 0
    prom_grupos_solapados = sum(len(v) for v in nodos_solapados.values()) / n_solapados if n_solapados else 0
    card_max = max(len(c['core']) + len(c['overlap']) for c in comunidades)
    card_min = min(len(c['core']) + len(c['overlap']) for c in comunidades)
    return {
        'granularidad': granularidad,
        'gamma': gamma,
        'alpha': alpha,
        'k_grupos': k_grupos,
        'n_superior': n_superior,
        'n_solapados': n_solapados,
        'porc_superior': porc_superior,
        'porc_solapados': porc_solapados,
        'prom_grupos_solapados': prom_grupos_solapados,
        'card_max': card_max,
        'card_min': card_min
    }

def unificar_resultados_to_excel(result_dirs, total_nodos_dict, output_excel_path):
    resumenes = []
    for d in result_dirs:
        granularidad, gamma, alpha = extract_params_from_folder(d)
        total_nodos = total_nodos_dict.get(granularidad, None)
        consenso_path = os.path.join(d, 'consenso_comunidades.txt')
        superpuestos_path = os.path.join(d, 'nodos_superpuestos.txt')
        if not (os.path.exists(consenso_path) and os.path.exists(superpuestos_path) and total_nodos):
            print(f"Saltando carpeta {d} por falta de archivos o total_nodos")
            continue
        resumen = resumen_consenso(consenso_path, superpuestos_path, granularidad, gamma, alpha, total_nodos)
        resumenes.append(resumen)
    df = pd.DataFrame(resumenes)
    df.to_excel(output_excel_path, index=False)

if __name__ == "__main__":
    # Procesa todas las carpetas base
    import glob
    base_dirs = [
        "output_experiments_signed_100",
        "output_experiments_signed_200",
        "output_experiments_signed_300"
    ]
    result_dirs = []
    for base_dir in base_dirs:
        result_dirs.extend([d for d in glob.glob(f"{base_dir}/gamma_*_alpha_*") if os.path.isdir(d)])
    # Define el número total de nodos por granularidad (ajusta según tu caso)
    total_nodos_dict = {
        100: 114,   # ejemplo
        300: 314,   # ejemplo
        400: 414    # ejemplo
    }
    output_excel_path = "resumen_consensos.xlsx"
    print(f"Procesando {len(result_dirs)} carpetas de resultados...")
    unificar_resultados_to_excel(result_dirs, total_nodos_dict, output_excel_path)
    print(f"Archivo Excel generado: {output_excel_path}")
