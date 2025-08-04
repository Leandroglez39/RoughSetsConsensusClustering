import os
import numpy as np
from consensus_signed import (
    find_overlapping_nodes,
    rough_clustering_signed,
    save_overlapping_to_txt,
    save_result,
    validate_and_fix_community_folder,
    
)

from consensus_visualization import plot_consensus_graph

# Paths y parámetros
R_FILE = "dataConnectome/fcs_ts_DZ_63_schaefer_subc_100_resting_state.npy"
COMMUNITIES_FOLDER = "communities"
OUTPUT_FOLDER = "output_consensus_signed"
GAMMA = 0.5
ALPHA = -0.25

def main():
    print("🧠 Cargando matrices R...")
    R_all = np.load(R_FILE)
    if R_all.ndim != 3:
        raise ValueError("El archivo .npy debe tener shape (T, N, N)")
    T, N, _ = R_all.shape
    print(f"✔ {T} matrices de correlación cargadas con dimensión {N}x{N}")

    print("\n🔍 Verificando y corrigiendo archivos de comunidades .npy...")
    communities = validate_and_fix_community_folder(COMMUNITIES_FOLDER)
    print(f"✔ {len(communities)} particiones cargadas correctamente")

    print("\n⚙️ Ejecutando consenso (Rough Clustering firmado)...")
    coverage_inf, coverage_sup = rough_clustering_signed(
        R_all=R_all,
        communities=communities,
        gamma=GAMMA,
        alpha=ALPHA,
        verbose=True
    )

    plot_consensus_graph(coverage_inf, coverage_sup, title="Consenso de comunidades firmadas", output_path=os.path.join(OUTPUT_FOLDER, "consenso_visual.png"))

    print("\n💾 Guardando resultados...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Encontrar nodos solapados (en la diferencia entre superior e inferior)
    overlapping = find_overlapping_nodes(coverage_inf, coverage_sup)
    save_overlapping_to_txt(overlapping, os.path.join(OUTPUT_FOLDER, "nodos_superpuestos.txt"))

    # Preparar consenso como diccionario: comunidad -> {'core': set, 'overlap': set}
    consensus_dict = {}
    for idx, (inf, sup) in enumerate(zip(coverage_inf, coverage_sup)):
        core = set(inf)
        overlap = set(sup) - set(inf)
        consensus_dict[f"comunidad_{idx}"] = {
            "core": sorted(core),
            "overlap": sorted(overlap)
        }

    # Guardar consenso en txt legible
    consensus_txt_path = os.path.join(OUTPUT_FOLDER, "consenso_comunidades.txt")
    with open(consensus_txt_path, "w") as f:
        for label, parts in consensus_dict.items():
            f.write(f"{label}:\n")
            f.write(f"  Núcleo: {parts['core']}\n")
            f.write(f"  Solapados: {parts['overlap']}\n\n")

    # Guardar cubrimientos originales para uso posterior
    save_result(OUTPUT_FOLDER, "consensus_inferior.pkl", coverage_inf)
    save_result(OUTPUT_FOLDER, "consensus_superior.pkl", coverage_sup)

    print("\n✅ ¡Proceso finalizado!")

if __name__ == "__main__":
    main()
    # TODO: el consenso da igual las aproximaciones inferiores y superiores, revisar detalladamente.
