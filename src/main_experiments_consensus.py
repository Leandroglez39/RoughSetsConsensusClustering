# main_experiments_consensus.py

import os
import numpy as np
from consensus_signed import (
    validate_and_fix_community_folder,
    rough_clustering_signed,
    find_overlapping_nodes,
    save_overlapping_to_txt,
    save_result
)

# === CONFIGURACIÓN GENERAL ===
R_FILE = "dataConnectome/fcs_ts_DZ_63_schaefer_subc_100_resting_state.npy"
COMMUNITIES_FOLDER = "communities"
OUTPUT_BASE = "output_experiments_signed"

# === PARÁMETROS A PROBAR ===
GAMMA_VALUES = [0.3, 0.5, 0.7]
ALPHA_VALUES = [-0.1, -0.25, -0.5]

def main():
    print("📥 Cargando matrices R...")
    R_all = np.load(R_FILE)
    if R_all.ndim != 3:
        raise ValueError("El archivo .npy debe tener shape (T, N, N)")
    T, N, _ = R_all.shape
    print(f"✔ {T} matrices con tamaño {N}x{N}")

    print("📦 Cargando particiones de comunidades...")
    communities = validate_and_fix_community_folder(COMMUNITIES_FOLDER)
    print(f"✔ {len(communities)} particiones cargadas")

    # === EXPERIMENTACIÓN ===
    for gamma in GAMMA_VALUES:
        for alpha in ALPHA_VALUES:
            print(f"\n🚀 Ejecutando experimento con GAMMA={gamma}, ALPHA={alpha}")

            # Salida específica para esta combinación
            out_folder = os.path.join(OUTPUT_BASE, f"gamma_{gamma}_alpha_{alpha}")
            os.makedirs(out_folder, exist_ok=True)

            # Consenso
            coverage_inf, coverage_sup = rough_clustering_signed(
                R_all=R_all,
                communities=communities,
                gamma=gamma,
                alpha=alpha,
                verbose=True
            )

            # Nodos solapados
            overlapping = find_overlapping_nodes(coverage_inf, coverage_sup)
            save_overlapping_to_txt(overlapping, os.path.join(out_folder, "nodos_superpuestos.txt"))

            # Guardar consenso legible
            consensus_dict = {}
            for idx, (inf, sup) in enumerate(zip(coverage_inf, coverage_sup)):
                core = set(inf)
                overlap = set(sup) - set(inf)
                consensus_dict[f"comunidad_{idx}"] = {
                    "core": sorted(core),
                    "overlap": sorted(overlap)
                }

            with open(os.path.join(out_folder, "consenso_comunidades.txt"), "w") as f:
                for label, parts in consensus_dict.items():
                    f.write(f"{label}:\n")
                    f.write(f"  Núcleo: {parts['core']}\n")
                    f.write(f"  Solapados: {parts['overlap']}\n\n")

            # Guardar cubrimientos internos
            save_result(out_folder, "coverage_inf.pkl", coverage_inf)
            save_result(out_folder, "coverage_sup.pkl", coverage_sup)

            print(f"✅ Resultados guardados en: {out_folder}")

if __name__ == "__main__":
    main()
