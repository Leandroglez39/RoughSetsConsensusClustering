import os
import numpy as np
from consensus_signed import (
    rough_clustering_signed,
    save_result,
    validate_and_fix_community_folder
)

# Paths y parámetros
R_FILE = "dataConnectome/fcs_ts_DZ_63_schaefer_subc_100_resting_state.npy"
COMMUNITIES_FOLDER = "communities"
OUTPUT_FOLDER = "output_consensus_signed"
GAMMA = 0.8
ALPHA = -1.0

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

    print("\n💾 Guardando resultados...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    save_result(OUTPUT_FOLDER, "consensus_inferior.pkl", coverage_inf)
    save_result(OUTPUT_FOLDER, "consensus_superior.pkl", coverage_sup)

    print("\n✅ ¡Proceso finalizado!")

if __name__ == "__main__":
    main()
    # TODO: el consenso da igual las aproximaciones inferiores y superiores, revisar detalladamente.
