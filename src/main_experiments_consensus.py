# main_experiments_consensus.py

import os
import numpy as np
from consensus_signed import (
    build_match_array,
    validate_and_fix_community_folder,
    rough_clustering_signed,
    find_overlapping_nodes,
    save_overlapping_to_txt,
    save_result
)
from consensus_visualization import (
    build_fuzzy_matrix_with_repeats, 
    plot_consensus_graph, 
    export_consensus_to_gephi_gexf, 
    plot_consensus_quality_metrics, 
    plot_fuzzy_matrix,  
    plot_structured_consensus_matrix,
    plot_community_evolution_across_gamma  # ← NUEVA IMPORTACIÓN
)
import pandas as pd
import re

# === CONFIGURACIÓN GENERAL ===
R_FILE = "dataConnectome/fcs_ts_DZ_63_schaefer_subc_400_resting_state.npy"
COMMUNITIES_FOLDER = "communities/granularity_400/"
OUTPUT_BASE = "output_experiments_signed_400"

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

    # === EXPERIMENTACIÓN POR ALPHA ===
    for alpha in ALPHA_VALUES:
        print(f"\n{'='*70}")
        print(f"🔬 INICIANDO EXPERIMENTOS PARA ALPHA = {alpha}")
        print(f"{'='*70}")
        
        # Diccionario para almacenar resultados de cada gamma (para visualización de evolución)
        consensus_results_for_alpha = []
        
        for gamma in GAMMA_VALUES:
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
            
            # Guardar para la visualización de evolución
            consensus_results_for_alpha.append((coverage_inf, coverage_sup))

            # ============= VISUALIZACIONES INDIVIDUALES =============
            print("📊 Generando visualizaciones de matriz de consenso...")
            
            # Construir matriz de consenso
            match_array = build_match_array(communities, N)
            
            # Matriz fuzzy con repeticiones
            W, extended_nodes, node_roles, community_boundaries, community_labels = build_fuzzy_matrix_with_repeats(
                match_array=match_array,
                coverage_inf=coverage_inf,
                coverage_sup=coverage_sup,
                w_core=1.0,
                w_overlap_base=0.6
            )

            plot_fuzzy_matrix(
                W,
                extended_nodes,
                community_boundaries,
                community_labels,
                gamma=gamma,
                alpha=alpha,
                output_path=os.path.join(out_folder, "fuzzy_matrix_community.png")
            )

            # Métricas de calidad del consenso
            plot_consensus_quality_metrics(
                match_array, coverage_inf, coverage_sup,
                gamma, alpha,
                output_path=os.path.join(out_folder, "consensus_quality_metrics.png")
            )

            # Grafo de consenso
            plot_consensus_graph(
                coverage_inf, coverage_sup, 
                title=f"Consenso firmado (γ={gamma}, α={alpha})", 
                output_path=os.path.join(out_folder, "consenso_visual.png")
            )
            
            export_consensus_to_gephi_gexf(
                coverage_inf, coverage_sup, 
                output_path=os.path.join(out_folder, "consenso.gexf")
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
        
        # ============= VISUALIZACIÓN DE EVOLUCIÓN POR GAMMA =============
        # ⚠️ ESTO VA AQUÍ, FUERA DEL BUCLE DE GAMMA ⚠️
        print(f"\n📊 Generando visualización de evolución por alpha (α={alpha}) y los gamma (γ={GAMMA_VALUES})...")

        evolution_folder = os.path.join(OUTPUT_BASE, f"evolution_gamma_{GAMMA_VALUES}_alpha_{alpha}")
        os.makedirs(evolution_folder, exist_ok=True)
        
        plot_community_evolution_across_gamma(
            gamma_values=GAMMA_VALUES,
            consensus_results=consensus_results_for_alpha,
            output_path=os.path.join(evolution_folder, f"evolution__gamma_{GAMMA_VALUES}_alpha_{alpha}.png"),
            title=f"Evolución de Comunidades según Gamma (α={GAMMA_VALUES})",
            figsize=(16, 12),
            reverse_gamma=True  # Mostrar de más riguroso a más flexible
        )
        
        print(f"✅ Visualización de evolución guardada en: {evolution_folder}")
    
    print(f"\n{'='*70}")
    print("🎉 TODOS LOS EXPERIMENTOS COMPLETADOS")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()




# # main_experiments_consensus.py

# import os
# import numpy as np
# from consensus_signed import (
#     build_match_array,
#     validate_and_fix_community_folder,
#     rough_clustering_signed,
#     find_overlapping_nodes,
#     save_overlapping_to_txt,
#     save_result
# )
# from consensus_visualization import build_fuzzy_matrix_with_repeats, plot_consensus_graph, export_consensus_to_gephi_gexf, plot_consensus_quality_metrics, plot_fuzzy_matrix,  plot_structured_consensus_matrix
# import pandas as pd
# import re

# # === CONFIGURACIÓN GENERAL ===
# R_FILE = "dataConnectome/fcs_ts_DZ_63_schaefer_subc_400_resting_state.npy"
# COMMUNITIES_FOLDER = "communities/granularity_400/"
# OUTPUT_BASE = "output_experiments_signed"

# # === PARÁMETROS A PROBAR ===
# # GAMMA_VALUES = [0.3]
# # ALPHA_VALUES = [-0.1]

# GAMMA_VALUES = [0.3, 0.5, 0.7]
# ALPHA_VALUES = [-0.1, -0.25, -0.5]

# def main():
#     print("📥 Cargando matrices R...")
#     R_all = np.load(R_FILE)
#     if R_all.ndim != 3:
#         raise ValueError("El archivo .npy debe tener shape (T, N, N)")
#     T, N, _ = R_all.shape
#     print(f"✔ {T} matrices con tamaño {N}x{N}")

#     print("📦 Cargando particiones de comunidades...")
#     communities = validate_and_fix_community_folder(COMMUNITIES_FOLDER)
#     print(f"✔ {len(communities)} particiones cargadas")

#     # === EXPERIMENTACIÓN ===
#     for gamma in GAMMA_VALUES:
#         for alpha in ALPHA_VALUES:
#             print(f"\n🚀 Ejecutando experimento con GAMMA={gamma}, ALPHA={alpha}")

#             # Salida específica para esta combinación
#             out_folder = os.path.join(OUTPUT_BASE, f"gamma_{gamma}_alpha_{alpha}")
#             os.makedirs(out_folder, exist_ok=True)

#             # Consenso
#             coverage_inf, coverage_sup = rough_clustering_signed(
#                 R_all=R_all,
#                 communities=communities,
#                 gamma=gamma,
#                 alpha=alpha,
#                 verbose=True
#             )

#             # ============= AQUÍ ENCAJAN LAS NUEVAS VISUALIZACIONES =============
#             print("📊 Generando visualizaciones de matriz de consenso...")
            
#             # Construir matriz de consenso
#             match_array = build_match_array(communities, N)
            
#             # Matriz ordenada por comunidades obtenidas
#             # plot_structured_consensus_matrix(
#             #     match_array=match_array,k
#             #     coverage_inf=coverage_inf,
#             #     coverage_sup=coverage_sup,
#             #     gamma=gamma,
#             #     alpha=alpha,
#             #     output_path=os.path.join(out_folder, "consensus_matrix_structured.png"))
            
#             W, extended_nodes, node_roles, community_boundaries, community_labels = build_fuzzy_matrix_with_repeats(
#                 match_array=match_array,
#                 coverage_inf=coverage_inf,
#                 coverage_sup=coverage_sup,
#                 w_core=1.0,
#                 w_overlap_base=0.6  # puedes ajustar esto según tu sensibilidad deseada
#             )

#             plot_fuzzy_matrix(W,
#                             extended_nodes,
#                             community_boundaries,
#                             community_labels,
#                             gamma=gamma,
#                             alpha=alpha,
#                             output_path=os.path.join(out_folder, "fuzzy_matrix_community.png"))


            

#             # Métricas de calidad del consenso
#             plot_consensus_quality_metrics(
#                 match_array, coverage_inf, coverage_sup,
#                 gamma, alpha,
#                 output_path=os.path.join(out_folder, "consensus_quality_metrics.png")
#             )

#             plot_consensus_graph(coverage_inf, coverage_sup, title=f"Consenso firmado (γ={gamma}, α={alpha})", output_path=os.path.join(out_folder, "consenso_visual.png"))
#             export_consensus_to_gephi_gexf(coverage_inf, coverage_sup, output_path=os.path.join(out_folder, "consenso.gexf"))

#             # Nodos solapados
#             overlapping = find_overlapping_nodes(coverage_inf, coverage_sup)
#             save_overlapping_to_txt(overlapping, os.path.join(out_folder, "nodos_superpuestos.txt"))

#             # Guardar consenso legible
#             consensus_dict = {}
#             for idx, (inf, sup) in enumerate(zip(coverage_inf, coverage_sup)):
#                 core = set(inf)
#                 overlap = set(sup) - set(inf)
#                 consensus_dict[f"comunidad_{idx}"] = {
#                     "core": sorted(core),
#                     "overlap": sorted(overlap)
#                 }

#             with open(os.path.join(out_folder, "consenso_comunidades.txt"), "w") as f:
#                 for label, parts in consensus_dict.items():
#                     f.write(f"{label}:\n")
#                     f.write(f"  Núcleo: {parts['core']}\n")
#                     f.write(f"  Solapados: {parts['overlap']}\n\n")

#             # Guardar cubrimientos internos
#             save_result(out_folder, "coverage_inf.pkl", coverage_inf)
#             save_result(out_folder, "coverage_sup.pkl", coverage_sup)

#             print(f"✅ Resultados guardados en: {out_folder}")

# if __name__ == "__main__":
#     main()
