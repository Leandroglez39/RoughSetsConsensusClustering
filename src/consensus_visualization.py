import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Set, Optional
import os
import numpy as np
import seaborn as sns
from consensus_signed import build_match_array
from collections import defaultdict
import matplotlib.patches as mpatches

def plot_consensus_graph(
    coverage_inferior: List[Set[int]],
    coverage_superior: List[Set[int]],
    title: str = "Visualización del Consenso",
    output_path: Optional[str] = None,
    show_labels: bool = False
):
    """
    Visualiza:
    - Núcleo de comunidades: color distinto por comunidad.
    - Nodos solapados: en gris uniforme.
    """
    G = nx.Graph()
    num_comms = len(coverage_inferior)

    # Colormap para núcleos
    cmap = plt.get_cmap("tab20")
    community_colors = [cmap(i % 20) for i in range(num_comms)]
    gray_color = "#999999"

    node_colors = {}
    node_labels = {}

    # Marcar núcleo
    for idx, (core, sup) in enumerate(zip(coverage_inferior, coverage_superior)):
        for node in core:
            node_colors[node] = community_colors[idx]
        for node in sup - core:
            node_colors[node] = gray_color  # solapados en gris

    # Añadir nodos y conexiones internas para layout
    for idx, (_, sup) in enumerate(zip(coverage_inferior, coverage_superior)):
        group = list(sup)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                G.add_edge(group[i], group[j])

    # Añadir nodos aislados también
    for node in node_colors.keys():
        if node not in G:
            G.add_node(node)

    # Layout
    pos = nx.spring_layout(G, seed=42)

    # Preparar nodos por color
    color_groups = {}
    for node, color in node_colors.items():
        color_groups.setdefault(color, []).append(node)

    # Dibujo
    plt.figure(figsize=(10, 10))
    for color, nodes in color_groups.items():
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=color,
            edgecolors="black" if color == gray_color else "none",
            linewidths=1.0,
            node_size=300,
            alpha=0.9
        )

    nx.draw_networkx_edges(G, pos, alpha=0.2)

    if show_labels:
        node_labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6)

    plt.title(title)
    plt.axis("off")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"[✔] Visualización guardada en {output_path}")
    else:
        plt.show()

def export_consensus_to_gephi_gexf(
    coverage_inf: List[Set[int]],
    coverage_sup: List[Set[int]],
    output_path: str
):
    """
    Crea un grafo con nodos anotados por su comunidad:
    - 'comunidad_core': comunidad núcleo (si aplica)
    - 'comunidades_superior': lista de comunidades donde está (solapamiento)

    Se crean aristas entre todos los nodos de cada comunidad superior para dar cohesión.
    """

    G = nx.Graph()
    node_info = {}

    for comm_id, (core, sup) in enumerate(zip(coverage_inf, coverage_sup)):
        for node in sup:
            if node not in node_info:
                node_info[node] = {
                    "comunidad_core": -1,
                    "comunidades_superior": set()
                }

            node_info[node]["comunidades_superior"].add(comm_id)

            if node in core:
                node_info[node]["comunidad_core"] = comm_id

    for node, attrs in node_info.items():
        G.add_node(
            node,
            comunidad_core=attrs["comunidad_core"],
            comunidades_superior=",".join(map(str, sorted(attrs["comunidades_superior"])))
        )

    # Crear aristas dentro de cada comunidad superior (clique completa)
    for sup in coverage_sup:
        sup = list(sup)
        for i in range(len(sup)):
            for j in range(i + 1, len(sup)):
                G.add_edge(sup[i], sup[j])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nx.write_gexf(G, output_path)
    print(f"[✔] Grafo exportado a Gephi (.gexf) en: {output_path}")

def plot_structured_consensus_matrix(match_array, coverage_inf, coverage_sup,
                                     gamma, alpha, output_path=None):
    """
    Visualiza la matriz de consenso estructurada por comunidades,
    distinguiendo nodos núcleo y solapados sin duplicación.
    """

    N = match_array.shape[0]
    node_order = []
    node_labels = []
    community_boundaries = []
    already_added = set()

    # Construcción de orden de nodos: primero núcleo, luego solapados únicos
    for i, (inf, sup) in enumerate(zip(coverage_inf, coverage_sup)):
        inf = sorted(set(inf) - already_added)
        sup_only = sorted(set(sup) - set(inf) - already_added)

        node_order.extend(inf)
        node_labels.extend([f'C{i} (núcleo)'] * len(inf))

        node_order.extend(sup_only)
        node_labels.extend([f'C{i} (solapado)'] * len(sup_only))

        already_added.update(inf)
        already_added.update(sup_only)

        community_boundaries.append(len(node_order))

    # Nodos no asignados a ninguna comunidad
    remaining = sorted(set(range(N)) - already_added)
    node_order.extend(remaining)
    node_labels.extend(['No asignado'] * len(remaining))
    if remaining:
        community_boundaries.append(len(node_order))

    # Reordenar matriz
    reordered_matrix = match_array[np.ix_(node_order, node_order)]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(reordered_matrix, cmap='RdYlBu_r', aspect='auto', 
                   vmin=0, vmax=np.max(match_array))

    # Líneas entre comunidades
    for b in community_boundaries[:-1]:
        ax.axhline(y=b - 0.5, color='white', linewidth=2)
        ax.axvline(x=b - 0.5, color='white', linewidth=2)

    # Etiquetas
    ax.set_title(f'Matriz de Consenso Estructurada\n(γ={gamma}, α={alpha})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Nodos (reordenados por comunidad)', fontsize=12)
    ax.set_ylabel('Nodos (reordenados por comunidad)', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Frecuencia de co-pertenencia', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matriz de consenso guardada: {output_path}")

    plt.show()

    return node_order, node_labels, community_boundaries

def build_fuzzy_matrix_with_repeats(match_array, coverage_inf, coverage_sup,
                                     w_core=1.0, w_overlap_base=0.6):
    """
    Construye matriz fuzzy MxM permitiendo repeticiones de nodos solapados.
    """

    # 1. Construir un diccionario: nodo → número de comunidades en las que aparece
    node_to_num_comms = defaultdict(int)
    for sup in coverage_sup:
        for node in sup:
            node_to_num_comms[node] += 1

    # 2. Construir la secuencia extendida de nodos (con repeticiones)
    extended_nodes = []
    node_roles = []
    community_boundaries = [0]
    community_labels = []

    for i, (inf, sup) in enumerate(zip(coverage_inf, coverage_sup)):
        core_nodes = sorted(inf)
        overlap_nodes = sorted(set(sup) - set(inf))

        extended_nodes.extend(core_nodes)
        node_roles.extend(['core'] * len(core_nodes))

        extended_nodes.extend(overlap_nodes)
        node_roles.extend(['overlap'] * len(overlap_nodes))

        community_boundaries.append(len(extended_nodes))
        community_labels.append(f'C{i}')

    M = len(extended_nodes)
    W = np.zeros((M, M))

    # 3. Rellenar la matriz MxM basada en la matriz de consenso original
    for i in range(M):
        for j in range(M):
            node_i = extended_nodes[i]
            node_j = extended_nodes[j]

            base_val = match_array[node_i, node_j]

            # Ajustar intensidad según el rol (núcleo o solapado)
            if node_roles[i] == 'core' and node_roles[j] == 'core':
                intensity = w_core
            else:
                # Solapado: ajustamos según el grado de solapamiento
                penalty_i = 1 / node_to_num_comms[node_i]
                penalty_j = 1 / node_to_num_comms[node_j]
                intensity = w_overlap_base * min(penalty_i, penalty_j)

            W[i, j] = base_val * intensity

    return W, extended_nodes, node_roles, community_boundaries, community_labels

def plot_fuzzy_matrix(W, extended_nodes, community_boundaries, community_labels,
                      gamma, alpha, output_path=None):

    M = W.shape[0]
    fig, ax = plt.subplots(figsize=(12, 10))

    W_masked = mask_offdiagonal_blocks(W, community_boundaries)
    im = ax.imshow(W_masked, cmap='YlGnBu', vmin=0, vmax=np.nanmax(W))

    # Quitar etiquetas de los ejes
    ax.set_xticks([])
    ax.set_yticks([])

    # Líneas entre comunidades
    for b in community_boundaries[1:-1]:
        ax.axhline(y=b - 0.5, color='white', linewidth=2)
        ax.axvline(x=b - 0.5, color='white', linewidth=2)

    # Etiquetas de comunidad en márgenes
    for idx, (start, end) in enumerate(zip(community_boundaries[:-1], community_boundaries[1:])):
        if end > start:
            mid = (start + end) // 2
            ax.text(-3, mid, community_labels[idx], va='center', ha='right', fontsize=8, fontweight='bold', transform=ax.transData)
            ax.text(mid, M + 1, community_labels[idx], va='top', ha='center', fontsize=8, fontweight='bold', rotation=90, transform=ax.transData)

    ax.set_title(f'Matriz Fuzzy por Comunidad\n(γ={gamma}, α={alpha*(-1)})', fontsize=14)
    ax.set_xlabel('Nodos (repetidos por comunidad)', fontsize=10)
    ax.set_ylabel('Nodos (repetidos por comunidad)', fontsize=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pertenencia fuzzy', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matriz fuzzy guardada: {output_path}")

    plt.show()

def mask_offdiagonal_blocks(W, community_boundaries):
    """
    Anula las celdas fuera de los bloques diagonales (inter-comunidad).
    """
    M = W.shape[0]
    masked_W = np.full_like(W, np.nan, dtype=float)

    for start, end in zip(community_boundaries[:-1], community_boundaries[1:]):
        masked_W[start:end, start:end] = W[start:end, start:end]

    return masked_W

def plot_consensus_quality_metrics(match_array, coverage_inf, coverage_sup, 
                                 gamma, alpha, output_path=None):
    """
    Gráfico complementario con métricas de calidad del consenso
    """
    fig, ax2 = plt.subplots(figsize=(12, 6))

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))    
    
    # # 1. Histograma de valores de consenso
    # upper_tri = match_array[np.triu_indices_from(match_array, k=1)]
    # ax1.hist(upper_tri, bins=30, alpha=0.7, edgecolor='black')
    # ax1.set_xlabel('Frecuencia de co-pertenencia')
    # ax1.set_ylabel('Número de pares de nodos')
    # ax1.set_title('Distribución de Valores de Consenso')
    # ax1.axvline(np.mean(upper_tri), color='red', linestyle='--', 
    #            label=f'Media: {np.mean(upper_tri):.2f}')
    # ax1.legend()
    
    # 2. Tamaños de comunidades
    community_sizes = []
    core_sizes = []
    overlap_sizes = []
    
    for inf, sup in zip(coverage_inf, coverage_sup):
        total_size = len(set(sup))
        core_size = len(set(inf))
        overlap_size = total_size - core_size
        
        community_sizes.append(total_size)
        core_sizes.append(core_size)
        overlap_sizes.append(overlap_size)
    
    x = range(len(community_sizes))
    ax2.bar(x, core_sizes, label='Núcleo', alpha=0.8)
    ax2.bar(x, overlap_sizes, bottom=core_sizes, label='Solapamiento', alpha=0.8)
    ax2.set_xlabel('Comunidad')
    ax2.set_ylabel('Número de nodos')
    ax2.set_title('Composición de Comunidades')
    ax2.legend()
    
    # # 3. Fuerza intra-comunidad vs inter-comunidad
    # intra_strengths = []
    # inter_strengths = []
    
    # for i, sup_i in enumerate(coverage_sup):
    #     nodes_i = set(sup_i)
        
    #     # Fuerza intra-comunidad
    #     if len(nodes_i) > 1:
    #         intra_pairs = [(n1, n2) for n1 in nodes_i for n2 in nodes_i if n1 < n2]
    #         intra_strength = np.mean([match_array[n1, n2] for n1, n2 in intra_pairs])
    #         intra_strengths.append(intra_strength)
        
    #     # Fuerza inter-comunidad (promedio con otras comunidades)
    #     inter_strength_vals = []
    #     for j, sup_j in enumerate(coverage_sup):
    #         if i != j:
    #             nodes_j = set(sup_j)
    #             inter_pairs = [(n1, n2) for n1 in nodes_i for n2 in nodes_j]
    #             if inter_pairs:
    #                 inter_strength_vals.extend([match_array[n1, n2] for n1, n2 in inter_pairs])
        
    #     if inter_strength_vals:
    #         inter_strengths.append(np.mean(inter_strength_vals))
    
    # ax3.scatter(range(len(intra_strengths)), intra_strengths, 
    #            label='Intra-comunidad', alpha=0.7, s=60)
    # if inter_strengths:
    #     ax3.scatter(range(len(inter_strengths)), inter_strengths, 
    #                label='Inter-comunidad', alpha=0.7, s=60)
    # ax3.set_xlabel('Comunidad')
    # ax3.set_ylabel('Fuerza de consenso promedio')
    # ax3.set_title('Fuerza Intra vs Inter-comunidad')
    # ax3.legend()
    
    # # 4. Modularidad del consenso (simplificada)
    # total_edges = np.sum(match_array) / 2
    # modularity_terms = []
    
    # for sup in coverage_sup:
    #     nodes = list(set(sup))
    #     if len(nodes) > 1:
    #         # Edges dentro de la comunidad
    #         internal_edges = sum(match_array[i, j] for i in nodes for j in nodes if i < j)
    #         # Grado esperado (simplificado)
    #         degree_sum = sum(np.sum(match_array[i, :]) for i in nodes)
    #         expected = (degree_sum ** 2) / (4 * total_edges) if total_edges > 0 else 0
    #         modularity_terms.append(internal_edges - expected)
    
    # ax4.bar(range(len(modularity_terms)), modularity_terms, alpha=0.7)
    # ax4.set_xlabel('Comunidad')
    # ax4.set_ylabel('Contribución a Modularidad')
    # ax4.set_title('Contribución por Comunidad a la Modularidad')
    # ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # plt.suptitle(f'Métricas de Calidad del Consenso (γ={gamma}, α={alpha})', 
    #             fontsize=16, fontweight='bold')
    # plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Métricas de calidad guardadas: {output_path}")
    
    plt.show()



def plot_community_evolution_across_gamma(
    gamma_values: List[float],
    consensus_results: List[tuple],
    output_path: Optional[str] = None,
    title: str = "Evolución de Comunidades según Gamma",
    figsize: tuple = (14, 10),
    reverse_gamma: bool = True
):
    """
    Visualiza cómo evolucionan las comunidades a medida que varía gamma.
    
    Los nodos que pertenecen a múltiples comunidades se REPITEN en la visualización,
    mostrando la cardinalidad real de cada comunidad.
    
    Interpretación de gamma:
    - Gamma ALTO (ej: 0.7): Más riguroso, MENOS solapamiento
    - Gamma BAJO (ej: 0.3): Menos riguroso, MÁS solapamiento
    
    Visualización:
    - Eje X: Gamma (de mayor a menor = de menos a más solapamiento)
    - Eje Y: Nodos REPETIDOS por cada comunidad
      * Por cada comunidad: PRIMERO núcleos, LUEGO solapados
      * Los nodos solapados aparecen en MÚLTIPLES comunidades
    - Colores:
      * Núcleo: color vivo de la comunidad
      * Solapado: escala de grises según intensidad
    
    Parameters:
    -----------
    gamma_values : List[float]
        Lista de valores de gamma, ej: [0.3, 0.5, 0.7]
    consensus_results : List[tuple]
        Lista de tuplas (coverage_inf, coverage_sup) para cada gamma
    output_path : str, optional
        Ruta donde guardar la imagen
    title : str
        Título del gráfico
    figsize : tuple
        Tamaño de la figura
    reverse_gamma : bool
        Si True, ordena gamma de mayor a menor
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Validación
    if len(gamma_values) != len(consensus_results):
        raise ValueError("gamma_values y consensus_results deben tener la misma longitud")
    
    # Ordenar por gamma (de mayor a menor si reverse_gamma=True)
    if reverse_gamma:
        sorted_indices = np.argsort(gamma_values)[::-1]
    else:
        sorted_indices = np.argsort(gamma_values)
    
    gamma_values = [gamma_values[i] for i in sorted_indices]
    consensus_results = [consensus_results[i] for i in sorted_indices]
    
    n_gammas = len(gamma_values)
    
    # Usar el consenso con gamma más alto (más riguroso) como base
    cov_inf_base, cov_sup_base = consensus_results[0]
    n_communities_base = len(cov_sup_base)
    
    # Determinar número máximo de comunidades
    max_communities = max(len(cov_sup) for _, cov_sup in consensus_results)
    
    # === PALETA DE COLORES VIVOS PARA COMUNIDADES ===
    if max_communities <= 10:
        base_cmap = plt.cm.Set1
        colors = base_cmap(np.linspace(0, 0.9, max_communities))
    elif max_communities <= 20:
        base_cmap = plt.cm.tab20
        colors = base_cmap(np.linspace(0, 1, max_communities))
    else:
        base_cmap = plt.cm.hsv
        colors = base_cmap(np.linspace(0, 0.95, max_communities))
    
    # Filtrar colores grises
    for i in range(len(colors)):
        r, g, b, a = colors[i]
        max_rgb = max(r, g, b)
        min_rgb = min(r, g, b)
        saturation = max_rgb - min_rgb
        
        if saturation < 0.3:
            hue = (i * 137.5) % 360
            colors[i] = plt.cm.hsv(hue / 360.0)
    
    # === PASO 1: CREAR LISTA DE NODOS CON REPETICIONES ===
    # Orden: Para cada comunidad -> primero núcleos, luego solapados
    node_order = []
    node_to_communities = {}
    community_boundaries = [0]
    community_sizes = []
    
    for comm_idx, (inf_base, sup_base) in enumerate(zip(cov_inf_base, cov_sup_base)):
        # Separar núcleos y solapados de esta comunidad
        core_nodes = sorted(set(inf_base))
        overlap_nodes = sorted(set(sup_base) - set(inf_base))
        
        # PRIMERO: Agregar núcleos
        for node in core_nodes:
            node_order.append((node, comm_idx))
            
            if node not in node_to_communities:
                node_to_communities[node] = []
            node_to_communities[node].append(comm_idx)
        
        # DESPUÉS: Agregar solapados
        for node in overlap_nodes:
            node_order.append((node, comm_idx))
            
            if node not in node_to_communities:
                node_to_communities[node] = []
            node_to_communities[node].append(comm_idx)
        
        # Guardar cardinalidad total de la comunidad
        community_sizes.append(len(core_nodes) + len(overlap_nodes))
        
        # Marcar fin de esta comunidad
        community_boundaries.append(len(node_order))
    
    n_ordered_nodes = len(node_order)
    
    # === PASO 2: CREAR MATRIZ DE COLORES ===
    color_matrix = np.ones((n_ordered_nodes, n_gammas, 3))
    
    for gamma_idx, (cov_inf, cov_sup) in enumerate(consensus_results):
        for ordered_idx, (node, base_comm) in enumerate(node_order):
            # Buscar en qué comunidades está el nodo en ESTE gamma
            communities_with_node = []
            is_core_somewhere = False
            core_community = None
            
            for comm_idx, (inf, sup) in enumerate(zip(cov_inf, cov_sup)):
                if node in sup:
                    communities_with_node.append(comm_idx)
                    if node in inf:
                        is_core_somewhere = True
                        # Si es núcleo en la comunidad base, usar esa
                        if comm_idx == base_comm:
                            core_community = comm_idx
            
            # Si es núcleo en alguna comunidad pero no en la base, usar la primera
            if is_core_somewhere and core_community is None:
                for comm_idx, inf in enumerate(cov_inf):
                    if node in inf:
                        core_community = comm_idx
                        break
            
            n_comms = len(communities_with_node)
            
            # Determinar color
            if n_comms == 0:
                # No asignado en este gamma: blanco
                color_matrix[ordered_idx, gamma_idx] = [1, 1, 1]
                
            elif is_core_somewhere and core_community is not None:
                # Es NÚCLEO: color de la comunidad
                color_matrix[ordered_idx, gamma_idx] = colors[core_community][:3]
                
            elif n_comms == 1:
                # En una sola comunidad (no núcleo): color de esa comunidad
                color_matrix[ordered_idx, gamma_idx] = colors[communities_with_node[0]][:3]
                
            else:
                # SOLAPADO en múltiples comunidades: gris según intensidad
                intensity = max(0.2, 0.8 - (n_comms / max_communities) * 0.6)
                color_matrix[ordered_idx, gamma_idx] = [intensity, intensity, intensity]
    
    # === PASO 3: CREAR FIGURA ===
    fig, ax = plt.subplots(figsize=figsize)
    
    # Mostrar matriz
    ax.imshow(color_matrix, aspect='auto', interpolation='nearest')
    
    # === PASO 4: CONFIGURAR EJES ===
    # Eje Y (nodos repetidos por comunidad)
    ax.set_ylabel('Nodos (repetidos por comunidad)', fontsize=14, fontweight='bold')
    
    # Etiquetas de nodos (mostrar algunos IDs)
    if n_ordered_nodes <= 150:
        tick_step = max(1, n_ordered_nodes // 30)
        y_ticks = list(range(0, n_ordered_nodes, tick_step))
        y_labels = [f'{node_order[i][0]}' for i in y_ticks]  # Solo el ID del nodo
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=7)
    
    # Líneas BLANCAS DISCONTINUAS separadoras entre comunidades
    for boundary in community_boundaries[1:-1]:
        if boundary < n_ordered_nodes:
            ax.axhline(y=boundary - 0.5, color='white', linewidth=2.5, 
                      linestyle='--', alpha=0.9)
    
    # Eje X (gamma) - CORREGIDO
    ax.set_xlabel('Gamma (Menos solapamiento  ←  |  →  Más solapamiento)', 
                  fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_gammas))
    ax.set_xticklabels([f'{g:.2f}' for g in gamma_values], fontsize=12)
    
    # Título
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Grid vertical sutil
    ax.set_xticks(np.arange(n_gammas) - 0.5, minor=True)
    ax.grid(which='minor', axis='x', color='black', linewidth=1, alpha=0.3)
    
    # === PASO 5: LEYENDA CON CARDINALIDADES ===
    legend_elements = []
    
    # Título de sección: Comunidades
    legend_elements.append(
        mpatches.Patch(
            facecolor='none',
            edgecolor='none',
            label='── Comunidades (|C|) ──'
        )
    )
    
    # Mostrar comunidades con sus cardinalidades
    n_show = min(n_communities_base, 12)
    for comm_idx in range(n_show):
        cardinality = community_sizes[comm_idx]
        legend_elements.append(
            mpatches.Patch(
                facecolor=colors[comm_idx],
                edgecolor='black',
                linewidth=0.5,
                label=f'C{comm_idx} (|{cardinality}|)'
            )
        )
    
    if n_communities_base > n_show:
        legend_elements.append(
            mpatches.Patch(
                facecolor='white',
                edgecolor='black',
                hatch='///',
                label=f'... +{n_communities_base - n_show} más'
            )
        )
    
    # Separador
    legend_elements.append(
        mpatches.Patch(
            facecolor='none',
            edgecolor='none',
            label='── Solapamiento ──'
        )
    )
    
    # Escala de grises
    gray_levels = [
        (0.75, 'Bajo (1-2 com.)'),
        (0.50, 'Medio (3-4 com.)'),
        (0.25, 'Alto (5+ com.)')
    ]
    
    for gray_val, label in gray_levels:
        legend_elements.append(
            mpatches.Patch(
                facecolor=[gray_val, gray_val, gray_val],
                edgecolor='black',
                linewidth=0.5,
                label=label
            )
        )
    
    # No asignado
    legend_elements.append(
        mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            linewidth=0.5,
            label='No asignado'
        )
    )
    
    # Etiquetas de comunidades en el eje Y (con cardinalidad)
    for comm_idx in range(n_communities_base):
        if comm_idx < len(community_boundaries) - 1:
            start = community_boundaries[comm_idx]
            end = community_boundaries[comm_idx + 1]
            if end > start:
                mid = (start + end) / 2
                cardinality = community_sizes[comm_idx]
                ax.text(-1.5, mid, f'C{comm_idx}\n|{cardinality}|', 
                       ha='right', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor=colors[comm_idx], 
                                alpha=0.7, 
                                edgecolor='black', 
                                linewidth=1.5))
    
    # Posicionar leyenda
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.15, 1),
        loc='upper left',
        fontsize=9,
        title='Comunidades y Solapamiento',
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1
    )
    
    plt.tight_layout()
    
    # Guardar
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualización guardada: {output_path}")
    
    plt.show()
    
    return fig, ax

# def plot_community_evolution_across_gamma(
#     gamma_values: List[float],
#     consensus_results: List[tuple],
#     output_path: Optional[str] = None,
#     title: str = "Evolución de Comunidades según Gamma",
#     figsize: tuple = (14, 10),
#     reverse_gamma: bool = True
# ):
#     """
#     Visualiza cómo evolucionan las comunidades a medida que varía gamma.
    
#     Los nodos que pertenecen a múltiples comunidades se REPITEN en la visualización,
#     mostrando la cardinalidad real de cada comunidad.
    
#     Interpretación de gamma:
#     - Gamma ALTO (ej: 0.7): Más riguroso, MENOS solapamiento
#     - Gamma BAJO (ej: 0.3): Menos riguroso, MÁS solapamiento
    
#     Visualización:
#     - Eje X: Gamma (de mayor a menor = de menos a más solapamiento)
#     - Eje Y: Nodos REPETIDOS por cada comunidad a la que pertenecen
#       * Cada comunidad muestra TODOS sus nodos (núcleo + solapados)
#       * Los nodos solapados aparecen en MÚLTIPLES comunidades
#     - Colores:
#       * Núcleo: color vivo de la comunidad
#       * Solapado: escala de grises según intensidad
    
#     Parameters:
#     -----------
#     gamma_values : List[float]
#         Lista de valores de gamma, ej: [0.3, 0.5, 0.7]
#     consensus_results : List[tuple]
#         Lista de tuplas (coverage_inf, coverage_sup) para cada gamma
#     output_path : str, optional
#         Ruta donde guardar la imagen
#     title : str
#         Título del gráfico
#     figsize : tuple
#         Tamaño de la figura
#     reverse_gamma : bool
#         Si True, ordena gamma de mayor a menor
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
    
#     # Validación
#     if len(gamma_values) != len(consensus_results):
#         raise ValueError("gamma_values y consensus_results deben tener la misma longitud")
    
#     # Ordenar por gamma (de mayor a menor si reverse_gamma=True)
#     if reverse_gamma:
#         sorted_indices = np.argsort(gamma_values)[::-1]
#     else:
#         sorted_indices = np.argsort(gamma_values)
    
#     gamma_values = [gamma_values[i] for i in sorted_indices]
#     consensus_results = [consensus_results[i] for i in sorted_indices]
    
#     n_gammas = len(gamma_values)
    
#     # Usar el consenso con gamma más alto (más riguroso) como base
#     cov_inf_base, cov_sup_base = consensus_results[0]
#     n_communities_base = len(cov_sup_base)
    
#     # Determinar número máximo de comunidades
#     max_communities = max(len(cov_sup) for _, cov_sup in consensus_results)
    
#     # === PALETA DE COLORES VIVOS PARA COMUNIDADES ===
#     if max_communities <= 10:
#         base_cmap = plt.cm.Set1
#         colors = base_cmap(np.linspace(0, 0.9, max_communities))
#     elif max_communities <= 20:
#         base_cmap = plt.cm.tab20
#         colors = base_cmap(np.linspace(0, 1, max_communities))
#     else:
#         base_cmap = plt.cm.hsv
#         colors = base_cmap(np.linspace(0, 0.95, max_communities))
    
#     # Filtrar colores grises
#     for i in range(len(colors)):
#         r, g, b, a = colors[i]
#         max_rgb = max(r, g, b)
#         min_rgb = min(r, g, b)
#         saturation = max_rgb - min_rgb
        
#         if saturation < 0.3:
#             hue = (i * 137.5) % 360
#             colors[i] = plt.cm.hsv(hue / 360.0)
    
#     # === PASO 1: CREAR LISTA DE NODOS CON REPETICIONES ===
#     # Cada nodo aparece una vez por cada comunidad a la que pertenece
#     node_order = []
#     node_to_communities = {}  # nodo -> [lista de comunidades]
#     community_boundaries = [0]
#     community_sizes = []  # Para la leyenda
    
#     for comm_idx, (inf_base, sup_base) in enumerate(zip(cov_inf_base, cov_sup_base)):
#         # TODOS los nodos de esta comunidad (núcleo + solapados)
#         all_nodes_in_comm = sorted(set(sup_base))
        
#         # Agregar cada nodo a la visualización
#         for node in all_nodes_in_comm:
#             node_order.append((node, comm_idx))  # Tupla: (nodo_id, comunidad)
            
#             # Registrar que este nodo está en esta comunidad
#             if node not in node_to_communities:
#                 node_to_communities[node] = []
#             node_to_communities[node].append(comm_idx)
        
#         # Guardar cardinalidad de la comunidad
#         community_sizes.append(len(all_nodes_in_comm))
        
#         # Marcar fin de esta comunidad
#         community_boundaries.append(len(node_order))
    
#     n_ordered_nodes = len(node_order)
    
#     # === PASO 2: CREAR MATRIZ DE COLORES ===
#     color_matrix = np.ones((n_ordered_nodes, n_gammas, 3))
    
#     for gamma_idx, (cov_inf, cov_sup) in enumerate(consensus_results):
#         for ordered_idx, (node, base_comm) in enumerate(node_order):
#             # Buscar en qué comunidades está el nodo en ESTE gamma
#             communities_with_node = []
#             is_core_somewhere = False
#             core_community = None
            
#             for comm_idx, (inf, sup) in enumerate(zip(cov_inf, cov_sup)):
#                 if node in sup:
#                     communities_with_node.append(comm_idx)
#                     if node in inf:
#                         is_core_somewhere = True
#                         # Si es núcleo en la comunidad base, usar esa
#                         if comm_idx == base_comm:
#                             core_community = comm_idx
            
#             # Si es núcleo en alguna comunidad pero no en la base, usar la primera
#             if is_core_somewhere and core_community is None:
#                 for comm_idx, inf in enumerate(cov_inf):
#                     if node in inf:
#                         core_community = comm_idx
#                         break
            
#             n_comms = len(communities_with_node)
            
#             # Determinar color
#             if n_comms == 0:
#                 # No asignado en este gamma: blanco
#                 color_matrix[ordered_idx, gamma_idx] = [1, 1, 1]
                
#             elif is_core_somewhere and core_community is not None:
#                 # Es NÚCLEO: color de la comunidad
#                 color_matrix[ordered_idx, gamma_idx] = colors[core_community][:3]
                
#             elif n_comms == 1:
#                 # En una sola comunidad (no núcleo): color de esa comunidad
#                 color_matrix[ordered_idx, gamma_idx] = colors[communities_with_node[0]][:3]
                
#             else:
#                 # SOLAPADO en múltiples comunidades: gris según intensidad
#                 intensity = max(0.2, 0.8 - (n_comms / max_communities) * 0.6)
#                 color_matrix[ordered_idx, gamma_idx] = [intensity, intensity, intensity]
    
#     # === PASO 3: CREAR FIGURA ===
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Mostrar matriz
#     ax.imshow(color_matrix, aspect='auto', interpolation='nearest')
    
#     # === PASO 4: CONFIGURAR EJES ===
#     # Eje Y (nodos repetidos por comunidad)
#     ax.set_ylabel('Nodos (repetidos por comunidad)', fontsize=14, fontweight='bold')
    
#     # Etiquetas de nodos (mostrar algunos IDs)
#     if n_ordered_nodes <= 150:
#         tick_step = max(1, n_ordered_nodes // 30)
#         y_ticks = list(range(0, n_ordered_nodes, tick_step))
#         y_labels = [f'{node_order[i][0]}' for i in y_ticks]  # Solo el ID del nodo
#         ax.set_yticks(y_ticks)
#         ax.set_yticklabels(y_labels, fontsize=7)
    
#     # Líneas BLANCAS DISCONTINUAS separadoras entre comunidades
#     for boundary in community_boundaries[1:-1]:
#         if boundary < n_ordered_nodes:
#             ax.axhline(y=boundary - 0.5, color='white', linewidth=2.5, 
#                       linestyle='--', alpha=0.9)
    
#     # Eje X (gamma) - CORREGIDO
#     ax.set_xlabel('Gamma (Menos solapamiento  ←  |  →  Más solapamiento)', 
#                   fontsize=14, fontweight='bold')
#     ax.set_xticks(range(n_gammas))
#     ax.set_xticklabels([f'{g:.2f}' for g in gamma_values], fontsize=12)
    
#     # Título
#     ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
#     # Grid vertical sutil
#     ax.set_xticks(np.arange(n_gammas) - 0.5, minor=True)
#     ax.grid(which='minor', axis='x', color='black', linewidth=1, alpha=0.3)
    
#     # === PASO 5: LEYENDA CON CARDINALIDADES ===
#     legend_elements = []
    
#     # Título de sección: Comunidades
#     legend_elements.append(
#         mpatches.Patch(
#             facecolor='none',
#             edgecolor='none',
#             label='── Comunidades (|C|) ──'
#         )
#     )
    
#     # Mostrar comunidades con sus cardinalidades
#     n_show = min(n_communities_base, 12)
#     for comm_idx in range(n_show):
#         cardinality = community_sizes[comm_idx]
#         legend_elements.append(
#             mpatches.Patch(
#                 facecolor=colors[comm_idx],
#                 edgecolor='black',
#                 linewidth=0.5,
#                 label=f'C{comm_idx} (|{cardinality}|)'
#             )
#         )
    
#     if n_communities_base > n_show:
#         legend_elements.append(
#             mpatches.Patch(
#                 facecolor='white',
#                 edgecolor='black',
#                 hatch='///',
#                 label=f'... +{n_communities_base - n_show} más'
#             )
#         )
    
#     # Separador
#     legend_elements.append(
#         mpatches.Patch(
#             facecolor='none',
#             edgecolor='none',
#             label='── Solapamiento ──'
#         )
#     )
    
#     # Escala de grises
#     gray_levels = [
#         (0.75, 'Bajo (1-2 com.)'),
#         (0.50, 'Medio (3-4 com.)'),
#         (0.25, 'Alto (5+ com.)')
#     ]
    
#     for gray_val, label in gray_levels:
#         legend_elements.append(
#             mpatches.Patch(
#                 facecolor=[gray_val, gray_val, gray_val],
#                 edgecolor='black',
#                 linewidth=0.5,
#                 label=label
#             )
#         )
    
#     # No asignado
#     legend_elements.append(
#         mpatches.Patch(
#             facecolor='white',
#             edgecolor='black',
#             linewidth=0.5,
#             label='No asignado'
#         )
#     )
    
#     # Etiquetas de comunidades en el eje Y (con cardinalidad)
#     for comm_idx in range(n_communities_base):
#         if comm_idx < len(community_boundaries) - 1:
#             start = community_boundaries[comm_idx]
#             end = community_boundaries[comm_idx + 1]
#             if end > start:
#                 mid = (start + end) / 2
#                 cardinality = community_sizes[comm_idx]
#                 ax.text(-1.5, mid, f'C{comm_idx}\n|{cardinality}|', 
#                        ha='right', va='center', fontsize=10, fontweight='bold',
#                        bbox=dict(boxstyle='round,pad=0.5', 
#                                 facecolor=colors[comm_idx], 
#                                 alpha=0.7, 
#                                 edgecolor='black', 
#                                 linewidth=1.5))
    
#     # Posicionar leyenda
#     ax.legend(
#         handles=legend_elements,
#         bbox_to_anchor=(1.15, 1),
#         loc='upper left',
#         fontsize=9,
#         title='Comunidades y Solapamiento',
#         title_fontsize=10,
#         frameon=True,
#         fancybox=True,
#         shadow=True,
#         ncol=1
#     )
    
#     plt.tight_layout()
    
#     # Guardar
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"✅ Visualización guardada: {output_path}")
    
#     plt.show()
    
#     return fig, ax

# def plot_community_evolution_across_gamma(
#     gamma_values: List[float],
#     consensus_results: List[tuple],
#     output_path: Optional[str] = None,
#     title: str = "Evolución de Comunidades según Gamma",
#     figsize: tuple = (14, 10),
#     reverse_gamma: bool = True
# ):
#     """
#     Visualiza cómo evolucionan las comunidades a medida que varía gamma.
    
#     Interpretación de gamma:
#     - Gamma ALTO (ej: 0.7): Más riguroso, MENOS solapamiento
#     - Gamma BAJO (ej: 0.3): Menos riguroso, MÁS solapamiento
    
#     Visualización:
#     - Eje X: Gamma (de mayor a menor = menos a más solapamiento)
#     - Eje Y: Nodos agrupados por comunidades
#       * Primero: núcleos de C0, luego solapados de C0
#       * Después: núcleos de C1, luego solapados de C1
#       * Y así sucesivamente...
#     - Colores por columna (gamma):
#       * Núcleo: color vivo de la comunidad (NO grises)
#       * Solapado: escala de grises (intensidad según número de comunidades)
    
#     Parameters:
#     -----------
#     gamma_values : List[float]
#         Lista de valores de gamma, ej: [0.3, 0.5, 0.7]
#     consensus_results : List[tuple]
#         Lista de tuplas (coverage_inf, coverage_sup) para cada gamma
#     output_path : str, optional
#         Ruta donde guardar la imagen
#     title : str
#         Título del gráfico
#     figsize : tuple
#         Tamaño de la figura
#     reverse_gamma : bool
#         Si True, ordena gamma de mayor a menor (menos a más solapamiento)
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
#     from matplotlib.colors import ListedColormap
    
#     # Validación
#     if len(gamma_values) != len(consensus_results):
#         raise ValueError("gamma_values y consensus_results deben tener la misma longitud")
    
#     # Ordenar por gamma (de mayor a menor si reverse_gamma=True)
#     if reverse_gamma:
#         sorted_indices = np.argsort(gamma_values)[::-1]
#     else:
#         sorted_indices = np.argsort(gamma_values)
    
#     gamma_values = [gamma_values[i] for i in sorted_indices]
#     consensus_results = [consensus_results[i] for i in sorted_indices]
    
#     n_gammas = len(gamma_values)
    
#     # Usar el consenso con gamma más alto (más riguroso) como base para estructura
#     cov_inf_base, cov_sup_base = consensus_results[0]
#     n_communities_base = len(cov_sup_base)
    
#     # Determinar número máximo de comunidades entre todos los gammas
#     max_communities = max(len(cov_sup) for _, cov_sup in consensus_results)
    
#     # === PALETA DE COLORES VIVOS (NO GRISES) PARA COMUNIDADES ===
#     # Usar colores distintivos y saturados
#     if max_communities <= 10:
#         # Para pocas comunidades: usar Set1 (colores muy distintivos)
#         base_cmap = plt.cm.Set1
#         colors = base_cmap(np.linspace(0, 0.9, max_communities))
#     elif max_communities <= 20:
#         # Para comunidades moderadas: tab20 (20 colores distintivos)
#         base_cmap = plt.cm.tab20
#         colors = base_cmap(np.linspace(0, 1, max_communities))
#     else:
#         # Para muchas comunidades: hsv (espectro completo)
#         base_cmap = plt.cm.hsv
#         colors = base_cmap(np.linspace(0, 0.95, max_communities))
    
#     # Asegurar que NO haya grises en la paleta de comunidades
#     # Reemplazar colores demasiado grises (saturación baja) con colores vivos
#     for i in range(len(colors)):
#         r, g, b, a = colors[i]
#         # Calcular saturación (diferencia entre max y min de RGB)
#         max_rgb = max(r, g, b)
#         min_rgb = min(r, g, b)
#         saturation = max_rgb - min_rgb
        
#         # Si saturación es muy baja (color grisáceo), reemplazar con color vivo
#         if saturation < 0.3:
#             # Generar color aleatorio vivo basado en índice
#             hue = (i * 137.5) % 360  # Proporción áurea para distribución uniforme
#             colors[i] = plt.cm.hsv(hue / 360.0)
    
#     # === PASO 1: ORDENAR NODOS EN EJE Y ===
#     # Por cada comunidad: primero núcleos, luego solapados
#     node_order = []
#     community_boundaries = [0]
#     node_to_base_community = {}
    
#     for comm_idx, (inf_base, sup_base) in enumerate(zip(cov_inf_base, cov_sup_base)):
#         # 1. Núcleos de esta comunidad (coverage_inf)
#         core_nodes = sorted(set(inf_base))
#         node_order.extend(core_nodes)
        
#         for node in core_nodes:
#             node_to_base_community[node] = comm_idx
        
#         # 2. Solapados de esta comunidad (en sup pero no en inf)
#         overlap_nodes = sorted(set(sup_base) - set(inf_base))
#         node_order.extend(overlap_nodes)
        
#         for node in overlap_nodes:
#             if node not in node_to_base_community:
#                 node_to_base_community[node] = comm_idx
        
#         # Marcar fin de esta comunidad
#         community_boundaries.append(len(node_order))
    
#     # Nodos no asignados en la base (opcional, al final)
#     all_nodes = set()
#     for sup in cov_sup_base:
#         all_nodes.update(sup)
    
#     max_node = max(all_nodes) if all_nodes else 0
#     assigned_nodes = set(node_order)
#     remaining_nodes = sorted(set(range(max_node + 1)) - assigned_nodes)
#     node_order.extend(remaining_nodes)
    
#     n_ordered_nodes = len(node_order)
    
#     # === PASO 2: CREAR MATRIZ DE COLORES ===
#     # Para cada (nodo, gamma) determinar color
#     color_matrix = np.ones((n_ordered_nodes, n_gammas, 3))
    
#     for gamma_idx, (cov_inf, cov_sup) in enumerate(consensus_results):
#         for ordered_idx, node in enumerate(node_order):
#             # Buscar en qué comunidades está el nodo en ESTE gamma
#             communities_with_node = []
#             is_core_in_any = False
#             core_community = None
            
#             for comm_idx, (inf, sup) in enumerate(zip(cov_inf, cov_sup)):
#                 if node in sup:
#                     communities_with_node.append(comm_idx)
#                     if node in inf:
#                         is_core_in_any = True
#                         core_community = comm_idx
            
#             n_comms = len(communities_with_node)
            
#             # Determinar color
#             if n_comms == 0:
#                 # No asignado: blanco
#                 color_matrix[ordered_idx, gamma_idx] = [1, 1, 1]
                
#             elif is_core_in_any:
#                 # Es NÚCLEO en este gamma: color VIVO de la comunidad
#                 color_matrix[ordered_idx, gamma_idx] = colors[core_community][:3]
                
#             else:
#                 # Es SOLAPADO en este gamma: GRIS
#                 if n_comms == 1:
#                     # En una sola comunidad (pero no es núcleo)
#                     # Gris muy claro
#                     intensity = 0.75
#                     color_matrix[ordered_idx, gamma_idx] = [intensity, intensity, intensity]
#                 else:
#                     # En múltiples comunidades: gris más oscuro según cantidad
#                     # Escala de grises: de 0.7 (claro, 2 comunidades) a 0.2 (oscuro, muchas)
#                     intensity = max(0.2, 0.8 - (n_comms / max_communities) * 0.6)
#                     color_matrix[ordered_idx, gamma_idx] = [intensity, intensity, intensity]
    
#     # === PASO 3: CREAR FIGURA ===
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Mostrar matriz
#     ax.imshow(color_matrix, aspect='auto', interpolation='nearest')
    
#     # === PASO 4: CONFIGURAR EJES ===
#     # Eje Y (nodos agrupados por comunidades)
#     ax.set_ylabel('Nodos (por comunidad)', fontsize=14, fontweight='bold')
    
#     # Etiquetas de nodos
#     if n_ordered_nodes <= 100:
#         tick_step = max(1, n_ordered_nodes // 25)
#         y_ticks = list(range(0, n_ordered_nodes, tick_step))
#         y_labels = [f'{node_order[i]}' for i in y_ticks]
#         ax.set_yticks(y_ticks)
#         ax.set_yticklabels(y_labels, fontsize=8)
    
#     # Líneas separadoras entre comunidades
#     for boundary in community_boundaries[1:-1]:
#         if boundary < n_ordered_nodes:
#             ax.axhline(y=boundary - 0.5, color='black', linewidth=2.5, 
#                       linestyle='-', alpha=0.6)
    
#     # Eje X (gamma)
#     ax.set_xlabel('Gamma (← Más solapamiento | Menos solapamiento →)', 
#                   fontsize=14, fontweight='bold')
#     ax.set_xticks(range(n_gammas))
#     ax.set_xticklabels([f'{g:.2f}' for g in gamma_values], fontsize=12)
    
#     # Título
#     ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
#     # Grid vertical sutil
#     ax.set_xticks(np.arange(n_gammas) - 0.5, minor=True)
#     ax.grid(which='minor', axis='x', color='black', linewidth=1, alpha=0.3)
    
#     # === PASO 5: LEYENDA MEJORADA ===
#     legend_elements = []
    
#     # Mostrar colores de comunidades (máximo 12 para no saturar)
#     n_show = min(n_communities_base, 12)
#     for comm_idx in range(n_show):
#         legend_elements.append(
#             mpatches.Patch(
#                 facecolor=colors[comm_idx],
#                 edgecolor='black',
#                 linewidth=0.5,
#                 label=f'C{comm_idx}'
#             )
#         )
    
#     if n_communities_base > n_show:
#         legend_elements.append(
#             mpatches.Patch(
#                 facecolor='white',
#                 edgecolor='black',
#                 hatch='///',
#                 label=f'... +{n_communities_base - n_show} más'
#             )
#         )
    
#     # Separador visual
#     legend_elements.append(
#         mpatches.Patch(
#             facecolor='none',
#             edgecolor='none',
#             label='―――――――――'
#         )
#     )
    
#     # Escala de grises para solapamiento
#     gray_levels = [
#         (0.75, '1-2 comunidades'),
#         (0.50, '3-4 comunidades'),
#         (0.25, '5+ comunidades')
#     ]
    
#     for gray_val, label in gray_levels:
#         legend_elements.append(
#             mpatches.Patch(
#                 facecolor=[gray_val, gray_val, gray_val],
#                 edgecolor='black',
#                 linewidth=0.5,
#                 label=label
#             )
#         )
    
#     # No asignado
#     legend_elements.append(
#         mpatches.Patch(
#             facecolor='white',
#             edgecolor='black',
#             linewidth=0.5,
#             label='No asignado'
#         )
#     )
    
#     # Etiquetas de comunidades en el eje Y
#     for comm_idx in range(n_communities_base):
#         if comm_idx < len(community_boundaries) - 1:
#             start = community_boundaries[comm_idx]
#             end = community_boundaries[comm_idx + 1]
#             if end > start:
#                 mid = (start + end) / 2
#                 ax.text(-0.8, mid, f'C{comm_idx}', 
#                        ha='right', va='center', fontsize=11, fontweight='bold',
#                        bbox=dict(boxstyle='round,pad=0.4', 
#                                 facecolor=colors[comm_idx], 
#                                 alpha=0.7, 
#                                 edgecolor='black', 
#                                 linewidth=1.5))
    
#     # Posicionar leyenda
#     ax.legend(
#         handles=legend_elements,
#         bbox_to_anchor=(1.15, 1),
#         loc='upper left',
#         fontsize=9,
#         title='Comunidades y Solapamiento',
#         title_fontsize=10,
#         frameon=True,
#         fancybox=True,
#         shadow=True,
#         ncol=1
#     )
    
#     plt.tight_layout()
    
#     # Guardar
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"✅ Visualización guardada: {output_path}")
    
#     plt.show()
    
#     return fig, ax