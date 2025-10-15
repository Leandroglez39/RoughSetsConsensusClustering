import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Set, Optional
import networkx as nx
from typing import List, Set
import os
import numpy as np
import seaborn as sns
from consensus_signed import build_match_array

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

def build_fuzzy_consensus_matrix(N, coverage_inf, coverage_sup,
                                  w_core=1.0, w_mixed=0.6, w_overlap=0.4):
    """
    Construye una matriz de similitud difusa basada en la pertenencia
    de nodos a comunidades: núcleo vs solapado.
    """

    W = np.zeros((N, N))

    for inf, sup in zip(coverage_inf, coverage_sup):
        core_nodes = set(inf)
        overlap_nodes = set(sup) - core_nodes

        # Núcleo–núcleo
        for i in core_nodes:
            for j in core_nodes:
                W[i, j] += w_core

        # Núcleo–solapado (y viceversa)
        for i in core_nodes:
            for j in overlap_nodes:
                W[i, j] += w_mixed
                W[j, i] += w_mixed

        # Solapado–solapado
        for i in overlap_nodes:
            for j in overlap_nodes:
                W[i, j] += w_overlap

    # Normalización opcional si un nodo aparece en varias comunidades
    W = np.clip(W, 0, 1.0)

    return W


def plot_fuzzy_consensus_heatmap(W, coverage_inf, coverage_sup,
                                 gamma, alpha, output_path=None):
    """
    Visualiza la matriz difusa de consenso, agrupando por comunidades y
    mostrando los índices de los nodos y los límites comunitarios.
    """

    N = W.shape[0]
    node_order = []
    community_boundaries = [0]
    community_labels = []
    already_added = set()

    for i, (inf, sup) in enumerate(zip(coverage_inf, coverage_sup)):
        inf = sorted(set(inf) - already_added)
        sup_only = sorted(set(sup) - set(inf) - already_added)

        node_order.extend(inf)
        node_order.extend(sup_only)

        already_added.update(inf)
        already_added.update(sup_only)

        community_boundaries.append(len(node_order))
        community_labels.append(f'C{i}')

    remaining = sorted(set(range(N)) - already_added)
    if remaining:
        node_order.extend(remaining)
        community_boundaries.append(len(node_order))
        community_labels.append('No asignado')

    reordered_W = W[np.ix_(node_order, node_order)]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(reordered_W, cmap='YlGnBu', vmin=0, vmax=1.0)

    # Etiquetas de ejes: índices reales de los nodos
    tick_positions = np.arange(len(node_order))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(node_order, rotation=90, fontsize=6)
    ax.set_yticklabels(node_order, fontsize=6)

    # Líneas blancas de comunidad
    for b in community_boundaries[1:-1]:
        ax.axhline(y=b - 0.5, color='white', linewidth=2)
        ax.axvline(x=b - 0.5, color='white', linewidth=2)

    # Etiquetas de comunidad en el margen
    for idx, (start, end) in enumerate(zip(community_boundaries[:-1], community_boundaries[1:])):
        if end > start:
            mid = (start + end) // 2
            ax.text(-3, mid, community_labels[idx], va='center', ha='right', fontsize=8, fontweight='bold', transform=ax.transData)
            ax.text(mid, len(node_order) + 1, community_labels[idx], va='top', ha='center', fontsize=8, fontweight='bold', rotation=90, transform=ax.transData)

    ax.set_title(f'Matriz Difusa de Consenso\n(γ={gamma}, α={alpha})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Índice de nodo (reordenado)', fontsize=10)
    ax.set_ylabel('Índice de nodo (reordenado)', fontsize=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Nivel de pertenencia estructural', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matriz difusa guardada: {output_path}")

    plt.show()

    return node_order



def plot_consensus_quality_metrics(match_array, coverage_inf, coverage_sup, 
                                 gamma, alpha, output_path=None):
    """
    Gráfico complementario con métricas de calidad del consenso
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Histograma de valores de consenso
    upper_tri = match_array[np.triu_indices_from(match_array, k=1)]
    ax1.hist(upper_tri, bins=30, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Frecuencia de co-pertenencia')
    ax1.set_ylabel('Número de pares de nodos')
    ax1.set_title('Distribución de Valores de Consenso')
    ax1.axvline(np.mean(upper_tri), color='red', linestyle='--', 
               label=f'Media: {np.mean(upper_tri):.2f}')
    ax1.legend()
    
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
    
    # 3. Fuerza intra-comunidad vs inter-comunidad
    intra_strengths = []
    inter_strengths = []
    
    for i, sup_i in enumerate(coverage_sup):
        nodes_i = set(sup_i)
        
        # Fuerza intra-comunidad
        if len(nodes_i) > 1:
            intra_pairs = [(n1, n2) for n1 in nodes_i for n2 in nodes_i if n1 < n2]
            intra_strength = np.mean([match_array[n1, n2] for n1, n2 in intra_pairs])
            intra_strengths.append(intra_strength)
        
        # Fuerza inter-comunidad (promedio con otras comunidades)
        inter_strength_vals = []
        for j, sup_j in enumerate(coverage_sup):
            if i != j:
                nodes_j = set(sup_j)
                inter_pairs = [(n1, n2) for n1 in nodes_i for n2 in nodes_j]
                if inter_pairs:
                    inter_strength_vals.extend([match_array[n1, n2] for n1, n2 in inter_pairs])
        
        if inter_strength_vals:
            inter_strengths.append(np.mean(inter_strength_vals))
    
    ax3.scatter(range(len(intra_strengths)), intra_strengths, 
               label='Intra-comunidad', alpha=0.7, s=60)
    if inter_strengths:
        ax3.scatter(range(len(inter_strengths)), inter_strengths, 
                   label='Inter-comunidad', alpha=0.7, s=60)
    ax3.set_xlabel('Comunidad')
    ax3.set_ylabel('Fuerza de consenso promedio')
    ax3.set_title('Fuerza Intra vs Inter-comunidad')
    ax3.legend()
    
    # 4. Modularidad del consenso (simplificada)
    total_edges = np.sum(match_array) / 2
    modularity_terms = []
    
    for sup in coverage_sup:
        nodes = list(set(sup))
        if len(nodes) > 1:
            # Edges dentro de la comunidad
            internal_edges = sum(match_array[i, j] for i in nodes for j in nodes if i < j)
            # Grado esperado (simplificado)
            degree_sum = sum(np.sum(match_array[i, :]) for i in nodes)
            expected = (degree_sum ** 2) / (4 * total_edges) if total_edges > 0 else 0
            modularity_terms.append(internal_edges - expected)
    
    ax4.bar(range(len(modularity_terms)), modularity_terms, alpha=0.7)
    ax4.set_xlabel('Comunidad')
    ax4.set_ylabel('Contribución a Modularidad')
    ax4.set_title('Contribución por Comunidad a la Modularidad')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.suptitle(f'Métricas de Calidad del Consenso (γ={gamma}, α={alpha})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Métricas de calidad guardadas: {output_path}")
    
    plt.show()