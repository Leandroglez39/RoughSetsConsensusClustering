import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Set, Optional

import networkx as nx
from typing import List, Set
import os

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
