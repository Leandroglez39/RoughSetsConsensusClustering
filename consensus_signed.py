from collections import defaultdict
import numpy as np
import networkx as nx
from typing import List, Set, Tuple
import os
import pickle


def similarity_between_subgraphs_from_R(
    R: np.ndarray,
    A: Set[int],
    B: Set[int],
    alpha: float = -1.0
) -> float:
    total = 0.0
    for i in A:
        for j in B:
            val = R[i, j]
            total += val if val >= 0 else alpha * abs(val)
    return total / (len(A) * len(B))


def build_match_array(communities: List[List[List[int]]], N: int) -> np.ndarray:
    match_array = np.zeros((N, N), dtype=int)
    for partition in communities:
        for community in partition:
            for i in community:
                for j in community:
                    match_array[i, j] += 1
    return match_array


def rough_clustering_signed(
    R_all: np.ndarray,
    communities: List[List[List[int]]],
    gamma: float = 0.8,
    alpha: float = -1.0
) -> Tuple[List[Set[int]], List[Set[int]]]:
    T, N, _ = R_all.shape
    R_mean = R_all.mean(axis=0)

    match_array = build_match_array(communities, N)
    b0 = len(communities) * 0.75

    G_B0 = nx.Graph()
    G_B0.add_nodes_from(range(N))
    i_idx, j_idx = np.where(match_array >= b0)
    G_B0.add_edges_from([(i, j) for i, j in zip(i_idx, j_idx)])

    components = list(nx.connected_components(G_B0))
    seeds_subgraphs = [set(component) for component in components]
    seeds_subgraphs.sort(key=len, reverse=True)

    k = int(np.percentile([len(c) for c in communities], 91))
    k = min(k, len(seeds_subgraphs) - 1)

    coverage_inferior = [set(sg) for sg in seeds_subgraphs[:k+1]]
    coverage_superior = [set(sg) for sg in seeds_subgraphs[:k+1]]

    for j in range(k+1, len(seeds_subgraphs)):
        group_j = seeds_subgraphs[j]

        sim_match = [
            np.sum([match_array[i, j] for i in group_j for j in coverage_inferior[g]])
            / (len(group_j) * len(coverage_inferior[g]))
            for g in range(k+1)
        ]

        sim_signed = [
            similarity_between_subgraphs_from_R(R_mean, group_j, coverage_inferior[g], alpha)
            for g in range(k+1)
        ]

        max_match = max(sim_match) if max(sim_match) > 0 else 1.0
        max_signed = max(sim_signed) if max(sim_signed) > 0 else 1.0
        sim_match_norm = [s / max_match for s in sim_match]
        sim_signed_norm = [s / max_signed for s in sim_signed]

        sim_total = [(sim_match_norm[i] + sim_signed_norm[i]) / 2 for i in range(k+1)]
        max_index = sim_total.index(max(sim_total))

        T = [i for i in range(k+1) if i != max_index and sim_total[i] >= gamma]

        if T:
            for i in T:
                coverage_superior[i].update(group_j)
            coverage_superior[max_index].update(group_j)
        else:
            coverage_superior[max_index].update(group_j)
            coverage_inferior[max_index].update(group_j)

    return coverage_inferior, coverage_superior


def load_communities_from_npy(folder_path: str) -> List[List[List[int]]]:
    """
    Carga múltiples archivos .npy que contienen particiones de comunidades.
    Cada archivo debe contener una partición: lista de listas de enteros (nodos).

    Retorna:
    - Lista de particiones (una por archivo)
    """
    all_partitions = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".npy"):
            full_path = os.path.join(folder_path, fname)
            data = np.load(full_path, allow_pickle=True)
            # Asegurar estructura: lista de listas
            if isinstance(data, np.ndarray):
                data = data.tolist()
            if not isinstance(data, list) or not all(isinstance(c, list) for c in data):
                raise ValueError(f"{fname} no contiene una partición válida.")
            all_partitions.append(data)
    return all_partitions



def save_result(output_path: str, name: str, obj) -> None:
    with open(os.path.join(output_path, name), "wb") as f:
        pickle.dump(obj, f)


def is_valid_partition(data) -> bool:
    """
    Devuelve True si data tiene formato List[List[int]]
    """
    return isinstance(data, list) and all(isinstance(c, list) for c in data)

def convert_labels_to_partition(labels: np.ndarray) -> List[List[int]]:
    """
    Convierte un vector de etiquetas en partición: List[List[int]]
    """
    comm_dict = defaultdict(list)
    for node, label in enumerate(labels):
        comm_dict[int(label)].append(node)
    return list(comm_dict.values())

def validate_and_fix_community_folder(folder_path: str, fixed_suffix: str = "_fixed") -> List[List[List[int]]]:
    """
    Verifica y corrige (si es necesario) los archivos .npy de comunidades.
    Guarda las particiones corregidas con sufijo "_fixed.npy".

    Retorna: lista de particiones válidas
    """
    all_partitions = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(folder_path, fname)
        data = np.load(fpath, allow_pickle=True)

        if isinstance(data, np.ndarray):
            if data.ndim == 1 and np.issubdtype(data.dtype, np.integer):
                print(f"[!] Corrigiendo vector de etiquetas: {fname}")
                partition = convert_labels_to_partition(data)
            elif data.ndim == 1 and all(isinstance(c, list) for c in data.tolist()):
                print(f"[✓] Partición válida encontrada: {fname}")
                partition = data.tolist()
            elif is_valid_partition(data.tolist()):
                print(f"[✓] Partición válida encontrada: {fname}")
                partition = data.tolist()
            else:
                raise ValueError(f"[✘] Estructura inválida en {fname}")
        elif is_valid_partition(data):
            print(f"[✓] Partición válida encontrada: {fname}")
            partition = data
        else:
            raise ValueError(f"[✘] Estructura inválida en {fname}")

        # Guardar versión corregida si aplica
        base = fname.replace(".npy", "")
        out_name = f"{base}{fixed_suffix}.npy"
        out_path = os.path.join(folder_path, out_name)
        np.save(out_path, partition)
        all_partitions.append(partition)

    return all_partitions