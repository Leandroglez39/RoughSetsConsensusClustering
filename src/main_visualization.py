import numpy as np
import os
from consensus_visualization import plot_community_evolution_across_gamma

def generate_synthetic_consensus(gamma, num_nodes=50):
    """
    Genera un consenso sintético que varía según gamma.
    A mayor gamma, más solapamiento entre comunidades.
    """
    np.random.seed(42)
    
    # Número de comunidades decrece con gamma alto
    num_comms = max(2, int(6 - gamma * 2))
    
    cov_inf = []
    cov_sup = []
    
    nodes_per_comm = num_nodes // num_comms
    
    for i in range(num_comms):
        start = i * nodes_per_comm
        end = min(start + nodes_per_comm + int(gamma * 10), num_nodes)
        
        # Núcleo: menos nodos a mayor gamma
        core_size = int(nodes_per_comm * (1 - gamma * 0.5))
        core = set(range(start, start + core_size))
        
        # Superior: incluye solapamiento
        superior = set(range(start, end))
        
        cov_inf.append(core)
        cov_sup.append(superior)
    
    return cov_inf, cov_sup


if __name__ == "__main__":
    print("🧪 Generando datos de prueba...")
    
    # Definir rango de gammas
    gamma_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Generar consensos para cada gamma
    consensus_results = []
    for gamma in gamma_values:
        cov_inf, cov_sup = generate_synthetic_consensus(gamma, num_nodes=50)
        consensus_results.append((cov_inf, cov_sup))
        print(f"  γ={gamma:.1f}: {len(cov_sup)} comunidades")
    
    # Crear directorio de salida
    output_dir = "output/test_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar visualización
    print("\n📊 Generando visualización...")
    plot_community_evolution_across_gamma(
        gamma_values=gamma_values,
        consensus_results=consensus_results,
        output_path=os.path.join(output_dir, "gamma_evolution_test.png"),
        title="Prueba: Evolución de Comunidades según Gamma"
    )
    
    print(f"\n✅ Visualización guardada en: {output_dir}/gamma_evolution_test.png")