# main.py
# This is the main script to run the simulation.

import numpy as np
import hnswlib
import matplotlib.pyplot as plt
from tqdm import tqdm  # For a nice progress bar

# Import our custom classes
from src import config
from src.host import Host
from src.standard_ssd import StandardSSD
from src.axon_ssd import AxonSSD

def prepare_data(num_vectors, dim):
    """Generates the random vector data and queries."""
    print(f"Generating {num_vectors} random vectors of {dim} dimensions...")
    # Generate main dataset
    vectors = np.random.rand(num_vectors, dim).astype('float32')
    
    # Generate queries
    queries = np.random.rand(config.QUERY_COUNT, dim).astype('float32')
    return vectors, queries

def setup_traditional_system(vectors):
    """Creates the 'dumb' SSD and pre-loads it with the index file."""
    print("Setting up Traditional System (Host-based search)...")
    ssd = StandardSSD()
    
    # 1. Host CPU creates the index
    print("  Host CPU is building the index...")
    index = hnswlib.Index(space='l2', dim=config.VECTOR_DIMENSIONS)
    index.init_index(max_elements=config.DATASET_SIZE, ef_construction=200, M=16)
    index.add_items(vectors, np.arange(config.DATASET_SIZE))
    
    # 2. Estimate index size
    # (This is an estimation for simulation)
    vector_data_size_gb = vectors.nbytes / (1024**3)
    index_size_gb = vector_data_size_gb * config.INDEX_OVERHEAD_MULTIPLIER
    
    # 3. Write the giant index file to the SSD
    print(f"  Writing {index_size_gb:.4f} GB index file to Standard SSD...")
    ssd.write("full_vector_index.bin", index, index_size_gb)
    
    return ssd, "full_vector_index.bin"

def setup_axon_system(vectors):
    """Creates the 'smart' SSD and writes vectors to it (which builds the index)."""
    print("Setting up Axon System (On-drive search)...")
    ssd = AxonSSD(num_elements=config.DATASET_SIZE, dim=config.VECTOR_DIMENSIONS)
    
    # Write each vector one by one. The Axon SSD builds its index automatically.
    print("  Writing vectors to Axon SSD (this builds the internal index)...")
    for i in tqdm(range(config.DATASET_SIZE), desc="  Writing to Axon"):
        ssd.write_vector(vectors[i])
        
    return ssd

def run_benchmark(host, queries, trad_ssd, index_file, axon_ssd):
    """Runs the same queries against both systems and records performance."""
    print(f"\nRunning benchmark with {config.QUERY_COUNT} queries...")
    
    trad_latencies, trad_traffics, trad_cpu = [], [], []
    axon_latencies, axon_traffics, axon_cpu = [], [], []

    for query in tqdm(queries, desc="Benchmark"):
        # 1. Run on Traditional System
        _, latency, traffic, cpu = host.find_similar_traditional(
            trad_ssd, index_file, query, config.TOP_K
        )
        trad_latencies.append(latency)
        trad_traffics.append(traffic)
        trad_cpu.append(cpu)

        # 2. Run on Axon System
        _, latency, traffic, cpu = host.find_similar_axon(
            axon_ssd, query, config.TOP_K
        )
        axon_latencies.append(latency)
        axon_traffics.append(traffic)
        axon_cpu.append(cpu)
        
    # Compile results
    results = {
        'trad_latency_avg': np.mean(trad_latencies),
        'trad_traffic_avg': np.mean(trad_traffics),
        'trad_cpu_avg': np.mean(trad_cpu),
        'axon_latency_avg': np.mean(axon_latencies),
        'axon_traffic_avg': np.mean(axon_traffics),
        'axon_cpu_avg': np.mean(axon_cpu),
    }
    return results

def plot_results(results):
    """Generates the final comparison graphs."""
    print("\nGenerating results...")
    
    # --- Graph 1: Average Query Latency (The "Speed" graph) ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    
    labels = ['Traditional', 'Project Axon']
    values = [results['trad_latency_avg'], results['axon_latency_avg']]
    
    bars = plt.bar(labels, values, color=['#FF6347', '#4682B4'])
    plt.title('Average Query Latency (ms)')
    plt.ylabel('Latency (ms) - Log Scale')
    plt.yscale('log') # Use log scale to see the huge difference
    plt.bar_label(bars, fmt='%.4f ms')

    # --- Graph 2: PCIe Data Traffic per Query (The "Efficiency" graph) ---
    plt.subplot(1, 2, 2)
    values_traffic = [results['trad_traffic_avg'] * 1024, results['axon_traffic_avg'] * 1024] # Convert to MB
    
    bars = plt.bar(labels, values_traffic, color=['#FF6347', '#4682B4'])
    plt.title('Average Data Traffic per Query (MB)')
    plt.ylabel('Data Traffic (MB) - Log Scale')
    plt.yscale('log')
    plt.bar_label(bars, fmt='%.4f MB')
    
    plt.suptitle('Project Axon vs. Traditional Storage Benchmark', fontsize=16)
    plt.tight_layout()
    
    # Save the file
    plt.savefig("benchmark_results.png")
    print("Benchmark graphs saved to benchmark_results.png")
    plt.show()

# --- Main execution ---
if __name__ == "__main__":
    # 1. Create data
    vectors_data, query_data = prepare_data(config.DATASET_SIZE, config.VECTOR_DIMENSIONS)
    
    # 2. Setup systems
    traditional_ssd, index_file_name = setup_traditional_system(vectors_data)
    axon_ssd = setup_axon_system(vectors_data)
    
    # 3. Create the host
    host_computer = Host()
    
    # 4. Run benchmark
    benchmark_results = run_benchmark(
        host_computer, query_data, traditional_ssd, index_file_name, axon_ssd
    )
    
    # 5. Show results
    print("\n--- Benchmark Complete ---")
    print(f"Traditional Latency: {benchmark_results['trad_latency_avg']:.4f} ms")
    print(f"Axon Latency:        {benchmark_results['axon_latency_avg']:.4f} ms")
    
    trad_traffic_mb = benchmark_results['trad_traffic_avg'] * 1024
    axon_traffic_mb = benchmark_results['axon_traffic_avg'] * 1024
    print(f"\nTraditional Traffic: {trad_traffic_mb:.4f} MB per query")
    print(f"Axon Traffic:        {axon_traffic_mb:.4f} MB per query")
    
    print(f"\nTraditional Host CPU:  {benchmark_results['trad_cpu_avg']:.4f} ms per query")
    print(f"Axon Host CPU:         {benchmark_results['axon_cpu_avg']:.4f} ms per query")
    
    plot_results(benchmark_results)