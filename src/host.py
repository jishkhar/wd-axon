# host.py
# Simulates the Host Computer (CPU, RAM).
from src import config
import numpy as np
import time

class Host:
    """
    Simulates the host machine.
    It tracks its own CPU work and PCIe traffic.
    """
    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        self.host_cpu_work_ms = 0.0
        self.pcie_traffic_gb = 0.0

    def find_similar_traditional(self, ssd, index_file_name, query_vector, k):
        """
        Scenario A: The "Old Way".
        1. Read the HUGE index file from SSD to Host RAM.
        2. Use the Host CPU to perform the search.
        """
        self.reset_stats()
        total_latency_ms = 0
        
        # 1. Read the entire index file across the PCIe bus
        index_size_gb = ssd.storage[index_file_name]['size_gb']
        index_data, read_latency = ssd.read(index_file_name)
        
        self.pcie_traffic_gb += index_size_gb
        total_latency_ms += read_latency
        
        # 2. Host CPU does the search
        host_index = index_data
        search_start_time = time.perf_counter()
        
        # --- THIS IS THE HOST CPU WORK ---
        labels, distances = host_index.knn_query(query_vector, k=k)
        # --- END HOST CPU WORK ---
        
        # Simulate CPU work latency
        search_latency_ms = (host_index.get_current_count() / 1000) * config.HOST_CPU_SEARCH_MS_PER_1K
        
        self.host_cpu_work_ms += search_latency_ms
        total_latency_ms += search_latency_ms
        
        # (We skip the final read of the data, as the index read is the main bottleneck)
        return labels, total_latency_ms, self.pcie_traffic_gb, self.host_cpu_work_ms

    def find_similar_axon(self, ssd, query_vector, k):
        """
        Scenario B: The "New Way" (Project Axon).
        1. Send a TINY command to the Axon SSD.
        2. The SSD does all the work and sends back only the result.
        """
        self.reset_stats()
        
        # 1. Send the small query vector to the SSD
        payload = {'vector': query_vector, 'k': k}
        query_size_bytes = query_vector.nbytes
        query_size_gb = query_size_bytes / (1024**3)
        
        self.pcie_traffic_gb += query_size_gb
        
        # 2. The SSD does all the work. The host CPU is idle.
        results, search_latency_ms = ssd.execute_custom_command("FIND_SIMILAR", payload)
        
        # 3. Receive the small result back
        result_size_bytes = np.array(results).nbytes
        result_size_gb = result_size_bytes / (1024**3)
        self.pcie_traffic_gb += result_size_gb

        # The total latency is just the command's latency.
        # The host CPU work is near-zero.
        return results, search_latency_ms, self.pcie_traffic_gb, self.host_cpu_work_ms