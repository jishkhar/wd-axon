# axon_ssd.py
# Simulates your innovative Axon Computational SSD.

from src import config
import hnswlib
import numpy as np
import time

class AxonSSD:
    """
    Simulates the Axon Computational Storage Drive (CSD).
    It maintains its own internal vector index.
    """
    def __init__(self, num_elements, dim):
        # The "mini-brain" (CSP) manages this index internally
        self.internal_index = hnswlib.Index(space='l2', dim=dim)
        self.internal_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        # A simple key-value store for the raw vector data
        self.storage = {}
        self.current_id = 0

    def write_vector(self, vector_data):
        """
        Simulates writing a single vector.
        The Axon drive automatically indexes the data it receives.
        """
        vector_id = self.current_id
        
        # 1. Store the vector
        self.storage[vector_id] = vector_data
        
        # 2. Add vector to the internal index
        self.internal_index.add_items(vector_data.reshape(1, -1), vector_id)
        
        self.current_id += 1
        
        # Latency is just the tiny base overhead for a small write
        return config.SSD_BASE_LATENCY_MS

    def execute_custom_command(self, command, payload):
        """
        This is the "magic" of the Axon drive.
        The host sends a command, and the drive does the work internally.
        """
        if command == "FIND_SIMILAR":
            query_vector = payload['vector']
            k = payload['k']
            
            # 1. Search happens *inside the SSD* using its "mini-brain"
            # We simulate this cost with a fixed latency
            search_latency_ms = config.AXON_INTERNAL_SEARCH_MS
            labels, distances = self.internal_index.knn_query(query_vector, k=k)
            
            # 2. Fetch *only* the resulting vectors from internal storage
            # This is a tiny operation
            fetch_latency_ms = config.SSD_BASE_LATENCY_MS * k
            
            results = [self.storage[label] for label in labels[0]]
            
            total_latency_ms = search_latency_ms + fetch_latency_ms
            return results, total_latency_ms
            
        return None, 0