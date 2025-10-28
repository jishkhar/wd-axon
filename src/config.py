# config.py
# This file stores all simulation constants.

# --- Data Configuration ---
DATASET_SIZE = 50_000       # Number of vectors to store (e.g., 50k images)
VECTOR_DIMENSIONS = 128   # Dimensions of each vector (e.g., 128)
QUERY_COUNT = 100         # How many queries to run in the benchmark
TOP_K = 5                 # How many similar results to find (e.g., find top 5)

# --- Hardware Simulation Constants ---
PCIE_BANDWIDTH_GB_S = 6.0   # Bandwidth of the "data highway" (GB/s)
SSD_BASE_LATENCY_MS = 0.02  # Base overhead for any single SSD operation (milliseconds)

# --- Host (Traditional) Simulation ---
# We estimate the host CPU's work.
# Let's say it takes 0.05ms to search 1,000 vectors.
HOST_CPU_SEARCH_MS_PER_1K = 0.05

# --- Axon (New) Simulation ---
# This is the latency of the "mini-brain" (CSP) inside the Axon SSD.
# We set it to a fixed time, as it's optimized for this one job.
AXON_INTERNAL_SEARCH_MS = 0.5   # 500 microseconds

# --- Index Size Estimation ---
# HNSW index overhead. Let's assume the index is ~1.5x the size of the raw data.
# (128 dimensions * 4 bytes/float * 50k vectors) * 1.5
INDEX_OVERHEAD_MULTIPLIER = 1.5