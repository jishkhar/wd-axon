# standard_ssd.py
# Simulates a traditional, "dumb" SSD.

from src import config
import time

class StandardSSD:
    """
    Simulates a standard NVMe SSD.
    It stores "files" and simulates latency based on bandwidth.
    """
    def __init__(self):
        # A simple dictionary to act as the drive's storage
        self.storage = {} # {'file_name': {'data': object, 'size_gb': float}}

    def write(self, file_name, file_data, file_size_gb):
        """Simulates writing a file to the drive."""
        
        # Calculate latency: time = size / speed
        transfer_time_s = file_size_gb / config.PCIE_BANDWIDTH_GB_S
        total_latency_ms = (transfer_time_s * 1000) + config.SSD_BASE_LATENCY_MS
        
        # Store the "file"
        self.storage[file_name] = {
            'data': file_data,
            'size_gb': file_size_gb
        }
        
        # In a real sim, we'd use time.sleep(), but here we just return the cost
        return total_latency_ms

    def read(self, file_name):
        """Simulates reading a file from the drive."""
        
        if file_name not in self.storage:
            raise FileNotFoundError(f"File not found on SSD: {file_name}")
            
        file_info = self.storage[file_name]
        file_data = file_info['data']
        file_size_gb = file_info['size_gb']

        # Calculate latency: time = size / speed
        transfer_time_s = file_size_gb / config.PCIE_BANDWIDTH_GB_S
        total_latency_ms = (transfer_time_s * 1000) + config.SSD_BASE_LATENCY_MS

        # Return the data and the time it took
        return file_data, total_latency_ms