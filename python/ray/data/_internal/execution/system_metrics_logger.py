import time
import os
import csv
import logging
import psutil
from datetime import datetime

try:
    from gpustat import GPUStatCollection
except ImportError:
    raise ModuleNotFoundError(
        "gpustat is required for monitoring GPU usage. To install, run `pip install gpustat`."
    )

KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB
TiB = 1024 * GiB


class SystemMetricsLogger:
    """
    Utility class for collecting various system metrics:
    CPU usage, memory usage, available memory,
    disk usage, available disk, network bytes sent, network bytes received
    GPU usage, GPU memory used

    Usage:
    >> logger = _SystemMetricsLogger(
        csv_path="temp.csv", log_path="temp.log", interval=5
    )
    >> logger.log_metrics()
    """

    def __init__(
        self,
        csv_path: str = "system_metrics_logger_temp.csv",
        log_path: str = "system_metrics_logger_temp.txt",
        interval: int = 1,
    ):
        self.csv_path = csv_path
        self.log_path = log_path
        self.interval = interval
        self.initial_net_io = psutil.net_io_counters()
        self.last_collection_time = time.perf_counter()

        self.headers = [
            "timestamp",  # timestamp
            "cpu_usage",  # psutil
            "cpu_usage_per_cpu",  # psutil
            "memory_usage",  # psutil
            "memory_available",  # psutil
            "disk_usage",  # psutil
            "disk_free",  # psutil
            "network_sent",  # psutil
            "network_received",  # psutil
            # "gpu_{i}_utilization",  # gpustat
            # "gpu_{i}_memory_used",  # gpustat
        ]
        self._init_logging()
        self._init_csv()

    def _init_logging(self):
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="[_SystemMetricsLogger] %(message)s",
        )

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as file:
                writer = csv.writer(file)
                gpu_stats = GPUStatCollection.new_query()
                for gpu in gpu_stats:
                    self.headers.append(f"gpu_{gpu.entry['index']}_utilization")
                    self.headers.append(f"gpu_{gpu.entry['index']}_memory_used")
                writer.writerow(self.headers)

    def log_once(self):
        virtual_memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage("/")
        net_io = psutil.net_io_counters()

        # Note that `psutil.cpu_percent` can also report per-cpu usage by setting `percpu=True`
        csv_data = {
            "timestamp": time.perf_counter(),
            "cpu_usage": psutil.cpu_percent(interval=None),
            "cpu_usage_per_cpu": psutil.cpu_percent(interval=None, percpu=True),
            "memory_usage": virtual_memory.percent,
            "memory_available": virtual_memory.available,
            "disk_usage": disk_usage.percent,
            "disk_free": disk_usage.free,
            "network_sent": net_io.bytes_sent - self.initial_net_io.bytes_sent,
            "network_received": net_io.bytes_recv - self.initial_net_io.bytes_recv,
        }

        gpu_stats = GPUStatCollection.new_query()
        for gpu in gpu_stats:
            csv_data[f"gpu_{gpu.entry['index']}_utilization"] = gpu["utilization.gpu"]
            csv_data[f"gpu_{gpu.entry['index']}_memory_used"] = gpu["memory.used"]

        beautify_data = csv_data.copy()
        beautify_data["timestamp"] = datetime.fromtimestamp(
            beautify_data["timestamp"]
        ).strftime("%Y-%m-%d %H:%M:%S")
        beautify_data["memory_available"] /= MiB
        beautify_data["disk_free"] /= GiB
        beautify_data["network_sent"] /= MiB
        beautify_data["network_received"] /= MiB

        self.last_collection_time = time.perf_counter()
        return csv_data, beautify_data

    def log_metrics(self):
        while True:
            csv_data, beautify_data = self.log_once()
            self._write_csv(csv_data)
            self._write_log(beautify_data)
            time.sleep(self.interval)

    def _write_csv(self, data):
        with open(self.csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([data[h] for h in self.headers])

    def _write_log(self, data):
        logging.info(f"Timestamp: {data['timestamp']}")
        logging.info(f"CPU Usage: {data['cpu_usage']}%")
        logging.info(f"CPU Usage (per CPU): {data['cpu_usage_per_cpu']}")
        logging.info(
            f"Memory Usage: {data['memory_usage']}%, Available: {data['memory_available']} MiB"
        )
        logging.info(
            f"Disk Usage: {data['disk_usage']}%, Free Space: {data['disk_free']} GiB"
        )
        logging.info(
            f"Network Sent: {data['network_sent']} MiB, Received: {data['network_received']} MiB"
        )
        gpu_keys = [key for key in data if key.startswith("gpu_")]
        gpu_indices = []
        for key in gpu_keys:
            idx = key.split("_")[1]
            gpu_indices.append(idx)
        gpu_indices = set(gpu_indices)
        for idx in gpu_indices:
            gpu_utilization, gpu_memory = (
                data[f"gpu_{idx}_utilization"],
                data[f"gpu_{idx}_memory_used"],
            )
            logging.info(
                f"GPU {idx} Utilization: {gpu_utilization}%, Memory Used: {gpu_memory} MiB\n"
            )

    def should_collect(self) -> bool:
        return time.perf_counter() - self.last_collection_time >= self.interval


if __name__ == "__main__":
    logger = SystemMetricsLogger(
        csv_path="temp_1.csv", log_path="temp_1.log", interval=1
    )
    logger.log_metrics()
