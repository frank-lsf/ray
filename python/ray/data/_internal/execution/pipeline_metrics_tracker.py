import ray
import time
import numpy as np
import subprocess
import logging

from typing import List, Optional
from dataclasses import dataclass
from collections import deque

from ray.data._internal.execution.system_metrics_logger import SystemMetricsLogger

LOGGING_FILE = "autoscaler_temp.log"
logging.basicConfig(
    filename=LOGGING_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


logger = logging.getLogger("PipelineMetricsLogger")


@ray.remote(num_cpus=0)
class PipelineMetricsTracker:
    def __init__(self):
        # ------------------------
        #     Pipeline Metrics
        # ------------------------

        # Stores all recorded raw metrics.
        self.raw_metrics = {}

        # Stores data stall metrics up to the last analyzer event.
        # To get the overall pipeline data stall, we can always compute it from the raw metrics.
        self.stall_metrics = {}

        # Stores summary statistics up to last analyzer event
        self.summary_metrics = {}

        # ------------------------
        #     System Metrics
        # ------------------------
        self.system_metrics = []
        self.system_metrics_logger = SystemMetricsLogger(interval=10)

        # ------------------------
        # Pipeline Helper Vairables
        # ------------------------
        self.pipeline_first_output_time = None
        self.sink_stage_name = "inference"

        # ------------------------
        #         Analyzer
        # ------------------------
        self.analyzer = PipelineBottleneckAnalyzer()
        self.should_record_metric = True
        self.last_analysis_time = None

        # ------------------------
        #          Scaler
        # ------------------------
        self.scaler = PipelineResourceScaler()

        logger.info("[Initialized] PipelineMetricsTracker")

    def get_raw_metrics(self, start_from=None):
        """Return a copy of raw metrics, optionally starting from a specific time."""
        raw_metrics = {}
        for stage, events in self.raw_metrics.copy().items():
            raw_metrics[stage] = [
                event for event in events if event["time"] >= (start_from or 0)
            ]
        return raw_metrics

    def record(
        self, stage: str, pid: str, end_time: float, wall_time: float, num_rows: int
    ):
        if not self.should_record_metric:
            return

        # Collect pipeline metrics
        if not self.raw_metrics.get(stage):
            self.raw_metrics[stage] = []
        self.raw_metrics[stage].append(
            {"pid": pid, "time": end_time, "wall_time": wall_time, "rows": num_rows}
        )
        if not self.pipeline_first_output_time:
            self.pipeline_first_output_time = end_time
            self.last_analysis_time = end_time

        # Collect system metrics
        if self.system_metrics_logger.should_collect():
            self.system_metrics.append(self.system_metrics_logger.log_once()[0])

        # Analysis and scaling logic
        if self.analyzer.should_analyze():
            summary_metrics_computed = self.compute_summary_metrics(
                start_from=self.last_analysis_time
            )
            if not summary_metrics_computed:
                return
            self.analyzer.add_metrics(self.output_summary_metrics())
            request, msg = self.analyzer.analyze()
            logger.info(msg)
            if request:
                self.scaler.add_request(request)
                logger.info(f"[Generated] Scaling Request: {request}")
            self.last_analysis_time = time.perf_counter()

        if self.scaler.request_queue:
            self.scaler.scale()

        self.should_record_metric = True

    def compute_overall_tput(self, start_from=None):
        """Computes the overall tput for the pipeline."""
        raw_metrics = self.get_raw_metrics(start_from=start_from)
        if "inference" not in raw_metrics.keys():
            return 0
        end_time = raw_metrics[self.sink_stage_name][-1]["time"]
        num_rows = np.sum([r["rows"] for r in raw_metrics[self.sink_stage_name]])
        start_time = (
            self.pipeline_first_output_time
            if not start_from
            else self.last_analysis_time
        )
        return num_rows / (end_time - start_time)

    def compute_data_stall(self, start_from=None):
        for stage, events in self.get_raw_metrics(start_from=start_from).items():
            stall_times = []

            events = [event for event in events if event["time"] >= (start_from or 0)]
            if len(events) <= 1:
                return False

            for i in range(len(events) - 1):
                current_event = events[i]
                next_event = events[i + 1]

                stall_time = max(
                    next_event["time"]
                    - (current_event["time"] + current_event["wall_time"]),
                    0,
                )
                stall_times.append(stall_time)
            self.stall_metrics[stage] = stall_times
        return True

    def compute_summary_metrics(self, start_from=None):
        data_stall_computed = self.compute_data_stall(start_from=start_from)
        if not data_stall_computed:
            return False

        for stage, events in self.get_raw_metrics(start_from=start_from).items():
            self.summary_metrics[stage] = {
                "total_wall_time": np.sum([r["wall_time"] for r in events]),
                "total_num_rows": np.sum([r["rows"] for r in events]),
            }
            # num pids
            self.summary_metrics[stage]["num_pids"] = len(
                set([r["pid"] for r in events])
            )
            # mean tput
            self.summary_metrics[stage]["mean_tput"] = (
                self.summary_metrics[stage]["total_num_rows"]
                / self.summary_metrics[stage]["total_wall_time"]
            )
            # concurrent tput
            self.summary_metrics[stage]["concurrent_tput"] = (
                self.summary_metrics[stage]["mean_tput"]
                * self.summary_metrics[stage]["num_pids"]
            )

        for stage, stall_times in self.stall_metrics.items():
            # mean data stall
            self.summary_metrics[stage]["mean_data_stall"] = np.mean(stall_times)

        self.overall_tput = self.compute_overall_tput(start_from=start_from)
        return True

    def output_summary_metrics(self):
        """Generate PipelineStageMetrics objects for each stage in the pipeline."""
        stage_metrics = []
        for stage, metrics in self.summary_metrics.items():
            stage_metrics.append(
                PipelineStageMetrics(
                    name=stage,
                    num_pids=metrics["num_pids"],
                    mean_tput=metrics["mean_tput"],
                    concurrent_tput=metrics["concurrent_tput"],
                    overall_pipeline_tput=self.overall_tput,
                    mean_data_stall=metrics["mean_data_stall"],
                )
            )
        logger.info(stage_metrics)
        return stage_metrics

    def find_bottleneck_stage(self):
        self.compute_summary_metrics(start_from=self.last_analysis_time)
        bottleneck_stage = None
        lowest_tput = float("inf")

        for stage, metrics in self.summary_metrics.items():
            if metrics["mean_tput"] < lowest_tput:
                lowest_tput = metrics["mean_tput"]
                bottleneck_stage = stage

        return bottleneck_stage, lowest_tput

    def print_summary(self):
        self.compute_summary_metrics()
        for stage, metrics in self.summary_metrics.items():
            logger.info(f"Stage: {stage}")
            logger.info(f"Total Wall Time: {metrics['total_wall_time']}")
            logger.info(f"Total Num Rows: {metrics['total_num_rows']}")
            logger.info(f"Num Pids: {metrics['num_pids']}")
            logger.info(f"Mean Tput: {metrics['mean_tput']}")
            logger.info(f"Concurrent Tput: {metrics['concurrent_tput']}")
            logger.info(f"Mean Data Stall: {metrics['mean_data_stall']}")

            logger.info(f"Overall Pipeline Tput: {self.overall_tput}")
            logger.info("")
        # logger.info(self.system_metrics)


@dataclass
class PipelineStageMetrics:
    """
    e.g.
    stage: read
    mean_tput: 1000
    num_pids: 2
    concurrent_tput: 2000
    data_stall: 0.001
    """

    name: str
    num_pids: int
    mean_tput: float
    concurrent_tput: float
    overall_pipeline_tput: float
    mean_data_stall: float


@dataclass
class ScalingRequest:
    """
    e.g.
    resource: CPU
    additional_num_pids: 2
    """

    resource: str
    additional_num_pids: int


class PipelineBottleneckAnalyzer:
    def __init__(self, interval=20, sink_stage_name="inference"):
        self.interval = interval
        self.last_analysis_time = time.perf_counter()

        self.sink_stage_name = sink_stage_name
        self.data_stall_threshold = 0.15  # Seconds
        self.data_stall_compensated = 0  # Number of additional PIDs we've already added to compensate for data stall

    def add_metrics(self, metrics: List[PipelineStageMetrics]):
        self.metrics = metrics

    def should_analyze(self):
        return time.perf_counter() - self.last_analysis_time >= self.interval

    def analyze(self) -> Optional[ScalingRequest]:
        # For now, we only consider the CPU-bound cases; to add support for GPU request later.
        bottleneck_stage_by_tput = None
        bottleneck_stage_by_data_stall = None
        bottleneck_num_pids = 0

        lowest_tput = float("inf")
        lowest_data_stall = float("inf")

        inference_tput = 0

        # Identify the bottleneck stage by throughput and by data stall time
        for stage in self.metrics:
            if stage.concurrent_tput < lowest_tput:
                lowest_tput = stage.concurrent_tput
                bottleneck_stage_by_tput = stage

            if stage.mean_data_stall < lowest_data_stall:
                lowest_data_stall = stage.mean_data_stall
                bottleneck_stage_by_data_stall = stage
                bottleneck_num_pids = stage.num_pids

            if stage.name == "inference":
                inference_tput = stage.concurrent_tput
                inference_data_stall = stage.mean_data_stall

        # First, optimize along throughput
        bottleneck_stage = bottleneck_stage_by_tput
        self.last_analysis_time = time.perf_counter()

        if bottleneck_stage.name != self.sink_stage_name:
            # Throughput is not yet optimized; scale based on throughput bottleneck
            num_pid_optim = int(inference_tput // bottleneck_stage.mean_tput) + 1
            num_pid_requested = num_pid_optim - bottleneck_stage.num_pids

            msg = (
                "[Analyzed] Bottleneck stage (throughput):",
                bottleneck_stage.name,
                "Throughput:",
                lowest_tput,
                "Target Throughput:",
                inference_tput,
            )

        else:
            # Throughput is optimized; now check for data stall times
            bottleneck_stage = bottleneck_stage_by_data_stall

            data_stall_difference = (
                inference_data_stall - bottleneck_stage.mean_data_stall
            )

            if data_stall_difference > self.data_stall_threshold:
                logger.info(
                    f"Data stall detected at stage '{bottleneck_stage.name}'. "
                    f"Mean data stall time: {bottleneck_stage.mean_data_stall}"
                )
                num_pid_optim = int(inference_tput // bottleneck_stage.mean_tput) + 1

                self.data_stall_compensated += 1
                num_pid_optim += self.data_stall_compensated

                num_pid_requested = 1
            else:
                num_pid_requested = 0

            # Pipeline metrics summary
            msg = (
                "[Analyzed] Bottleneck stage (data stall):",
                bottleneck_stage.name,
                "Data stall time:",
                bottleneck_stage.mean_data_stall,
                "Data stall time difference:",
                data_stall_difference,
                "Target data stall difference:",
                self.data_stall_threshold,
            )
        # Num PIDs requested should not exceed the number of CPUs
        import psutil

        num_cpus = psutil.cpu_count(logical=False)
        if bottleneck_num_pids + num_pid_requested > num_cpus:
            logger.info(
                f"Optimal number of PIDs ({bottleneck_num_pids + num_pid_requested}) exceeds the number of CPUs ({num_cpus})."
            )
            return None, msg
        elif num_pid_requested == 0:
            logger.info("No scaling request generated.")
            return None, msg

        return ScalingRequest("CPU", num_pid_requested), msg


class PipelineResourceScaler:
    def __init__(self):
        self.request_queue = deque()

    def add_request(self, request: ScalingRequest):
        self.request_queue.append(request)

    def scale(self):
        # logger.info(list_nodes())

        if not self.request_queue:
            return
        request = self.request_queue.popleft()
        additional_num_pids = int(request.additional_num_pids)
        # ray.init() does not expose necessary ports for other workers to join
        # To fix, run `ray start --head` in the terminal and then ray.init("auto") in the script
        ray_start_command = f"ray start --address=10.3.0.4:6379 --num-cpus {additional_num_pids} --num-gpus 0"

        try:
            subprocess.run(ray_start_command, shell=True, check=True)
            logger.info(
                f"Successfully started {additional_num_pids} workers. Run `ray status` to check the current cluster status."
            )
            logger.info(f"[Completed] scaling request: {request}")
            time.sleep(5)
            ray_status_command = "ray status"
            result = subprocess.run(
                ray_status_command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )

            logger.info("\nCluster status after scaling:\n")
            logger.info(result.stdout)

        except subprocess.CalledProcessError as e:
            logger.info(f"Failed to start {additional_num_pids} workers. Error: {e}")
