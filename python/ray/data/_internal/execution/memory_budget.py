import logging
import time
from typing import TYPE_CHECKING

import ray
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer

if TYPE_CHECKING:
    from ray.data._internal.execution.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


def humanize(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.2f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


# @lsf
class GlobalMemoryBudget:
    def __init__(self, initial_budget: float):
        self._initial_budget = initial_budget
        self._budget = initial_budget
        self._last_replenish_time = -1
        logger.debug(f"@lsf Initial global memory budget is {humanize(self._budget)}")

    def get(self) -> float:
        return self._budget

    def can_spend(self, amount: float) -> bool:
        return self._budget >= amount

    def spend(self, amount: float):
        if not self.can_spend(amount):
            raise ValueError("Insufficient global memory budget")
        self._budget -= amount
        return self._budget

    def spend_if_available(self, amount: float) -> bool:
        if self.can_spend(amount):
            self.spend(amount)
            return True
        return False

    def global_replenish(self, resource_manager: "ResourceManager", topology):
        grow_rate = _get_global_growth_rate(resource_manager, topology)
        now = time.time()
        time_elapsed = now - self._last_replenish_time

        self._budget += time_elapsed * grow_rate
        # Cap output_budget to object_store_memory
        self._budget = min(self._initial_budget, self._budget)
        logger.debug(
            f"@lsf INITIAL_BUDGET: {humanize(self._initial_budget)}, "
            f"self.output_budget: {humanize(self._budget)}, "
            f"time elapsed: {time_elapsed:.2f}s, "
            f"replenish rate: {humanize(grow_rate)}/s"
        )
        self._last_replenish_time = now


def tasks_recently_completed(metrics, duration) -> tuple[int, float]:
    """Return the number of tasks that started running recently, and the time span
    between the first task started running and now."""
    now = time.time()
    time_start = now - duration
    first_task_start = now

    num_tasks = 0
    for t in metrics._running_tasks_start_time.values():
        if t >= time_start:
            num_tasks += 1
            first_task_start = min(first_task_start, t)

    return num_tasks, now - first_task_start


def is_first_op(op) -> bool:
    return len(op.input_dependencies) == 1 and isinstance(
        op.input_dependencies[0], InputDataBuffer
    )


def _get_num_slots(
    op, resource_manager: "ResourceManager", cpu_taken: int = 0, gpu_taken: int = 0
) -> int:
    if op.incremental_resource_usage().cpu > 0:
        return resource_manager.get_global_limits().cpu - cpu_taken
    if op.incremental_resource_usage().gpu > 0:
        return resource_manager.get_global_limits().gpu - gpu_taken
    return 0


def _get_global_growth_rate(resource_manager: "ResourceManager", topology):
    DURATION = 30  # consider tasks in the last 30 seconds

    ret = 0
    for op, _state in topology.items():
        task_input_size = op._metrics.average_bytes_inputs_per_task
        if task_input_size is None:
            if is_first_op(op):
                task_input_size = 0
            else:
                task_input_size = (
                    ray.data.DataContext.get_current().target_max_block_size
                )

        num_completed, timespan = tasks_recently_completed(op._metrics, DURATION)
        if num_completed > 0:
            conservative_rate = num_completed / timespan
        else:
            conservative_rate = 0

        optimistic_rate = 0
        task_duration = op._metrics.average_task_duration
        if task_duration is not None:
            num_slots = _get_num_slots(op, resource_manager)
            optimistic_rate = num_slots / task_duration
        else:
            optimistic_rate = 0

        conservative_rate = (
            conservative_rate if conservative_rate > 0 else optimistic_rate
        )
        optimistic_rate = optimistic_rate if optimistic_rate > 0 else conservative_rate
        avg_rate = (optimistic_rate + conservative_rate) / 2
        logger.debug(
            f"@lsf {op.name} input size {humanize(task_input_size)} "
            f"optimistic {optimistic_rate:.2f}/s conservative {conservative_rate:.2f}/s avg {avg_rate:.2f}/s"
        )

        ret += task_input_size * avg_rate

    return ret


# @mzm
class PerOpMemoryBudget:
    def __init__(self):
        self._budget = -1
        self._last_replenish_time = -1

    def get(self) -> float:
        return self._budget

    def can_spend(self, amount: float) -> bool:
        return self._budget >= amount

    def spend(self, amount: float):
        if not self.can_spend(amount):
            raise ValueError("Insufficient memory budget")
        self._budget -= amount
        return self._budget

    def spend_if_available(self, amount: float) -> bool:
        if self.can_spend(amount):
            self.spend(amount)
            return True
        return False

    def replenish(self, op, resource_manager: "ResourceManager"):
        # Initialize output_budget to object_store_memory.
        INITIAL_BUDGET = resource_manager.get_global_limits().object_store_memory
        if self._budget == -1:
            self._budget = INITIAL_BUDGET
            self._last_replenish_time = time.time()
            return

        grow_rate = _get_per_op_grow_rate(op, resource_manager)
        now = time.time()
        time_elapsed = now - self._last_replenish_time

        self._budget += time_elapsed * grow_rate
        # Cap output_budget to object_store_memory
        self._budget = min(INITIAL_BUDGET, self._budget)
        logger.debug(
            f"@mzm INITIAL_BUDGET: {humanize(INITIAL_BUDGET)}, "
            f"budget: {humanize(self._budget)}, "
            f"time_elapsed: {time_elapsed:.2f}s, "
            f"grow_rate: {humanize(grow_rate)}/s"
        )
        self._last_replenish_time = now


def _get_per_op_grow_rate(op, resource_manager: "ResourceManager") -> float:
    total_processing_time = 0
    op_output_size = op._metrics.average_bytes_outputs_per_task or 0

    next_op = op
    output_input_multipler = (
        op._metrics.average_bytes_outputs_per_task or 0
    ) / ray.data.DataContext.get_current().target_max_block_size
    if output_input_multipler == 0:
        output_input_multipler = 5  # TODO: This can be a user hint

    cpu_taken = op.incremental_resource_usage().cpu * op.num_active_tasks()
    gpu_taken = op.incremental_resource_usage().gpu * op.num_active_tasks()

    while len(next_op.output_dependencies) > 0:
        assert len(next_op.output_dependencies) == 1
        next_op = next_op.output_dependencies[0]

        # TODO: The current metrics seem to be overestimating the actual task duration.
        # time_for_op = next_op._metrics.average_task_duration or 1
        time_for_op = 0.5
        num_slots = _get_num_slots(next_op, resource_manager, cpu_taken, gpu_taken)
        processing_time = time_for_op / num_slots * output_input_multipler
        total_processing_time += processing_time
        logger.debug(
            f"@lsf {next_op.name} time_for_op {time_for_op:.2f}s num_slots {num_slots} alpha {output_input_multipler} "
            f"processing_time {processing_time:.2f}s"
        )
        if (
            next_op._metrics.average_bytes_outputs_per_task
            and (next_op._metrics.average_bytes_inputs_per_task or 0) > 0
        ):
            output_input_multipler *= (
                next_op._metrics.average_bytes_outputs_per_task
                / next_op._metrics.average_bytes_inputs_per_task
            )

    ret = op_output_size / total_processing_time if total_processing_time > 0 else 0
    logger.debug(
        f"@lsf op_output_size {humanize(op_output_size)} total_processing_time {total_processing_time:.2f}s grow_rate {humanize(ret)}/s"
    )
    return ret
