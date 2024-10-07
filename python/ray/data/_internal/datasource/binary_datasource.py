from typing import TYPE_CHECKING
import time

import ray
from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.datasource.file_based_datasource import FileBasedDatasource

from ray.data._internal.execution.pipeline_metrics_tracker import PipelineMetricsTracker

if TYPE_CHECKING:
    import pyarrow


class BinaryDatasource(FileBasedDatasource):
    """Binary datasource, for reading and writing binary files."""

    _COLUMN_NAME = "bytes"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _read_stream(self, f: "pyarrow.NativeFile", path: str):
        start_time = time.perf_counter()
        data = f.readall()
        end_time = time.perf_counter()
        tracker = PipelineMetricsTracker.options(
            name="tracker", get_if_exists=True
        ).remote()
        tracker.record.remote(
            stage="read",
            pid=ray.get_runtime_context().get_worker_id(),
            end_time=end_time,
            wall_time=(end_time - start_time),
            num_rows=self._rows_per_file(),
        )

        builder = ArrowBlockBuilder()
        item = {self._COLUMN_NAME: data}
        builder.add(item)
        yield builder.build()

    def _rows_per_file(self):
        return 1
