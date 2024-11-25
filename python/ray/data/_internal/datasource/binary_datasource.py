from typing import TYPE_CHECKING

from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data.datasource.file_based_datasource import FileBasedDatasource

if TYPE_CHECKING:
    import pyarrow


class BinaryDatasource(FileBasedDatasource):
    """Binary datasource, for reading and writing binary files."""

    _COLUMN_NAME = "bytes"

    def _read_stream(self, f: "pyarrow.NativeFile", path: str):
        data = f.readall()
        item = {self._COLUMN_NAME: data}
        yield item

    def _rows_per_file(self):
        return 1
