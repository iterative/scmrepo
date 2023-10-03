import io
import logging
from typing import TYPE_CHECKING, Callable, List, Optional

from pygit2 import GIT_FILTER_CLEAN, Filter, Passthrough

if TYPE_CHECKING:
    from pygit2 import FilterSource

logger = logging.getLogger(__name__)


class LFSFilter(Filter):
    attributes = "filter=*"

    def __init__(self, *args, **kwargs):
        self._smudge_buf: Optional[io.BytesIO] = None

    def check(self, src: "FilterSource", attr_values: List[str]):
        if attr_values[0] == "lfs":
            if src.mode != GIT_FILTER_CLEAN:
                self._smudge_buf = io.BytesIO()
                return
        raise Passthrough

    def write(
        self, data: bytes, src: "FilterSource", write_next: Callable[[bytes], None]
    ):
        if src.mode == GIT_FILTER_CLEAN:
            write_next(data)
            return
        if self._smudge_buf is None:
            self._smudge_buf = io.BytesIO()
        self._smudge_buf.write(data)

    def close(self, write_next: Callable[[bytes], None]):
        if self._smudge_buf is not None:
            self._smudge(write_next)

    def _smudge(self, write_next: Callable[[bytes], None]):
        from scmrepo.git import Git
        from scmrepo.git.lfs import smudge

        self._smudge_buf.seek(0)
        scm = Git()
        try:
            with smudge(scm.lfs_storage, self._smudge_buf) as fobj:
                data = fobj.read(io.DEFAULT_BUFFER_SIZE)
                while data:
                    write_next(data)
                    data = fobj.read(io.DEFAULT_BUFFER_SIZE)
        finally:
            scm.close()
