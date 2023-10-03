import io
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, BinaryIO, Optional

from .pointer import ALLOWED_VERSIONS, Pointer

if TYPE_CHECKING:
    from .storage import LFSStorage

logger = logging.getLogger(__name__)


_HEADERS = (b"version " + version.encode("utf-8") for version in ALLOWED_VERSIONS)


@contextmanager
def smudge(storage: "LFSStorage", fobj: BinaryIO) -> BinaryIO:
    """Wrap the specified binary IO stream and run LFS smudge if necessary."""
    reader = io.BufferedReader(fobj)
    data = reader.peek(100)
    if any(data.startswith(header) for header in _HEADERS):
        lfs_obj: Optional[BinaryIO] = None
        try:
            pointer = Pointer.load(reader)
            lfs_obj = storage.open(pointer.oid, mode="rb")
        except (ValueError, OSError):
            logger.warning("Could not open LFS object, falling back to pointer")
        if lfs_obj:
            with lfs_obj:
                yield lfs_obj
            return
    yield reader


if __name__ == "__main__":
    # Minimal `git lfs smudge` CLI implementation
    import sys

    from scmrepo.git import Git

    if sys.stdin.isatty():
        sys.exit(
            "Cannot read from STDIN: "
            "This command should be run by the Git 'smudge' filter"
        )
    scm = Git()
    try:
        with smudge(scm.lfs_storage, sys.stdin.buffer) as fobj:
            sys.stdout.buffer.write(fobj.read())
    finally:
        scm.close()
