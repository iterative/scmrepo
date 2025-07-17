import io
import logging
from typing import TYPE_CHECKING, BinaryIO, Optional

from .pointer import HEADERS, Pointer

if TYPE_CHECKING:
    from .storage import LFSStorage

logger = logging.getLogger(__name__)


def smudge(
    storage: "LFSStorage",
    fobj: BinaryIO,
    url: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> BinaryIO:
    """Wrap the specified binary IO stream and run LFS smudge if necessary."""
    reader = io.BufferedReader(fobj)  # type: ignore[type-var]
    data = reader.peek(100)
    if any(data.startswith(header) for header in HEADERS):
        # read the pointer data into memory since the raw stream is unseekable
        # and we may need to return it in fallback case
        data = reader.read()
        lfs_obj: Optional[BinaryIO] = None
        try:
            pointer = Pointer.load(io.BytesIO(data))
            lfs_obj = storage.open(pointer, mode="rb", fetch_url=url)
        except (ValueError, OSError):
            logger.warning("Could not open LFS object, falling back to raw pointer")
        if lfs_obj:
            return lfs_obj
        return io.BytesIO(data)
    return reader


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
        with smudge(scm.lfs_storage, sys.stdin.buffer) as fobj_:
            sys.stdout.buffer.write(fobj_.read())
    finally:
        scm.close()
