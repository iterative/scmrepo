import errno
import os
from collections.abc import Collection
from typing import TYPE_CHECKING, BinaryIO, Callable, Optional, Union

from .pointer import Pointer
from .progress import LFSCallback

if TYPE_CHECKING:
    from scmrepo.git import Git
    from scmrepo.progress import GitProgressEvent


class LFSStorage:
    def __init__(self, path: Union[str, os.PathLike[str]]):
        self.path = path

    def fetch(
        self,
        url: str,
        objects: Collection[Pointer],
        progress: Optional[Callable[["GitProgressEvent"], None]] = None,
        batch_size: Optional[int] = None,
    ):
        from .client import LFSClient

        with LFSCallback.as_lfs_callback(progress) as cb:
            cb.set_size(len(objects))
            with LFSClient.from_git_url(url) as client:
                client.download(self, objects, callback=cb, batch_size=batch_size)

    def oid_to_path(self, oid: str):
        return os.path.join(self.path, "objects", oid[0:2], oid[2:4], oid)

    def exists(self, obj: Union[Pointer, str]):
        oid = obj if isinstance(obj, str) else obj.oid
        path = self.oid_to_path(oid)
        return os.path.exists(path)

    def open(
        self,
        obj: Union[Pointer, str],
        fetch_url: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> BinaryIO:
        oid = obj if isinstance(obj, str) else obj.oid
        path = self.oid_to_path(oid)
        try:
            return open(path, **kwargs)
        except FileNotFoundError:
            if not fetch_url or not isinstance(obj, Pointer):
                raise
        try:
            self.fetch(fetch_url, [obj], batch_size=batch_size)
        except BaseException as exc:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path
            ) from exc
        return open(path, **kwargs)

    def close(self):
        pass


def get_storage_path(scm: "Git") -> str:
    """Return the LFS storage directory for the specified repository."""

    config = scm.get_config()
    git_dir = scm._get_git_dir(scm.root_dir)  # pylint: disable=protected-access
    try:
        path = config.get(("lfs",), "storage")
        if os.path.isabs(path):
            return path
    except KeyError:
        path = "lfs"
    return os.path.join(git_dir, path)
