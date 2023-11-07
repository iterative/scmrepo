import errno
import os
from typing import TYPE_CHECKING, BinaryIO, Collection, Optional, Union

from dvc_objects.fs.callbacks import TqdmCallback

from .pointer import Pointer

if TYPE_CHECKING:
    from scmrepo.git import Git


class LFSStorage:
    def __init__(self, path: str):
        self.path = path

    def fetch(self, url: str, objects: Collection[Pointer]):
        from .client import LFSClient

        with TqdmCallback(desc="Fetching LFS objects", unit="obj") as cb:
            cb.set_size(len(objects))
            with LFSClient.from_git_url(url) as client:
                client.download(self, objects, callback=cb)

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
            self.fetch(fetch_url, [obj])
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
