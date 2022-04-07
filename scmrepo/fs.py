import errno
import os
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Optional,
    Tuple,
)

from fsspec.spec import AbstractFileSystem

if TYPE_CHECKING:
    from io import BytesIO

    from scmrepo.git import Git
    from scmrepo.git.objects import GitTrie


def bytesio_len(obj: "BytesIO") -> Optional[int]:
    try:
        offset = obj.tell()
        length = obj.seek(0, os.SEEK_END)
        obj.seek(offset)
    except (AttributeError, OSError):
        return None
    return length


class GitFileSystem(AbstractFileSystem):
    # pylint: disable=abstract-method
    cachable = False

    def __init__(
        self,
        path: str = None,
        rev: str = None,
        scm: "Git" = None,
        trie: "GitTrie" = None,
        rev_resolver: Callable[["Git", str], str] = None,
        **kwargs,
    ):
        from scmrepo.git import Git
        from scmrepo.git.objects import GitTrie

        super().__init__(**kwargs)
        if not trie:
            scm = scm or Git(path)
            resolver = rev_resolver or Git.resolve_rev
            resolved = resolver(scm, rev or "HEAD")
            tree_obj = scm.pygit2.get_tree_obj(rev=resolved)
            trie = GitTrie(tree_obj, resolved)

        self.trie = trie
        self.rev = self.trie.rev

    def _get_key(self, path: str) -> Tuple[str, ...]:
        relparts = path.split(self.sep)
        if relparts and relparts[0] in (".", ""):
            relparts = relparts[1:]
        return tuple(relparts)

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int = None,
        autocommit: bool = True,
        cache_options: Dict = None,
        **kwargs: Any,
    ) -> BinaryIO:
        if mode != "rb":
            raise NotImplementedError

        key = self._get_key(path)
        try:
            obj = self.trie.open(key, mode=mode)
            obj.size = bytesio_len(obj)
            return obj
        except KeyError as exc:
            msg = os.strerror(errno.ENOENT) + f" in branch '{self.rev}'"
            raise FileNotFoundError(errno.ENOENT, msg, path) from exc
        except IsADirectoryError as exc:
            raise IsADirectoryError(
                errno.EISDIR, os.strerror(errno.EISDIR), path
            ) from exc

    def info(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        key = self._get_key(path)
        try:
            return {
                **self.trie.info(key),
                "name": path,
            }
        except KeyError:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path
            )

    def exists(self, path: str, **kwargs: Any) -> bool:
        key = self._get_key(path)
        return self.trie.exists(key)

    def checksum(self, path: str) -> str:
        return self.info(path)["sha"]

    def ls(self, path, detail=True, **kwargs):
        info = self.info(path)
        if info["type"] != "directory":
            return [info] if detail else [path]

        key = self._get_key(path)
        try:
            names = self.trie.ls(key)
        except KeyError as exc:
            raise FileNotFoundError from exc

        paths = [
            self.sep.join((path, name)) if path else name for name in names
        ]

        if not detail:
            return paths

        return [self.info(_path) for _path in paths]
