import errno
import os
import posixpath
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Dict, Optional, Tuple

from fsspec.callbacks import _DEFAULT_CALLBACK
from fsspec.spec import AbstractFileSystem
from fsspec.utils import isfilelike

if TYPE_CHECKING:
    from io import BytesIO

    from scmrepo.git import Git
    from scmrepo.git.objects import GitTrie


class Path:
    def __init__(self, sep, getcwd=None, realpath=None):
        def _getcwd():
            return ""

        self.getcwd = getcwd or _getcwd
        self.realpath = realpath or self.abspath

        assert sep == posixpath.sep
        self.flavour = posixpath

    def chdir(self, path):
        def _getcwd():
            return path

        self.getcwd = _getcwd

    def join(self, *parts):
        return self.flavour.join(*parts)

    def split(self, path):
        return self.flavour.split(path)

    def normpath(self, path):
        return self.flavour.normpath(path)

    def isabs(self, path):
        return self.flavour.isabs(path)

    def abspath(self, path):
        if not self.isabs(path):
            path = self.join(self.getcwd(), path)
        return self.normpath(path)

    def commonprefix(self, path):
        return self.flavour.commonprefix(path)

    def parts(self, path):
        drive, path = self.flavour.splitdrive(path.rstrip(self.flavour.sep))

        ret = []
        while True:
            path, part = self.flavour.split(path)

            if part:
                ret.append(part)
                continue

            if path:
                ret.append(path)

            break

        ret.reverse()

        if drive:
            ret = [drive] + ret

        return tuple(ret)

    def parent(self, path):
        return self.flavour.dirname(path)

    def dirname(self, path):
        return self.parent(path)

    def parents(self, path):
        parts = self.parts(path)
        return tuple(
            self.join(*parts[:length]) for length in range(len(parts) - 1, 0, -1)
        )

    def name(self, path):
        return self.parts(path)[-1]

    def suffix(self, path):
        name = self.name(path)
        _, dot, suffix = name.partition(".")
        return dot + suffix

    def with_name(self, path, name):
        parts = list(self.parts(path))
        parts[-1] = name
        return self.join(*parts)

    def with_suffix(self, path, suffix):
        parts = list(self.parts(path))
        real_path, _, _ = parts[-1].partition(".")
        parts[-1] = real_path + suffix
        return self.join(*parts)

    def isin(self, left, right):
        left_parts = self.parts(left)
        right_parts = self.parts(right)
        left_len = len(left_parts)
        right_len = len(right_parts)
        return left_len > right_len and left_parts[:right_len] == right_parts

    def isin_or_eq(self, left, right):
        return left == right or self.isin(left, right)

    def overlaps(self, left, right):
        # pylint: disable=arguments-out-of-order
        return self.isin_or_eq(left, right) or self.isin(right, left)

    def relpath(self, path, start=None):
        if start is None:
            start = "."
        return self.flavour.relpath(self.abspath(path), start=self.abspath(start))

    def relparts(self, path, start=None):
        return self.parts(self.relpath(path, start=start))

    def as_posix(self, path):
        return path.replace(self.flavour.sep, posixpath.sep)


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
    root_marker = "/"

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

        def _getcwd():
            return self.root_marker

        self.path = Path(self.sep, getcwd=_getcwd)

    def _get_key(self, path: str) -> Tuple[str, ...]:
        path = self.path.abspath(path)
        if path == self.root_marker:
            return ()
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
            # NOTE: to avoid wasting time computing object size, trie.info
            # will return a LazyDict instance, that will compute compute size
            # only when it is accessed.
            ret = self.trie.info(key)
            ret["name"] = path
            return ret
        except KeyError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

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

        paths = [posixpath.join(path, name) if path else name for name in names]

        if not detail:
            return paths

        return [self.info(_path) for _path in paths]

    def get_file(
        self, rpath, lpath, callback=_DEFAULT_CALLBACK, outfile=None, **kwargs
    ):
        # NOTE: temporary workaround while waiting for
        # https://github.com/fsspec/filesystem_spec/pull/1191

        if isfilelike(lpath):
            outfile = lpath
        elif self.isdir(rpath):
            os.makedirs(lpath, exist_ok=True)
            return None

        with self.open(rpath, "rb", **kwargs) as f1:
            if outfile is None:
                outfile = open(lpath, "wb")

            try:
                callback.set_size(getattr(f1, "size", None))
                data = True
                while data:
                    data = f1.read(self.blocksize)
                    segment_len = outfile.write(data)
                    if segment_len is None:
                        segment_len = len(data)
                    callback.relative_update(segment_len)
            finally:
                if not isfilelike(lpath):
                    outfile.close()
