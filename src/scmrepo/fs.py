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
        path: Optional[str] = None,
        rev: Optional[str] = None,
        scm: "Git" = None,
        trie: "GitTrie" = None,
        rev_resolver: Optional[Callable[["Git", str], str]] = None,
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
        self._cwd = self.root_marker

    def getcwd(self):
        return self._cwd

    def chdir(self, path):
        self._cwd = path

    @classmethod
    def join(cls, *parts):
        return posixpath.join(*parts)

    @classmethod
    def split(cls, path):
        return posixpath.split(path)

    def normpath(self, path):
        return posixpath.normpath(path)

    @classmethod
    def isabs(cls, path):
        return posixpath.isabs(path)

    def abspath(self, path):
        if not self.isabs(path):
            path = self.join(self.getcwd(), path)
        return self.normpath(path)

    @classmethod
    def commonprefix(cls, path):
        return posixpath.commonprefix(path)

    @classmethod
    def parts(cls, path):
        ret = []
        while True:
            path, part = cls.split(path)

            if part:
                ret.append(part)
                continue

            if path:
                ret.append(path)

            break

        ret.reverse()

        return tuple(ret)

    @classmethod
    def parent(cls, path):
        return posixpath.dirname(path)

    @classmethod
    def dirname(cls, path):
        return cls.parent(path)

    @classmethod
    def parents(cls, path):
        parts = cls.parts(path)
        return tuple(
            cls.join(*parts[:length]) for length in range(len(parts) - 1, 0, -1)
        )

    @classmethod
    def name(cls, path):
        return cls.parts(path)[-1]

    @classmethod
    def suffix(cls, path):
        name = cls.name(path)
        _, dot, suffix = name.partition(".")
        return dot + suffix

    @classmethod
    def with_name(cls, path, name):
        parts = list(cls.parts(path))
        parts[-1] = name
        return cls.join(*parts)

    @classmethod
    def with_suffix(cls, path, suffix):
        parts = list(cls.parts(path))
        real_path, _, _ = parts[-1].partition(".")
        parts[-1] = real_path + suffix
        return cls.join(*parts)

    @classmethod
    def isin(cls, left, right):
        left_parts = cls.parts(left)
        right_parts = cls.parts(right)
        left_len = len(left_parts)
        right_len = len(right_parts)
        return left_len > right_len and left_parts[:right_len] == right_parts

    @classmethod
    def isin_or_eq(cls, left, right):
        return left == right or cls.isin(left, right)

    @classmethod
    def overlaps(cls, left, right):
        # pylint: disable=arguments-out-of-order
        return cls.isin_or_eq(left, right) or cls.isin(right, left)

    def relpath(self, path, start=None):
        if start is None:
            start = "."
        return posixpath.relpath(self.abspath(path), start=self.abspath(start))

    def relparts(self, path, start=None):
        return self.parts(self.relpath(path, start=start))

    @classmethod
    def as_posix(cls, path):
        return path

    def _get_key(self, path: str) -> Tuple[str, ...]:
        path = self.abspath(path)
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
        block_size: Optional[int] = None,
        autocommit: bool = True,
        cache_options: Optional[Dict] = None,
        raw: bool = False,
        **kwargs: Any,
    ) -> BinaryIO:
        if mode != "rb":
            raise NotImplementedError

        key = self._get_key(path)
        try:
            obj = self.trie.open(key, mode=mode, raw=raw)
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
        except KeyError as exc:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path
            ) from exc

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
                outfile = open(lpath, "wb")  # noqa: SIM115

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
