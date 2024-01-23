import datetime
import stat
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, cast

from pygtrie import Trie

S_IFGITLINK = 0o160000


def S_ISGITLINK(m: int) -> bool:  # noqa: N802
    return stat.S_IFMT(m) == S_IFGITLINK


def _to_datetime(time: int, offset: int) -> datetime.datetime:
    tz = datetime.timezone(datetime.timedelta(seconds=offset))
    return datetime.datetime.fromtimestamp(time, tz=tz)


class GitObject(ABC):
    @abstractmethod
    def open(self, mode: str = "r", encoding: Optional[str] = None, **kwargs):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def mode(self) -> int:
        pass

    @abstractmethod
    def scandir(self) -> Iterable["GitObject"]:
        pass

    @property
    def isfile(self) -> bool:
        return stat.S_ISREG(self.mode)

    @property
    def isdir(self) -> bool:
        return stat.S_ISDIR(self.mode)

    @property
    def issubmodule(self) -> bool:
        return S_ISGITLINK(self.mode)

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    @abstractmethod
    def sha(self) -> str:
        pass


class GitTrie:
    def __init__(self, tree: GitObject, rev: str):
        self.tree = tree
        self.rev = rev
        self.trie = Trie()

        self.trie[()] = tree
        self._build(tree, ())

    def _build(self, tree: GitObject, path: tuple):
        for obj in tree.scandir():
            obj_path = (*path, obj.name)
            self.trie[obj_path] = obj

            if obj.isdir:
                self._build(obj, obj_path)

    def open(
        self,
        key: tuple,
        mode: Optional[str] = "r",
        encoding: Optional[str] = None,
        raw: bool = True,
    ):
        obj = self.trie[key]
        if obj.isdir:
            raise IsADirectoryError

        return obj.open(mode=mode, encoding=encoding, key=key, raw=raw, rev=self.rev)

    def exists(self, key: tuple) -> bool:
        return bool(self.trie.has_node(key))

    def isdir(self, key: tuple) -> bool:
        try:
            obj = self.trie[key]
        except KeyError:
            return False
        return obj.isdir

    def isfile(self, key: tuple) -> bool:
        try:
            obj = self.trie[key]
        except KeyError:
            return False

        return obj.isfile

    def ls(self, key: tuple):
        ret = []

        def node_factory(_, _key, children, obj):
            if key == _key:
                assert obj.isdir
                list(filter(None, children))
            else:
                ret.append(_key[-1])

        self.trie.traverse(node_factory, prefix=key)

        return ret

    def walk(self, top: tuple, topdown: Optional[bool] = True):
        dirs = []
        nondirs = []

        for name in self.ls(top):
            info = self.info((*top, name))
            if info["type"] == "directory":
                dirs.append(name)
            else:
                nondirs.append(name)

        if topdown:
            yield top, dirs, nondirs

        for dname in dirs:
            yield from self.walk((*top, dname), topdown=topdown)

        if not topdown:
            yield top, dirs, nondirs

    def info(self, key: tuple) -> dict:
        from scmrepo.utils import LazyDict

        obj = self.trie[key]

        def size():
            return obj.size

        ret = LazyDict(
            {
                "size": size,
                "type": "directory" if stat.S_ISDIR(obj.mode) else "file",
                "sha": obj.sha,
                "mode": obj.mode,
            }
        )

        return cast(dict, ret)


@dataclass
class GitCommit:
    hexsha: str
    commit_time: int
    commit_time_offset: int
    message: str
    parents: List[str]
    committer_name: str
    committer_email: str
    author_name: str
    author_email: str
    author_time: int
    author_time_offset: int

    @property
    def commit_datetime(self) -> datetime.datetime:
        return _to_datetime(self.commit_time, self.commit_time_offset)

    @property
    def author_datetime(self) -> datetime.datetime:
        return _to_datetime(self.author_time, self.author_time_offset)


@dataclass
class GitTag:
    name: str
    hexsha: str  # SHA for the tag object itself
    target: str  # SHA for the object the tag points to
    tagger_name: str
    tagger_email: str
    tag_time: int
    tag_time_offset: int
    message: str

    @property
    def tag_datetime(self) -> datetime.datetime:
        return _to_datetime(self.tag_time, self.tag_time_offset)
