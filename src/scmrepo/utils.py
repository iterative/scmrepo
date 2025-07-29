import os
from collections.abc import Iterator, MutableMapping
from typing import Callable, TypeVar, Union

K = TypeVar("K")
V = TypeVar("V")


class LazyDict(MutableMapping[K, V]):
    def __init__(self, values: dict[K, Union[V, Callable[[], V]]]):
        self._values = values

    def __getitem__(self, item):
        value = self._values[item]
        if callable(value):
            value = value()
            self._values[item] = value
        return value

    def __setitem__(self, key: K, value: Union[V, Callable[[], V]]) -> None:
        self._values[key] = value

    def __delitem__(self, key: K) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def relpath(path, start=os.curdir):
    path = os.fspath(path)
    start = os.path.abspath(os.fspath(start))

    # Windows path on different drive than curdir doesn't have relpath
    if os.name == "nt":
        # Since python 3.8 os.realpath resolves network shares to their UNC
        # path. So, to be certain that relative paths correctly captured,
        # we need to resolve to UNC path first. We resolve only the drive
        # name so that we don't follow any 'real' symlinks on the path
        def resolve_network_drive_windows(path_to_resolve):
            drive, tail = os.path.splitdrive(path_to_resolve)
            return os.path.join(os.path.realpath(drive), tail)

        path = resolve_network_drive_windows(os.path.abspath(path))
        start = resolve_network_drive_windows(start)
        if not os.path.commonprefix([start, path]):
            return path
    return os.path.relpath(path, start)
