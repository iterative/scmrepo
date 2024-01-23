import hashlib
import io
import logging
from dataclasses import dataclass
from typing import IO, BinaryIO, TextIO, Tuple

logger = logging.getLogger(__name__)


LFS_VERSION = "https://git-lfs.github.com/spec/v1"
LEGACY_LFS_VERSION = "https://hawser.github.com/spec/v1"
ALLOWED_VERSIONS = (LFS_VERSION, LEGACY_LFS_VERSION)
HEADERS = [(b"version " + version.encode("utf-8")) for version in ALLOWED_VERSIONS]


def _get_kv(line: str) -> Tuple[str, str]:
    key, value = line.strip().split(maxsplit=1)
    return key, value


@dataclass
class Pointer:
    oid: str
    size: int

    def __init__(self, oid: str, size: int, **kwargs):
        self.oid = oid
        self.size = size
        self._dict = kwargs

    def __hash__(self):
        return hash(self.dump())

    @classmethod
    def build(cls, fobj: BinaryIO) -> "Pointer":
        m = hashlib.sha256()
        data = fobj.read()
        m.update(data)
        return cls(m.hexdigest(), len(data))

    @classmethod
    def load(cls, fobj: IO) -> "Pointer":
        """Load the specified pointer file."""

        if isinstance(fobj, io.TextIOBase):  # type: ignore[unreachable]
            text_obj: TextIO = fobj  # type: ignore[unreachable]

        else:
            text_obj = io.TextIOWrapper(fobj, encoding="utf-8")

        cls.check_version(text_obj.readline())
        d = dict(_get_kv(line) for line in text_obj.readlines())
        try:
            value = d.pop("oid")
            hash_method, oid = value.split(":", maxsplit=1)
            if hash_method != "sha256":
                raise ValueError("Invalid LFS hash method '{hash_method}'")
        except ValueError as e:
            raise ValueError("Invalid LFS pointer oid") from e
        try:
            value = d.pop("size")
            size = int(value)
        except ValueError as e:
            raise ValueError("Invalid LFS pointer size") from e

        return cls(oid, size, **d)

    @staticmethod
    def check_version(line: str):
        try:
            key, value = _get_kv(line)
            if key != "version":
                raise ValueError("LFS pointer file must start with 'version'")
            if value not in ALLOWED_VERSIONS:
                raise ValueError(f"Unsupported LFS pointer version '{value}'")
        except (ValueError, OSError) as e:
            raise ValueError("Invalid LFS pointer file") from e

    def dump(self) -> str:
        d = {
            "oid": f"sha256:{self.oid}",
            "size": self.size,
        }
        d.update(self._dict)
        return "\n".join(
            [f"version {LFS_VERSION}"] + [f"{key} {d[key]}" for key in sorted(d)] + [""]
        )

    def to_bytes(self) -> bytes:
        return self.dump().encode("utf-8")


if __name__ == "__main__":
    # Minimal `git lfs pointer` CLI implementation
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Build generated pointer files.",
    )
    parser.add_argument("--file", help="A local file to build the pointer from.")
    args = parser.parse_args()
    if not args.file:
        sys.exit("Nothing to do")

    with open(args.file, mode="rb") as fobj_:
        p = Pointer.build(fobj_)
