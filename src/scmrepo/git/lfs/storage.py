import os
from typing import BinaryIO


class LFSStorage:
    def __init__(self, path: str):
        self.path = path

    def oid_to_path(self, oid: str):
        return os.path.join(self.path, "objects", oid[0:2], oid[2:4], oid)

    def open(self, oid: str, **kwargs) -> BinaryIO:
        return open(self.oid_to_path(oid), **kwargs)
