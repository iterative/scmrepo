from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class LFSObject:
    oid: str
    size: int

    def __str__(self) -> str:
        return self.oid

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LFSObject":
        return cls(**{k: v for k, v in d.items() if k in fields(cls)})
