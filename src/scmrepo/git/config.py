"""git config convenience wrapper."""
import logging
from abc import ABC, abstractmethod
from typing import Iterator, Tuple

logger = logging.getLogger(__name__)


class Config(ABC):
    """Read-only Git config."""

    @abstractmethod
    def get(self, section: Tuple[str, ...], name: str) -> str:
        """Return the specified setting as a string.

        Raises:
            KeyError: Option was not set.
        """

    @abstractmethod
    def get_bool(self, section: Tuple[str, ...], name: str) -> bool:
        """Return the specified setting as a boolean.

        Raises:
            KeyError: Option was not set.
            ValueError: Option is not a valid boolean.
        """

    @abstractmethod
    def get_multivar(self, section: Tuple[str, ...], name: str) -> Iterator[str]:
        """Iterate over string values in the specified multivar setting.

        Raises:
            KeyError: Option was not set.
        """
