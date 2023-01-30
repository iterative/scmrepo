"""git stash convenience wrapper."""

import logging
from typing import Optional

from scmrepo.exceptions import SCMError

logger = logging.getLogger(__name__)


class Stash:
    """Wrapper for representing any arbitrary Git stash ref."""

    DEFAULT_STASH = "refs/stash"

    def __init__(self, scm, ref: Optional[str] = None):
        self.ref = ref if ref else self.DEFAULT_STASH
        self.scm = scm

    def __iter__(self):
        yield from self.scm._stash_iter(self.ref)

    def __len__(self):
        return len(self.list())

    def __getitem__(self, index):
        return self.list()[index]

    def list(self):
        return list(iter(self))

    def push(
        self, message: Optional[str] = None, include_untracked: bool = False
    ) -> Optional[str]:
        if not self.scm.is_dirty(untracked_files=include_untracked):
            logger.debug("No changes to stash")
            return None

        logger.debug("Stashing changes in '%s'", self.ref)
        rev, reset = self.scm._stash_push(  # pylint: disable=protected-access
            self.ref, message=message, include_untracked=include_untracked
        )
        if reset:
            self.scm.reset(hard=True)
        return rev

    def pop(self, **kwargs):
        """Pop the last stash commit.

        Supports the same keyword arguments as apply().
        """
        logger.debug("Popping from stash '%s'", self.ref)
        ref = f"{self.ref}@{{0}}"
        rev = self.scm.resolve_rev(ref)
        try:
            self.apply(rev, **kwargs)
        except Exception as exc:
            raise SCMError("Could not apply stash commit") from exc
        self.drop()
        return rev

    def apply(
        self,
        rev: str,
        reinstate_index: bool = False,
        skip_conflicts: bool = False,
    ):
        """Apply a stash commit.

        Arguments:
            rev: Stash commit to apply.
            reinstate_index: If True, stashed index changes will be reapplied.
            skip_conflicts: If True, conflicting changes will be skipped and
                will not be applied from the stash. By default, apply will
                fail if any conflicts are found.
        """
        logger.debug("Applying stash commit '%s'", rev)
        self.scm._stash_apply(  # pylint: disable=protected-access
            rev, reinstate_index=reinstate_index, skip_conflicts=skip_conflicts
        )

    def drop(self, index: int = 0):
        if index < 0 or index >= len(self):
            raise SCMError(f"Invalid stash ref '{self.ref}@{{{index}}}'")
        logger.debug("Dropping '%s@{%d}'", self.ref, index)
        self.scm._stash_drop(self.ref, index)  # pylint: disable=protected-access

    def clear(self):
        logger.debug("Clear stash '%s'", self.ref)
        for _ in range(len(self)):
            self.drop()
