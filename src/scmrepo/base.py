"""Manages source control systems (e.g. Git) in DVC."""

from contextlib import AbstractContextManager


class Base(AbstractContextManager):
    """Base class for source control management driver implementations."""

    def __init__(self, root_dir=None):
        import os

        self._root_dir = os.path.realpath(root_dir or os.curdir)

    @property
    def root_dir(self) -> str:
        return self._root_dir

    def __repr__(self):
        return f"{type(self).__name__}: '{self.dir}'"

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def dir(self):
        """Path to a directory with SCM specific information."""
        return None

    @staticmethod
    def is_repo(root_dir):  # pylint: disable=unused-argument
        """Returns whether or not root_dir is a valid SCM repository."""
        return True

    @staticmethod
    def is_submodule(root_dir):  # pylint: disable=unused-argument
        """Returns whether or not root_dir is a valid SCM repository
        submodule.
        """
        return True

    def is_ignored(self, path):  # pylint: disable=unused-argument
        """Returns whether or not path is ignored by SCM."""
        return False

    def ignore(self, path):  # pylint: disable=unused-argument
        """Makes SCM ignore a specified path."""

    def ignore_remove(self, path):  # pylint: disable=unused-argument
        """Makes SCM stop ignoring a specified path."""

    @property
    def ignore_file(self):
        """Filename for a file that contains ignored paths for this SCM."""

    def add(self, paths):
        """Makes SCM track every path from a specified list of paths."""

    def commit(self, msg):
        """Makes SCM create a commit."""

    def checkout(self, branch, create_new=False):
        """Makes SCM checkout a branch."""

    def branch(self, branch):
        """Makes SCM create a branch with a specified name."""

    def tag(self, tag):
        """Makes SCM create a tag with a specified name."""

    def untracked_files(self):
        """Returns a list of untracked files."""
        return []

    def is_tracked(self, path):  # pylint: disable=unused-argument
        """Returns whether or not a specified path is tracked."""
        return False

    def is_dirty(self):
        """Return whether the SCM contains uncommitted changes."""
        return False

    def active_branch(self):
        """Returns current branch in the repo."""
        return ""

    def list_branches(self):
        """Returns a list of available branches in the repo."""
        return []

    def list_tags(self):
        """Returns a list of available tags in the repo."""
        return []

    def list_all_commits(self):
        """Returns a list of commits in the repo."""
        return []

    def belongs_to_scm(self, path):
        """Return boolean whether file belongs to scm"""

    def close(self):
        """Method to close the files"""

    def _reset(self) -> None:
        pass

    def list_submodules(self) -> list[str]:
        """Returns a list of submodules in the repo."""
        return []
