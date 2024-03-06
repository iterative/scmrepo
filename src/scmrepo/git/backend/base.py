import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Union

from scmrepo.exceptions import SCMError
from scmrepo.git.objects import GitObject

if TYPE_CHECKING:
    from scmrepo.git.config import Config
    from scmrepo.git.objects import GitCommit, GitTag
    from scmrepo.progress import GitProgressEvent


class NoGitBackendError(SCMError):
    def __init__(self, func):
        super().__init__(f"No valid Git backend for '{func}'")


class SyncStatus(Enum):
    SUCCESS = 0
    UP_TO_DATE = 1
    DIVERGED = 2


class BaseGitBackend(ABC):
    """Base Git backend class."""

    @abstractmethod
    def __init__(self, root_dir=os.curdir, search_parent_directories=True):
        pass

    def close(self):  # noqa: B027
        pass

    @abstractmethod
    def is_ignored(self, path: str) -> bool:
        """Return True if the specified path is gitignored."""

    @property
    @abstractmethod
    def root_dir(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def clone(
        url: str,
        to_path: Union[str, os.PathLike[str]],
        shallow_branch: Optional[str] = None,
        progress: Optional[Callable[["GitProgressEvent"], None]] = None,
        bare: bool = False,
        mirror: bool = False,
    ):
        pass

    @staticmethod
    @abstractmethod
    def init(path: str, bare: bool = False) -> None:
        pass

    @property
    @abstractmethod
    def dir(self) -> str:
        pass

    @abstractmethod
    def add(
        self,
        paths: Union[str, Iterable[str]],
        update: bool = False,
        force: bool = False,
    ):
        pass

    @abstractmethod
    def commit(self, msg: str, no_verify: bool = False):
        pass

    @abstractmethod
    def checkout(
        self,
        branch: str,
        create_new: Optional[bool] = False,
        force: bool = False,
        **kwargs,
    ):
        pass

    @abstractmethod
    def fetch(
        self,
        remote: Optional[str] = None,
        force: bool = False,
        unshallow: bool = False,
    ):
        pass

    @abstractmethod
    def pull(self, **kwargs):
        pass

    @abstractmethod
    def push(self):
        pass

    @abstractmethod
    def branch(self, branch: str):
        pass

    @abstractmethod
    def tag(
        self,
        tag: str,
        target: Optional[str] = None,
        annotated: bool = False,
        message: Optional[str] = None,
    ):
        pass

    @abstractmethod
    def untracked_files(self) -> Iterable[str]:
        pass

    @abstractmethod
    def is_tracked(self, path: str) -> bool:
        pass

    @abstractmethod
    def is_dirty(self, untracked_files: bool = False) -> bool:
        pass

    @abstractmethod
    def active_branch(self) -> str:
        pass

    @abstractmethod
    def active_branch_remote(self) -> str:
        """Return the fetch remote name for the current branch."""

    @abstractmethod
    def list_branches(self) -> Iterable[str]:
        pass

    @abstractmethod
    def list_tags(self) -> Iterable[str]:
        pass

    @abstractmethod
    def list_all_commits(self) -> Iterable[str]:
        pass

    @abstractmethod
    def get_tree_obj(self, rev: str, **kwargs) -> GitObject:
        pass

    @abstractmethod
    def get_rev(self) -> str:
        pass

    @abstractmethod
    def resolve_rev(self, rev: str) -> str:
        pass

    @abstractmethod
    def resolve_commit(self, rev: str) -> "GitCommit":
        pass

    @abstractmethod
    def set_ref(
        self,
        name: str,
        new_ref: str,
        old_ref: Optional[str] = None,
        message: Optional[str] = None,
        symbolic: Optional[bool] = False,
    ):
        """Set the specified git ref.

        Optional kwargs:
            old_ref: If specified, ref will only be set if it currently equals
                old_ref. Has no effect is symbolic is True.
            message: Optional reflog message.
            symbolic: If True, ref will be set as a symbolic ref to new_ref
                rather than the dereferenced object.
        """

    @abstractmethod
    def get_ref(self, name: str, follow: bool = True) -> Optional[str]:
        """Return the value of specified ref.

        If follow is false, symbolic refs will not be dereferenced.
        Returns None if the ref does not exist.
        """

    @abstractmethod
    def remove_ref(self, name: str, old_ref: Optional[str] = None):
        """Remove the specified git ref.

        If old_ref is specified, ref will only be removed if it currently
        equals old_ref.
        """

    @abstractmethod
    def iter_refs(self, base: Optional[str] = None):
        """Iterate over all refs in this git repo.

        If base is specified, only refs which begin with base will be yielded.
        """

    @abstractmethod
    def iter_remote_refs(self, url: str, base: Optional[str] = None, **kwargs):
        """Iterate over all refs in the specified remote Git repo.

        If base is specified, only refs which begin with base will be yielded.
        URL can be a named Git remote or URL.
        """

    @abstractmethod
    def get_refs_containing(self, rev: str, pattern: Optional[str] = None):
        """Iterate over all git refs containing the specified revision."""

    @abstractmethod
    def push_refspecs(
        self,
        url: str,
        refspecs: Union[str, Iterable[str]],
        force: bool = False,
        on_diverged: Optional[Callable[[str, str], bool]] = None,
        progress: Optional[Callable[["GitProgressEvent"], None]] = None,
        **kwargs,
    ) -> Mapping[str, SyncStatus]:
        """Push refspec to a remote Git repo.

        Args:
            url: Git remote name or absolute Git URL.
            refspecs: Iterable containing refspecs to push.
                Note that this will not match subkeys.
            force: If True, remote refs will be overwritten.
            on_diverged: Callback function which will be called if local ref
                and remote have diverged and force is False. If the callback
                returns True the remote ref will be overwritten.
                Callback will be of the form:
                    on_diverged(local_refname, remote_sha)
        """

    @abstractmethod
    def fetch_refspecs(
        self,
        url: str,
        refspecs: Union[str, Iterable[str]],
        force: bool = False,
        on_diverged: Optional[Callable[[str, str], bool]] = None,
        progress: Optional[Callable[["GitProgressEvent"], None]] = None,
        **kwargs,
    ) -> Mapping[str, SyncStatus]:
        """Fetch refspecs from a remote Git repo.

        Args:
            url: Git remote name or absolute Git URL.
            refspecs: Iterable containing refspecs to fetch.
                Note that this will not match subkeys.
            force: If True, local refs will be overwritten.
            on_diverged: Callback function which will be called if local ref
                and remote have diverged and force is False. If the callback
                returns True the local ref will be overwritten.
                Callback will be of the form:
                    on_diverged(local_refname, remote_sha)

        Returns:
            Mapping of local_refname to sync status.
        """

    @abstractmethod
    def _stash_iter(self, ref: str):
        """Iterate over stash commits in the specified ref."""

    @abstractmethod
    def _stash_push(
        self,
        ref: str,
        message: Optional[str] = None,
        include_untracked: bool = False,
    ) -> tuple[Optional[str], bool]:
        """Push a commit onto the specified stash.

        Returns a tuple of the form (rev, need_reset) where need_reset
        indicates whether or not the workspace should be `reset --hard`
        (some backends will not clean the workspace after creating a stash
        commit).
        """

    @abstractmethod
    def _stash_apply(
        self,
        rev: str,
        reinstate_index: bool = False,
        skip_conflicts: bool = False,
        **kwargs,
    ):
        """Apply the specified stash revision."""

    @abstractmethod
    def _stash_drop(self, ref: str, index: int):
        """Drop the specified stash revision."""

    @abstractmethod
    def _describe(
        self,
        revs: Iterable[str],
        base: Optional[str] = None,
        match: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> Mapping[str, Optional[str]]:
        """Return the first ref which points to each revs.

        Roughly equivalent to `git describe --all --exact-match`.

        If no matching ref was found, returns None.

        Optional kwargs:
            base: Base ref prefix to search, defaults to "refs/tags"
            match: Glob pattern. If specified only refs matching this glob
                pattern will be returned.
            exclude: Glob pattern. If specified, only refs which do not match
                this pattern will be returned.
        """

    @abstractmethod
    def diff(self, rev_a: str, rev_b: str, binary=False) -> str:
        """Return the git diff for two commits."""

    @abstractmethod
    def reset(self, hard: bool = False, paths: Optional[Iterable[str]] = None):
        """Reset current git HEAD."""

    @abstractmethod
    def checkout_index(
        self,
        paths: Optional[Iterable[str]] = None,
        force: bool = False,
        ours: bool = False,
        theirs: bool = False,
    ):
        """Checkout the specified paths from index."""

    @abstractmethod
    def status(
        self, ignored: bool = False, untracked_files: str = "all"
    ) -> tuple[Mapping[str, Iterable[str]], Iterable[str], Iterable[str]]:
        """Return tuple of (staged_files, unstaged_files, untracked_files).

        staged_files will be a dict mapping status (add, delete, modify) to a
        list of paths.

        If ignored is True, gitignored files will be included in
        untracked_paths.

        untracked_files can be one of the following:
          - "no" return no untracked files
          - "normal" (git cli default) return untracked files and directories
          - "all" (default) return all untracked files in untracked directories

        Using "no" or "normal" will be faster than "all" when large untracked
        directories are present in the workspace, as collecting all untracked
        files can take some time.
        """

    def _reset(self) -> None:  # noqa: B027
        pass

    @abstractmethod
    def merge(
        self,
        rev: str,
        commit: bool = True,
        msg: Optional[str] = None,
        squash: bool = False,
    ) -> Optional[str]:
        """Merge specified commit into the current (working copy) branch.

        Args:
            rev: Git revision to merge.
            commit: If True, merge will be committed using `msg` as the commit
                message. If False, the merge changes will be staged but not
                committed.
            msg: Merge commit message, required if `commit` is True.
            squash: If True, working tree state will be updated with merge
                results, but no commit will be made and no changes will be
                staged. `commit` must be False when `squash` is True.

        If conflicts are present after merging, MergeConflictError will be
        raised. The caller is responsible for either resolving the conflicts
        or resetting the repository state.

        Returns revision of the merge commit or None if no commit was made.
        """

    @abstractmethod
    def validate_git_remote(self, url: str, **kwargs):
        """Verify that url is a valid git URL or remote name."""

    @abstractmethod
    def get_remote_url(self, remote: str) -> str:
        """Return URL for the specified remote."""

    @abstractmethod
    def check_ref_format(self, refname: str) -> bool:
        """Check if a reference name is well formed."""

    @abstractmethod
    def get_tag(self, name: str) -> Optional[Union[str, "GitTag"]]:
        """Return the specified tag object.

        Args:
            name: Tag name (without 'refs/tags/' prefix).

        Returns:
            None if the specified tag does not exist.
            String SHA for the target object if the tag is a lightweight tag.
            GitTag object if the tag is an annotated tag.
        """

    @abstractmethod
    def get_config(self, path: Optional[str] = None) -> "Config":
        """Return a Git config object.

        Args:
            path: If set, a config object for the specified config file will be
                returned. By default, the standard Git system/global/repo config
                stack object will be returned.
        """

    @abstractmethod
    def check_attr(
        self,
        path: str,
        attr: str,
        source: Optional[str] = None,
    ) -> Optional[Union[bool, str]]:
        """Return the value of the specified attribute for a pathname.

        Args:
            path: Pathname to check.
            attr: Attribute to check.
            source: Optional tree-ish source to check.

        Returns:
            None when the attribute is not defined for the path (unspecified).
            True when the attribute is defined as true (set).
            False when the attribute is defined as false (unset).
            The value of the attribute when a value has been assigned.
        """
