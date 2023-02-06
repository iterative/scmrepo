import locale
import logging
import os
import stat
from contextlib import contextmanager
from io import BytesIO, StringIO
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from funcy import cached_property, reraise
from shortuuid import uuid

from scmrepo.exceptions import CloneError, MergeConflictError, RevError, SCMError
from scmrepo.utils import relpath

from ..objects import GitCommit, GitObject
from .base import BaseGitBackend, SyncStatus

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pygit2.remote import Remote  # type: ignore

    from scmrepo.progress import GitProgressEvent


class Pygit2Object(GitObject):
    def __init__(self, obj):
        self.obj = obj

    def open(self, mode: str = "r", encoding: str = None):
        if not encoding:
            encoding = locale.getpreferredencoding(False)
        data = self.obj.read_raw()
        if mode == "rb":
            return BytesIO(data)
        return StringIO(data.decode(encoding))

    @property
    def name(self) -> str:
        return self.obj.name

    @property
    def mode(self):
        if not self.obj.filemode and self.obj.type_str == "tree":
            return stat.S_IFDIR
        return self.obj.filemode

    @cached_property
    def size(self) -> int:  # pylint: disable=invalid-overridden-method
        # NOTE: obj.size is currently only available for blobs
        if self.obj.type_str == "blob":
            return self.obj.size
        try:
            return len(self.obj.read_raw())
        except KeyError:
            return 0

    @property
    def sha(self) -> str:
        return self.obj.hex

    def scandir(self) -> Iterable["Pygit2Object"]:
        for entry in self.obj:  # noqa: B301
            yield Pygit2Object(entry)


class Pygit2Backend(BaseGitBackend):  # pylint:disable=abstract-method
    def __init__(  # pylint:disable=W0231
        self, root_dir=os.curdir, search_parent_directories=True
    ):
        import pygit2

        if search_parent_directories:
            ceiling_dirs = ""
        else:
            ceiling_dirs = os.path.abspath(root_dir)

        # NOTE: discover_repository will return path/.git/
        path = pygit2.discover_repository(  # pylint:disable=no-member
            os.fspath(root_dir), True, ceiling_dirs
        )
        if not path:
            raise SCMError(f"{root_dir} is not a git repository")

        self.repo = pygit2.Repository(path)

        self._stashes: dict = {}

    def close(self):
        if hasattr(self, "_refdb"):
            del self._refdb
        self.repo.free()

    @property
    def root_dir(self) -> str:
        return self.repo.workdir

    @cached_property
    def _refdb(self):
        from pygit2 import RefdbFsBackend

        return RefdbFsBackend(self.repo)

    def _resolve_refish(self, refish: str):
        from pygit2 import GIT_OBJ_COMMIT, Tag

        commit, ref = self.repo.resolve_refish(refish)
        if isinstance(commit, Tag):
            ref = commit
            commit = commit.peel(GIT_OBJ_COMMIT)
        return commit, ref

    @property
    def default_signature(self):
        try:
            return self.repo.default_signature
        except KeyError as exc:
            raise SCMError("Git username and email must be configured") from exc

    @staticmethod
    def _get_checkout_strategy(strategy: Optional[int] = None):
        from pygit2 import (
            GIT_CHECKOUT_RECREATE_MISSING,
            GIT_CHECKOUT_SAFE,
            GIT_CHECKOUT_SKIP_LOCKED_DIRECTORIES,
        )

        if strategy is None:
            strategy = GIT_CHECKOUT_SAFE | GIT_CHECKOUT_RECREATE_MISSING
        if os.name == "nt":
            strategy |= GIT_CHECKOUT_SKIP_LOCKED_DIRECTORIES
        return strategy

    # Workaround to force git_backend_odb_pack to release open file handles
    # in DVC's mixed git-backend environment.
    # See https://github.com/iterative/dvc/issues/5641
    @contextmanager
    def release_odb_handles(self):
        yield
        # It is safe to free the libgit2 repo context multiple times - free
        # just forces libgit/pygit to release git ODB related contexts which
        # can be reacquired later as needed.
        self.repo.free()

    @staticmethod
    def clone(
        url: str,
        to_path: str,
        shallow_branch: Optional[str] = None,
        progress: Callable[["GitProgressEvent"], None] = None,
    ):
        from pygit2 import GitError, clone_repository

        if shallow_branch:
            raise NotImplementedError
        try:
            clone_repository(url, to_path)
        except GitError as exc:
            raise CloneError(url, to_path) from exc

    @staticmethod
    def init(path: str, bare: bool = False) -> None:
        from pygit2 import init_repository

        init_repository(path, bare=bare)

    @property
    def dir(self) -> str:
        raise NotImplementedError

    def add(self, paths: Union[str, Iterable[str]], update=False):
        raise NotImplementedError

    def commit(self, msg: str, no_verify: bool = False):
        raise NotImplementedError

    def checkout(
        self,
        branch: str,
        create_new: Optional[bool] = False,
        force: bool = False,
        **kwargs,
    ):
        from pygit2 import GIT_CHECKOUT_FORCE, GitError

        strategy = self._get_checkout_strategy(GIT_CHECKOUT_FORCE if force else None)

        with self.release_odb_handles():
            if create_new:
                commit = self.repo.revparse_single("HEAD")
                new_branch = self.repo.branches.local.create(branch, commit)
                self.repo.checkout(new_branch, strategy=strategy)
            else:
                if branch == "-":
                    branch = "@{-1}"
                try:
                    commit, ref = self._resolve_refish(branch)
                except (KeyError, GitError):
                    raise RevError(f"unknown Git revision '{branch}'")
                self.repo.checkout_tree(commit, strategy=strategy)
                detach = kwargs.get("detach", False)
                if ref and not detach:
                    self.repo.set_head(ref.name)
                else:
                    self.repo.set_head(commit.id)

    def fetch(
        self,
        remote: Optional[str] = None,
        force: bool = False,
        unshallow: bool = False,
    ):
        raise NotImplementedError

    def pull(self, **kwargs):
        raise NotImplementedError

    def push(self):
        raise NotImplementedError

    def branch(self, branch: str):
        from pygit2 import GitError

        try:
            commit = self.repo[self.repo.head.target]
            self.repo.create_branch(branch, commit)
        except GitError as exc:
            raise SCMError(f"Failed to create branch '{branch}'") from exc

    def tag(self, tag: str):
        raise NotImplementedError

    def untracked_files(self) -> Iterable[str]:
        raise NotImplementedError

    def is_tracked(self, path: str) -> bool:
        raise NotImplementedError

    def is_dirty(self, untracked_files: bool = False) -> bool:
        raise NotImplementedError

    def active_branch(self) -> str:
        if self.repo.head_is_detached:
            raise SCMError("No active branch (detached HEAD)")
        if self.repo.head_is_unborn:
            # if HEAD points to a nonexistent branch we still return the
            # branch name (without "refs/heads/" prefix) to match gitpython's
            # behavior
            return self.repo.references["HEAD"].target[11:]
        return self.repo.head.shorthand

    def list_branches(self) -> Iterable[str]:
        raise NotImplementedError

    def list_tags(self) -> Iterable[str]:
        raise NotImplementedError

    def list_all_commits(self) -> Iterable[str]:
        raise NotImplementedError

    def get_tree_obj(self, rev: str, **kwargs) -> Pygit2Object:
        tree = self.repo[rev].tree
        return Pygit2Object(tree)

    def get_rev(self) -> str:
        raise NotImplementedError

    def resolve_rev(self, rev: str) -> str:
        from pygit2 import GitError

        try:
            commit, _ref = self._resolve_refish(rev)
            return str(commit.id)
        except (KeyError, GitError):
            pass

        # Look for single exact match in remote refs
        shas = {
            self.get_ref(f"refs/remotes/{remote.name}/{rev}")
            for remote in self.repo.remotes
        } - {None}
        if len(shas) > 1:
            raise RevError(f"ambiguous Git revision '{rev}'")
        if len(shas) == 1:
            return shas.pop()  # type: ignore
        raise RevError(f"unknown Git revision '{rev}'")

    def resolve_commit(self, rev: str) -> "GitCommit":
        from pygit2 import GitError

        try:
            commit, _ref = self._resolve_refish(rev)
        except (KeyError, GitError):
            raise SCMError(f"Invalid commit '{rev}'")
        return GitCommit(
            str(commit.id),
            commit.commit_time,
            commit.commit_time_offset,
            commit.message,
            [str(parent) for parent in commit.parent_ids],
        )

    def _get_stash(self, ref: str):
        raise NotImplementedError

    def is_ignored(self, path: "Union[str, os.PathLike[str]]") -> bool:
        rel = relpath(path, self.root_dir)
        if os.name == "nt":
            rel = rel.replace("\\", "/")
        return self.repo.path_is_ignored(rel)

    def set_ref(
        self,
        name: str,
        new_ref: str,
        old_ref: Optional[str] = None,
        message: Optional[str] = None,
        symbolic: Optional[bool] = False,
    ):
        if old_ref and old_ref != self.get_ref(name, follow=False):
            raise SCMError(f"Failed to set '{name}'")

        if message:
            self._refdb.ensure_log(name)
        if symbolic:
            self.repo.create_reference_symbolic(name, new_ref, True, message=message)
        else:
            self.repo.create_reference_direct(name, new_ref, True, message=message)

    def get_ref(self, name, follow: bool = True) -> Optional[str]:
        from pygit2 import GIT_OBJ_COMMIT, GIT_REF_SYMBOLIC, InvalidSpecError, Tag

        try:
            ref = self.repo.references.get(name)
        except InvalidSpecError:
            return None
        if not ref:
            return None
        if follow and ref.type == GIT_REF_SYMBOLIC:
            ref = ref.resolve()
        try:
            obj = self.repo[ref.target]
            if isinstance(obj, Tag):
                return str(obj.peel(GIT_OBJ_COMMIT).id)
        except ValueError:
            pass

        return str(ref.target)

    def remove_ref(self, name: str, old_ref: Optional[str] = None):
        ref = self.repo.references.get(name)
        if not ref and not old_ref:
            return
        if old_ref and old_ref != str(ref.target):
            raise SCMError(f"Failed to remove '{name}'")
        ref.delete()

    def iter_refs(self, base: Optional[str] = None):
        if base:
            for ref in self.repo.references:
                if ref.startswith(base):
                    yield ref
        else:
            yield from self.repo.references

    def get_refs_containing(self, rev: str, pattern: Optional[str] = None):
        import fnmatch

        from pygit2 import GitError

        def _contains(repo, ref, search_commit):
            commit, _ref = self._resolve_refish(ref)
            base = repo.merge_base(search_commit.id, commit.id)
            return base == search_commit.id

        try:
            search_commit, _ref = self._resolve_refish(rev)
        except (KeyError, GitError):
            raise SCMError(f"Invalid rev '{rev}'")

        if not pattern:
            yield from (
                ref
                for ref in self.iter_refs()
                if _contains(self.repo, ref, search_commit)
            )
            return

        literal = pattern.rstrip("/").split("/")
        for ref in self.iter_refs():
            if (
                ref.split("/")[: len(literal)] == literal
                or fnmatch.fnmatch(ref, pattern)
            ) and _contains(self.repo, ref, search_commit):
                yield ref

    def push_refspecs(
        self,
        url: str,
        refspecs: Union[str, Iterable[str]],
        force: bool = False,
        on_diverged: Optional[Callable[[str, str], bool]] = None,
        progress: Callable[["GitProgressEvent"], None] = None,
        **kwargs,
    ) -> Mapping[str, SyncStatus]:
        raise NotImplementedError

    def _merge_remote_branch(
        self,
        rh: str,
        lh: str,
        force: bool = False,
        on_diverged: Optional[Callable[[str, str], bool]] = None,
    ) -> SyncStatus:
        import pygit2

        rh_rev = self.resolve_rev(rh)

        if force:
            self.set_ref(lh, rh_rev)
            return SyncStatus.SUCCESS

        try:
            merge_result, _ = self.repo.merge_analysis(rh_rev, lh)
        except KeyError:
            self.set_ref(lh, rh_rev)
            return SyncStatus.SUCCESS

        if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
            return SyncStatus.UP_TO_DATE
        if merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
            self.set_ref(lh, rh_rev)
            return SyncStatus.SUCCESS
        if merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
            if on_diverged and on_diverged(lh, rh_rev):
                return SyncStatus.SUCCESS
            return SyncStatus.DIVERGED
        logger.debug("Unexpected merge result: %s", pygit2.GIT_MERGE_ANALYSIS_NORMAL)
        raise SCMError("Unknown merge analysis result")

    @contextmanager
    def get_remote(self, url: str) -> Generator["Remote", None, None]:
        try:
            remote = self.repo.remotes[url]
            url = remote.url
        except ValueError:
            pass
        except KeyError:
            raise SCMError(f"'{url}' is not a valid Git remote or URL")

        if os.name == "nt" and url.startswith("ssh://"):
            raise NotImplementedError
        if os.name == "nt" and url.startswith("file://"):
            url = url[len("file://") :]

        try:
            remote_name = uuid()
            yield self.repo.remotes.create(remote_name, url)
        finally:
            self.repo.remotes.delete(remote_name)

    def fetch_refspecs(
        self,
        url: str,
        refspecs: Union[str, Iterable[str]],
        force: bool = False,
        on_diverged: Optional[Callable[[str, str], bool]] = None,
        progress: Callable[["GitProgressEvent"], None] = None,
        **kwargs,
    ) -> Mapping[str, SyncStatus]:
        from pygit2 import GitError

        if isinstance(refspecs, str):
            refspecs = [refspecs]

        with self.get_remote(url) as remote:
            fetch_refspecs: List[str] = []
            for refspec in refspecs:
                if ":" in refspec:
                    lh, rh = refspec.split(":")
                else:
                    lh = rh = refspec
                if not rh.startswith("refs/"):
                    rh = f"refs/heads/{rh}"
                if not lh.startswith("refs/"):
                    lh = f"refs/heads/{lh}"
                rh = rh[len("refs/") :]
                refspec = f"+{lh}:refs/remotes/{remote.name}/{rh}"
                fetch_refspecs.append(refspec)

            logger.debug("fetch_refspecs: %s", fetch_refspecs)
            with reraise(
                GitError,
                SCMError(f"Git failed to fetch ref from '{url}'"),
            ):
                remote.fetch(refspecs=fetch_refspecs)

            result: Dict[str, "SyncStatus"] = {}
            for refspec in fetch_refspecs:
                _, rh = refspec.split(":")
                if not rh.endswith("*"):
                    refname = rh.split("/", 3)[-1]
                    refname = f"refs/{refname}"
                    result[refname] = self._merge_remote_branch(
                        rh, refname, force, on_diverged
                    )
                    continue
                rh = rh.rstrip("*").rstrip("/") + "/"
                for branch in self.iter_refs(base=rh):
                    refname = f"refs/{branch[len(rh):]}"
                    result[refname] = self._merge_remote_branch(
                        branch, refname, force, on_diverged
                    )
        return result

    def _stash_iter(self, ref: str):
        raise NotImplementedError

    def _stash_push(
        self,
        ref: str,
        message: Optional[str] = None,
        include_untracked: Optional[bool] = False,
    ) -> Tuple[Optional[str], bool]:
        from scmrepo.git import Stash

        oid = self.repo.stash(
            self.default_signature,
            message=message,
            include_untracked=include_untracked,
        )
        commit = self.repo[oid]

        if ref != Stash.DEFAULT_STASH:
            self.set_ref(ref, commit.id, message=commit.message)
            self.repo.stash_drop()
        return str(oid), False

    def _stash_apply(
        self,
        rev: str,
        reinstate_index: bool = False,
        skip_conflicts: bool = False,
        **kwargs,
    ):
        from pygit2 import GIT_CHECKOUT_ALLOW_CONFLICTS, GitError

        from scmrepo.git import Stash

        def _apply(index):
            try:
                self.repo.index.read(False)
                strategy = self._get_checkout_strategy()
                if skip_conflicts:
                    strategy |= GIT_CHECKOUT_ALLOW_CONFLICTS
                self.repo.stash_apply(
                    index, strategy=strategy, reinstate_index=reinstate_index
                )
            except GitError as exc:
                raise MergeConflictError(
                    "Stash apply resulted in merge conflicts"
                ) from exc

        # libgit2 stash apply only accepts refs/stash items by index. If rev is
        # not in refs/stash, we will push it onto the stash, and then pop it
        commit, _ref = self._resolve_refish(rev)
        stash = self.repo.references.get(Stash.DEFAULT_STASH)
        if stash:
            for i, entry in enumerate(stash.log()):
                if entry.oid_new == commit.id:
                    _apply(i)
                    return

        self.set_ref(Stash.DEFAULT_STASH, commit.id, message=commit.message)
        try:
            _apply(0)
        finally:
            self.repo.stash_drop()

    def _stash_drop(self, ref: str, index: int):
        from scmrepo.git import Stash

        if ref != Stash.DEFAULT_STASH:
            raise NotImplementedError

        self.repo.stash_drop(index)

    def _describe(
        self,
        revs: Iterable[str],
        base: Optional[str] = None,
        match: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> Mapping[str, Optional[str]]:
        raise NotImplementedError

    def diff(self, rev_a: str, rev_b: str, binary=False) -> str:
        raise NotImplementedError

    def reset(self, hard: bool = False, paths: Iterable[str] = None):
        from pygit2 import GIT_RESET_HARD, GIT_RESET_MIXED, IndexEntry

        self.repo.index.read(False)
        if paths is not None:
            tree = self.repo.revparse_single("HEAD").tree
            for path in paths:
                rel = relpath(path, self.root_dir)
                if os.name == "nt":
                    rel = rel.replace("\\", "/")
                obj = tree[rel]
                self.repo.index.add(IndexEntry(rel, obj.oid, obj.filemode))
            self.repo.index.write()
        elif hard:
            self.repo.reset(self.repo.head.target, GIT_RESET_HARD)
        else:
            self.repo.reset(self.repo.head.target, GIT_RESET_MIXED)

    def checkout_index(
        self,
        paths: Optional[Iterable[str]] = None,
        force: bool = False,
        ours: bool = False,
        theirs: bool = False,
    ):
        from pygit2 import (
            GIT_CHECKOUT_ALLOW_CONFLICTS,
            GIT_CHECKOUT_FORCE,
            GIT_CHECKOUT_RECREATE_MISSING,
            GIT_CHECKOUT_SAFE,
        )

        assert not (ours and theirs)
        strategy = GIT_CHECKOUT_RECREATE_MISSING
        if force or ours or theirs:
            strategy |= GIT_CHECKOUT_FORCE
        else:
            strategy |= GIT_CHECKOUT_SAFE

        if ours or theirs:
            strategy |= GIT_CHECKOUT_ALLOW_CONFLICTS
        strategy = self._get_checkout_strategy(strategy)

        index = self.repo.index
        if paths:
            path_list: Optional[List[str]] = [
                relpath(path, self.root_dir) for path in paths
            ]
            if os.name == "nt":
                path_list = [
                    path.replace("\\", "/")
                    for path in path_list  # type: ignore[union-attr]
                ]
        else:
            path_list = None

        with self.release_odb_handles():
            self.repo.checkout_index(index=index, paths=path_list, strategy=strategy)

            if index.conflicts and (ours or theirs):
                for ancestor, ours_entry, theirs_entry in index.conflicts:
                    if not ancestor:
                        continue
                    if ours:
                        entry = ours_entry
                        index.add(ours_entry)
                    else:
                        entry = theirs_entry
                    path = os.path.join(self.root_dir, entry.path)
                    with open(path, "wb") as fobj:
                        fobj.write(self.repo.get(entry.id).read_raw())
                    index.add(entry.path)
                index.write()

    def status(
        self, ignored: bool = False, untracked_files: str = "all"
    ) -> Tuple[Mapping[str, Iterable[str]], Iterable[str], Iterable[str]]:
        from pygit2 import (
            GIT_STATUS_IGNORED,
            GIT_STATUS_INDEX_DELETED,
            GIT_STATUS_INDEX_MODIFIED,
            GIT_STATUS_INDEX_NEW,
            GIT_STATUS_WT_DELETED,
            GIT_STATUS_WT_MODIFIED,
            GIT_STATUS_WT_NEW,
            GIT_STATUS_WT_RENAMED,
            GIT_STATUS_WT_TYPECHANGE,
            GIT_STATUS_WT_UNREADABLE,
        )

        staged: Mapping[str, List[str]] = {
            "add": [],
            "delete": [],
            "modify": [],
        }
        unstaged: List[str] = []
        untracked: List[str] = []

        states = {
            GIT_STATUS_WT_NEW: untracked,
            GIT_STATUS_WT_MODIFIED: unstaged,
            GIT_STATUS_WT_TYPECHANGE: staged["modify"],
            GIT_STATUS_WT_DELETED: staged["modify"],
            GIT_STATUS_WT_RENAMED: staged["modify"],
            GIT_STATUS_INDEX_NEW: staged["add"],
            GIT_STATUS_INDEX_MODIFIED: staged["modify"],
            GIT_STATUS_INDEX_DELETED: staged["delete"],
            GIT_STATUS_WT_UNREADABLE: untracked,
        }

        if untracked_files != "no" and ignored:
            states[GIT_STATUS_IGNORED] = untracked

        for file, state in self.repo.status(
            untracked_files=untracked_files, ignored=ignored
        ).items():
            for git_state in states:
                flag = state & git_state
                if flag:
                    states[flag].append(file)

        return (
            {status: paths for status, paths in staged.items() if paths},
            unstaged,
            untracked,
        )

    def iter_remote_refs(self, url: str, base: Optional[str] = None, **kwargs):
        raise NotImplementedError

    def merge(
        self,
        rev: str,
        commit: bool = True,
        msg: Optional[str] = None,
        squash: bool = False,
    ) -> Optional[str]:
        from pygit2 import (
            GIT_MERGE_ANALYSIS_FASTFORWARD,
            GIT_MERGE_ANALYSIS_NONE,
            GIT_MERGE_ANALYSIS_UNBORN,
            GIT_MERGE_ANALYSIS_UP_TO_DATE,
            GIT_MERGE_PREFERENCE_FASTFORWARD_ONLY,
            GIT_MERGE_PREFERENCE_NO_FASTFORWARD,
            GitError,
        )

        if commit and squash:
            raise SCMError("Cannot merge with 'squash' and 'commit'")

        with self.release_odb_handles():
            self.repo.index.read(False)
            obj, _ref = self.repo.resolve_refish(rev)
            try:
                analysis, ff_pref = self.repo.merge_analysis(obj.id)
            except GitError as exc:
                raise SCMError("Merge analysis failed") from exc

            if analysis == GIT_MERGE_ANALYSIS_NONE:
                raise SCMError(f"'{rev}' cannot be merged into HEAD")
            if analysis & GIT_MERGE_ANALYSIS_UP_TO_DATE:
                return None

            try:
                self.repo.merge(obj.id)
                self.repo.index.write()
            except GitError as exc:
                raise SCMError("Merge failed") from exc

            if self.repo.index.conflicts:
                raise MergeConflictError("Merge contained conflicts")

            try:
                if not (squash or ff_pref & GIT_MERGE_PREFERENCE_NO_FASTFORWARD):
                    if analysis & GIT_MERGE_ANALYSIS_FASTFORWARD:
                        return self._merge_ff(rev, obj)

                    if analysis & GIT_MERGE_ANALYSIS_UNBORN:
                        self.repo.set_head(obj.id)
                        return str(obj.id)

                if ff_pref & GIT_MERGE_PREFERENCE_FASTFORWARD_ONLY:
                    raise SCMError(f"Cannot fast-forward HEAD to '{rev}'")

                if commit:
                    return self._merge_commit(msg, obj)

                # --squash merge:
                # HEAD is not moved and merge changes stay in index
                return None
            finally:
                self.repo.state_cleanup()
                self.repo.index.write()

    def _merge_ff(self, rev: str, obj) -> str:
        if self.repo.head_is_detached:
            self.repo.set_head(obj.id)
        else:
            branch = self.get_ref("HEAD", follow=False)
            assert branch
            self.set_ref(
                branch,
                str(obj.id),
                message=f"merge {rev}: Fast-forward",
            )
        return str(obj.id)

    def _merge_commit(self, msg: Optional[str], obj) -> str:
        if not msg:
            raise SCMError("Merge commit message is required")
        user = self.default_signature
        tree = self.repo.index.write_tree()
        merge_commit = self.repo.create_commit(
            "HEAD",
            user,
            user,
            msg,
            tree,
            [self.repo.head.target, obj.id],
        )
        return str(merge_commit)

    def validate_git_remote(self, url: str, **kwargs):
        raise NotImplementedError

    def check_ref_format(self, refname: str):
        raise NotImplementedError
