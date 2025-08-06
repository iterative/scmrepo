import locale
import logging
import os
import stat
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from io import BytesIO, StringIO, TextIOWrapper
from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Union,
)
from urllib.parse import urlparse

from funcy import cached_property, reraise

from scmrepo.exceptions import (
    CloneError,
    InvalidRemote,
    MergeConflictError,
    RevError,
    SCMError,
)
from scmrepo.git.backend.base import BaseGitBackend, SyncStatus
from scmrepo.git.config import Config
from scmrepo.git.objects import GitCommit, GitObject, GitTag
from scmrepo.urls import is_scp_style_url
from scmrepo.utils import relpath

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pygit2 import Commit, Oid, Signature
    from pygit2.config import Config as _Pygit2Config
    from pygit2.enums import CheckoutStrategy
    from pygit2.remotes import Remote
    from pygit2.repository import Repository

    from scmrepo.progress import GitProgressEvent


class Pygit2Object(GitObject):
    def __init__(self, obj, backend: Optional["Pygit2Backend"] = None):
        self.obj = obj
        self.backend = backend

    def open(
        self,
        mode: str = "r",
        encoding: Optional[str] = None,
        key: Optional[tuple[str, ...]] = None,
        raw: bool = True,
        rev: Optional[str] = None,
        **kwargs,
    ):
        from pygit2 import BlobIO, GitError

        if not encoding:
            encoding = locale.getpreferredencoding(False)
        if self.backend is not None:
            try:
                if rev:
                    # pylint: disable-next=protected-access
                    commit, _ref = self.backend._resolve_refish(rev)
                else:
                    pass
                if raw:
                    blob_kwargs = {}
                else:
                    assert key is not None
                    path = "/".join(key)
                    blob_kwargs = {
                        "as_path": path,
                        "commit_id": commit.id,
                    }
                blobio = BlobIO(self.obj, **blob_kwargs)
                if mode == "rb":
                    return blobio
                return TextIOWrapper(blobio, encoding=encoding)
            except GitError as exc:
                raise SCMError("failed to read git blob") from exc
        else:
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
        return str(self.obj.id)

    def scandir(self) -> Iterable["Pygit2Object"]:
        for entry in self.obj:
            yield Pygit2Object(entry, backend=self.backend)


class Pygit2Config(Config):
    def __init__(self, config: "_Pygit2Config"):
        self._config = config

    def _key(self, section: tuple[str, ...], name: str) -> str:
        return ".".join((*section, name))

    def get(self, section: tuple[str, ...], name: str) -> str:
        return self._config[self._key(section, name)]

    def get_bool(self, section: tuple[str, ...], name: str) -> bool:
        from pygit2 import GitError

        try:
            return self._config.get_bool(self._key(section, name))
        except GitError as exc:
            raise ValueError("invalid boolean config entry") from exc

    def get_multivar(self, section: tuple[str, ...], name: str) -> Iterator[str]:
        from pygit2 import GitError

        try:
            yield from self._config.get_multivar(self._key(section, name))
        except GitError as exc:
            raise ValueError("invalid multivar config entry") from exc


class Pygit2Backend(BaseGitBackend):  # pylint:disable=abstract-method
    def __init__(  # pylint:disable=W0231
        self, root_dir=os.curdir, search_parent_directories=True
    ):
        import pygit2

        from .filter import LFSFilter

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

        try:
            # NOTE: we want this init to be lazy so we do it on backend init.
            # for subsequent backend instances, this call will error out since
            # the filter is already registered
            pygit2.filter_register("lfs", LFSFilter)  # type: ignore[attr-defined]
        except ValueError:
            pass

    def close(self):
        if hasattr(self, "_refdb"):
            del self._refdb
        self.repo.free()

    @property
    def root_dir(self) -> Optional[str]:
        return self.repo.workdir

    @cached_property
    def _refdb(self):
        from pygit2 import RefdbFsBackend

        return RefdbFsBackend(self.repo)

    def _resolve_refish(self, refish: str):
        from pygit2 import Tag
        from pygit2.enums import ObjectType

        commit, ref = self.repo.resolve_refish(refish)  # type: ignore[attr-defined]
        if isinstance(commit, Tag):
            ref = commit
            commit = commit.peel(ObjectType.COMMIT)  # type: ignore[call-overload]
        return commit, ref

    @property
    def default_signature(self) -> "Signature":
        try:
            return self.repo.default_signature
        except KeyError as exc:
            raise SCMError("Git username and email must be configured") from exc

    @property
    def author(self) -> "Signature":
        return self._get_signature("GIT_AUTHOR")

    @property
    def committer(self) -> "Signature":
        return self._get_signature("GIT_COMMITTER")

    def _get_signature(self, name: str) -> "Signature":
        from pygit2 import Signature

        try:
            sig = self.default_signature
        except SCMError:
            logger.debug("No default signature (config is not set in repo)")
            sig = None

        if os.environ.get(f"{name}_DATE"):
            raise NotImplementedError("signature date override unsupported")

        user_name = os.environ.get(f"{name}_NAME", sig.name if sig else None)
        user_email = os.environ.get(f"{name}_EMAIL", sig.email if sig else None)

        if not user_email or not user_name:
            raise SCMError("Git user name and email must be configured")

        if sig:
            return Signature(
                name=user_name,
                email=user_email,
                time=sig.time,
                offset=sig.offset,
            )

        return Signature(
            name=user_name,
            email=user_email,
        )

    @staticmethod
    def _get_checkout_strategy(
        strategy: Optional["CheckoutStrategy"] = None,
    ) -> "CheckoutStrategy":
        from pygit2.enums import CheckoutStrategy

        if strategy is None:
            strategy = CheckoutStrategy.SAFE | CheckoutStrategy.RECREATE_MISSING
        if os.name == "nt":
            strategy |= CheckoutStrategy.SKIP_LOCKED_DIRECTORIES
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

    def list_submodules(self) -> list[str]:
        return self.repo.listall_submodules()

    @classmethod
    def clone(
        cls,
        url: str,
        to_path: Union[str, os.PathLike[str]],
        shallow_branch: Optional[str] = None,
        progress: Optional[Callable[["GitProgressEvent"], None]] = None,
        bare: bool = False,
        mirror: bool = False,
    ):
        from pygit2 import GitError, clone_repository

        from .callbacks import RemoteCallbacks

        if shallow_branch:
            raise NotImplementedError
        if mirror:
            bare = True
        try:
            with RemoteCallbacks(progress=progress) as cb:
                repo = clone_repository(
                    url, os.fspath(to_path), callbacks=cb, bare=bare
                )
                if mirror:
                    cls._set_mirror(repo, progress=progress)
        except GitError as exc:
            raise CloneError(url, os.fsdecode(to_path)) from exc

    @staticmethod
    def _set_mirror(
        repo: "Repository",
        progress: Optional[Callable[["GitProgressEvent"], None]] = None,
    ):
        from .callbacks import RemoteCallbacks

        url = repo.remotes["origin"].url
        repo.remotes.delete("origin")
        # NOTE: Pygit2 remotes.create("origin", url, fetch_refspec) creates a
        # duplicate config section for each remote config entry. We just edit
        # the config directly so that it creates a single section to be
        # consistent with CLI Git
        repo.config["remote.origin.url"] = url
        repo.config["remote.origin.fetch"] = "+refs/*:refs/*"
        repo.config["remote.origin.mirror"] = True
        with RemoteCallbacks(progress=progress) as cb:
            repo.remotes["origin"].fetch(callbacks=cb)

    @staticmethod
    def init(path: str, bare: bool = False) -> None:
        from pygit2 import init_repository

        init_repository(path, bare=bare)

    @property
    def dir(self) -> str:
        raise NotImplementedError

    def add(
        self,
        paths: Union[str, Iterable[str]],
        update: bool = False,
        force: bool = False,
    ):
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
        from pygit2 import GitError
        from pygit2.enums import CheckoutStrategy

        strategy = self._get_checkout_strategy(
            CheckoutStrategy.FORCE if force else None
        )

        with self.release_odb_handles():
            if create_new:
                commit = self.repo.revparse_single("HEAD")
                new_branch = self.repo.branches.local.create(branch, commit)  # type: ignore[arg-type]
                self.repo.checkout(new_branch, strategy=strategy)  # type: ignore[attr-defined]
            else:
                if branch == "-":
                    branch = "@{-1}"
                try:
                    commit, ref = self._resolve_refish(branch)
                except (KeyError, GitError) as exc:
                    raise RevError(f"unknown Git revision '{branch}'") from exc
                self.repo.checkout_tree(commit, strategy=strategy)  # type: ignore[attr-defined]
                detach = kwargs.get("detach", False)
                if ref and not detach:
                    self.repo.set_head(ref.name)  # type: ignore[attr-defined]
                else:
                    self.repo.set_head(commit.id)  # type: ignore[attr-defined]

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
            self.repo.create_branch(branch, commit)  # type: ignore[arg-type]
        except GitError as exc:
            raise SCMError(f"Failed to create branch '{branch}'") from exc

    def tag(
        self,
        tag: str,
        target: Optional[str] = None,
        annotated: bool = False,
        message: Optional[str] = None,
    ):
        from pygit2 import GitError
        from pygit2.enums import ObjectType

        if annotated and not message:
            raise SCMError("message is required for annotated tag")
        target_obj = self.repo.revparse_single(target or "HEAD")
        with reraise(GitError, SCMError("Failed to create tag")):
            self.repo.create_tag(
                tag,
                target_obj.id,
                ObjectType.COMMIT,
                self.committer,
                message or "",
            )

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
            return self.repo.references["HEAD"].target[11:]  # type: ignore[index]
        return self.repo.head.shorthand

    def active_branch_remote(self) -> str:
        try:
            upstream = self.repo.branches[self.active_branch()].upstream
            if upstream:
                return upstream.remote_name
        except (KeyError, ValueError):
            pass
        raise SCMError("No active branch tracking remote")

    def list_branches(self) -> Iterable[str]:
        base = "refs/heads/"
        return sorted(ref[len(base) :] for ref in self.iter_refs(base))

    def list_tags(self) -> Iterable[str]:
        base = "refs/tags/"
        return sorted(ref[len(base) :] for ref in self.iter_refs(base))

    def list_all_commits(self) -> Iterable[str]:
        import pygit2
        from pygit2.enums import SortMode

        # Add HEAD
        starting_points: list[Union[Oid, str]] = []
        if not self.repo.head_is_unborn:
            starting_points.append(self.repo.head.target)

        # Add all branches, remotes, and tags
        for ref in self.repo.references:
            if ref.startswith(("refs/heads/", "refs/remotes/")):
                oid = self.repo.revparse_single(ref).id
                starting_points.append(oid)
            elif ref.startswith("refs/tags/"):
                tag_obj = self.repo.revparse_single(ref)
                if isinstance(tag_obj, pygit2.Tag):
                    starting_points.append(tag_obj.target)
                else:
                    starting_points.append(tag_obj.id)

        # Walk all commits
        walker = self.repo.walk(None)
        for o in starting_points:
            walker.push(o)
        walker.sort(SortMode.TIME)
        return [str(commit.id) for commit in walker]

    def get_tree_obj(self, rev: str, **kwargs) -> Pygit2Object:
        tree = self.repo[rev].tree  # type: ignore[attr-defined]
        return Pygit2Object(tree, backend=self)

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
            sha = shas.pop()
            if sha is not None:
                return sha
        raise RevError(f"unknown Git revision '{rev}'")

    def resolve_commit(self, rev: str) -> "GitCommit":
        from pygit2 import GitError

        try:
            commit, _ref = self._resolve_refish(rev)
        except (KeyError, GitError) as exc:
            raise SCMError(f"Invalid commit '{rev}'") from exc
        return GitCommit(
            str(commit.id),
            commit.commit_time,
            commit.commit_time_offset * 60,
            commit.message,
            [str(parent) for parent in commit.parent_ids],
            commit.committer.name,
            commit.committer.email,
            commit.author.name,
            commit.author.email,
            commit.author.time,
            commit.author.offset * 60,
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
        from pygit2 import InvalidSpecError, Tag
        from pygit2.enums import ObjectType, ReferenceType

        try:
            ref = self.repo.references.get(name)
        except InvalidSpecError:
            return None
        if not ref:
            return None
        if follow and ref.type == ReferenceType.SYMBOLIC:
            ref = ref.resolve()
        try:
            obj = self.repo[ref.target]
            if isinstance(obj, Tag):
                return str(obj.peel(ObjectType.COMMIT).id)  # type: ignore[call-overload]
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
        except (KeyError, GitError) as exc:
            raise SCMError(f"Invalid rev '{rev}'") from exc

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
        progress: Optional[Callable[["GitProgressEvent"], None]] = None,
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
        from pygit2.enums import MergeAnalysis

        rh_rev = self.resolve_rev(rh)

        if force:
            self.set_ref(lh, rh_rev)
            return SyncStatus.SUCCESS

        try:
            merge_result, _ = self.repo.merge_analysis(rh_rev, lh)
        except KeyError:
            self.set_ref(lh, rh_rev)
            return SyncStatus.SUCCESS

        if merge_result & MergeAnalysis.UP_TO_DATE:
            return SyncStatus.UP_TO_DATE
        if merge_result & MergeAnalysis.FASTFORWARD:
            self.set_ref(lh, rh_rev)
            return SyncStatus.SUCCESS
        if merge_result & MergeAnalysis.NORMAL:
            if on_diverged and on_diverged(lh, rh_rev):
                return SyncStatus.SUCCESS
            return SyncStatus.DIVERGED
        logger.debug("Unexpected merge result: %s", MergeAnalysis.NORMAL)
        raise SCMError("Unknown merge analysis result")

    @contextmanager
    def _get_remote(self, url: str) -> Iterator["Remote"]:
        """Return a pygit2.Remote suitable for the specified Git URL or remote name."""
        try:
            remote = self.repo.remotes[url]
            url = remote.url
        except ValueError:
            pass
        except KeyError as exc:
            raise SCMError(f"'{url}' is not a valid Git remote or URL") from exc

        if os.name == "nt":
            url = url.removeprefix("file://")
        remote = self.repo.remotes.create_anonymous(url)
        parsed = urlparse(remote.url)
        if parsed.scheme in ("git", "git+ssh", "ssh") or is_scp_style_url(remote.url):
            raise NotImplementedError
        yield remote

    def fetch_refspecs(
        self,
        url: str,
        refspecs: Union[str, Iterable[str]],
        force: bool = False,
        on_diverged: Optional[Callable[[str, str], bool]] = None,
        progress: Optional[Callable[["GitProgressEvent"], None]] = None,
        **kwargs,
    ) -> Mapping[str, SyncStatus]:
        import fnmatch

        from pygit2 import GitError

        from .callbacks import RemoteCallbacks

        refspecs = self._refspecs_list(refspecs, force=force)

        # libgit2 rejects diverged refs but does not have a callback to notify
        # when a ref was rejected so we have to determine whether no callback
        # means up to date or rejected
        def _default_status(
            src: str, dst: str, remote_refs: dict[str, "Oid"]
        ) -> SyncStatus:
            try:
                if remote_refs[src] != self.repo.references[dst].target:
                    return SyncStatus.DIVERGED
            except KeyError:
                # remote_refs lookup is skipped when force is set, refs cannot
                # be diverged on force
                pass
            return SyncStatus.UP_TO_DATE

        with self._get_remote(url) as remote:
            with reraise(
                GitError,
                SCMError(f"Git failed to fetch ref from '{url}'"),
            ):
                with RemoteCallbacks(progress=progress) as cb:
                    remote_refs: dict[str, Oid] = (
                        {
                            head["name"]: head["oid"]
                            for head in remote.ls_remotes(callbacks=cb, proxy=True)
                        }
                        if not force
                        else {}
                    )
                    remote.fetch(
                        refspecs=refspecs, callbacks=cb, message="fetch", proxy=True
                    )

            result: dict[str, SyncStatus] = {}
            for refspec in refspecs:
                lh, rh = refspec.split(":")
                if lh.endswith("*"):
                    assert rh.endswith("*")
                    lh_prefix = lh[:-1]
                    rh_prefix = rh[:-1]
                    for refname in remote_refs:
                        if fnmatch.fnmatch(refname, lh):
                            src = refname
                            dst = f"{rh_prefix}{refname[len(lh_prefix) :]}"
                            result[dst] = cb.result.get(
                                src, _default_status(src, dst, remote_refs)
                            )
                else:
                    result[rh] = cb.result.get(lh, _default_status(lh, rh, remote_refs))

        return result

    @staticmethod
    def _refspecs_list(
        refspecs: Union[str, Iterable[str]],
        force: bool = False,
    ) -> list[str]:
        if isinstance(refspecs, str):
            if force and not refspecs.startswith("+"):
                refspecs = f"+{refspecs}"
            return [refspecs]
        if force:
            return [
                (refspec if refspec.startswith("+") else f"+{refspec}")
                for refspec in refspecs
            ]
        return list(refspecs)

    def _stash_iter(self, ref: str):
        raise NotImplementedError

    def _stash_push(
        self,
        ref: str,
        message: Optional[str] = None,
        include_untracked: bool = False,
    ) -> tuple[Optional[str], bool]:
        from scmrepo.git import Stash

        try:
            oid = self.repo.stash(  # type: ignore[attr-defined]
                self.committer,
                message=message,
                include_untracked=include_untracked,
            )
        except KeyError:
            # GIT_ENOTFOUND, nothing to stash
            return None, False
        commit = self.repo[oid]

        if ref != Stash.DEFAULT_STASH:
            self.set_ref(ref, commit.id, message=commit.message)  # type: ignore[attr-defined,arg-type]
            self.repo.stash_drop()  # type: ignore[attr-defined]
        return str(oid), False

    def _stash_apply(
        self,
        rev: str,
        reinstate_index: bool = False,
        skip_conflicts: bool = False,
        **kwargs,
    ):
        from pygit2 import GitError
        from pygit2.enums import CheckoutStrategy

        from scmrepo.git import Stash

        def _apply(index):
            try:
                self.repo.index.read(False)  # type: ignore[attr-defined]
                strategy = self._get_checkout_strategy()
                if skip_conflicts:
                    strategy |= CheckoutStrategy.ALLOW_CONFLICTS
                self.repo.stash_apply(  # type: ignore[attr-defined]
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
            self.repo.stash_drop()  # type: ignore[attr-defined]

    def _stash_drop(self, ref: str, index: int):
        from scmrepo.git import Stash

        if ref != Stash.DEFAULT_STASH:
            raise NotImplementedError

        self.repo.stash_drop(index)  # type: ignore[attr-defined]

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

    def reset(self, hard: bool = False, paths: Optional[Iterable[str]] = None):
        from pygit2 import IndexEntry
        from pygit2.enums import ResetMode

        if self.list_submodules():
            raise NotImplementedError(
                "pygit2 seems to remove files from submodules on reset"
            )

        self.repo.index.read(False)  # type: ignore[attr-defined]
        if paths is not None:
            tree = self.repo.revparse_single("HEAD").tree  # type: ignore[attr-defined]
            for path in paths:
                rel = relpath(path, self.root_dir)
                if os.name == "nt":
                    rel = rel.replace("\\", "/")
                obj = tree[rel]
                self.repo.index.add(IndexEntry(rel, obj.id, obj.filemode))  # type: ignore[attr-defined]
            self.repo.index.write()  # type: ignore[attr-defined]
        elif hard:
            self.repo.reset(self.repo.head.target, ResetMode.HARD)
        else:
            self.repo.reset(self.repo.head.target, ResetMode.MIXED)

    def checkout_index(
        self,
        paths: Optional[Iterable[str]] = None,
        force: bool = False,
        ours: bool = False,
        theirs: bool = False,
    ):
        from pygit2.enums import CheckoutStrategy

        assert not (ours and theirs)
        strategy = CheckoutStrategy.RECREATE_MISSING
        if force or ours or theirs:
            strategy |= CheckoutStrategy.FORCE
        else:
            strategy |= CheckoutStrategy.SAFE

        if ours or theirs:
            strategy |= CheckoutStrategy.ALLOW_CONFLICTS
        strategy = self._get_checkout_strategy(strategy)

        index = self.repo.index  # type: ignore[attr-defined]
        if paths:
            path_list: Optional[list[str]] = [
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
            self.repo.checkout_index(index=index, paths=path_list, strategy=strategy)  # type: ignore[attr-defined]

            if index.conflicts and (ours or theirs):
                for ancestor, ours_entry, theirs_entry in index.conflicts:
                    if not ancestor:
                        continue
                    if ours:
                        entry = ours_entry
                        index.add(ours_entry)
                    else:
                        entry = theirs_entry
                    assert self.root_dir
                    path = os.path.join(self.root_dir, entry.path)
                    with open(path, "wb") as fobj:
                        fobj.write(self.repo.get(entry.id).read_raw())  # type: ignore[attr-defined]
                    index.add(entry.path)
                index.write()

    def status(
        self, ignored: bool = False, untracked_files: str = "all"
    ) -> tuple[Mapping[str, Iterable[str]], Iterable[str], Iterable[str]]:
        from pygit2.enums import FileStatus

        staged: Mapping[str, list[str]] = {
            "add": [],
            "delete": [],
            "modify": [],
        }
        unstaged: list[str] = []
        untracked: list[str] = []

        states = {
            FileStatus.WT_NEW: untracked,
            FileStatus.WT_MODIFIED: unstaged,
            FileStatus.WT_TYPECHANGE: staged["modify"],
            FileStatus.WT_DELETED: staged["modify"],
            FileStatus.WT_RENAMED: staged["modify"],
            FileStatus.INDEX_NEW: staged["add"],
            FileStatus.INDEX_MODIFIED: staged["modify"],
            FileStatus.INDEX_DELETED: staged["delete"],
            FileStatus.WT_UNREADABLE: untracked,
        }

        if untracked_files != "no" and ignored:
            states[FileStatus.IGNORED] = untracked

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

    def merge(  # noqa: C901
        self,
        rev: str,
        commit: bool = True,
        msg: Optional[str] = None,
        squash: bool = False,
    ) -> Optional[str]:
        from pygit2 import GitError
        from pygit2.enums import MergeAnalysis, MergePreference

        if commit and squash:
            raise SCMError("Cannot merge with 'squash' and 'commit'")

        with self.release_odb_handles():
            self.repo.index.read(False)  # type: ignore[attr-defined]
            obj, _ref = self.repo.resolve_refish(rev)  # type: ignore[attr-defined]
            try:
                analysis, ff_pref = self.repo.merge_analysis(obj.id)
            except GitError as exc:
                raise SCMError("Merge analysis failed") from exc

            if analysis == MergeAnalysis.NONE:
                raise SCMError(f"'{rev}' cannot be merged into HEAD")
            if analysis & MergeAnalysis.UP_TO_DATE:
                return None

            try:
                self.repo.merge(obj.id)  # type: ignore[attr-defined]
                self.repo.index.write()  # type: ignore[attr-defined]
            except GitError as exc:
                raise SCMError("Merge failed") from exc

            if self.repo.index.conflicts:  # type: ignore[attr-defined]
                raise MergeConflictError("Merge contained conflicts")

            try:
                if not (squash or ff_pref & MergePreference.NO_FASTFORWARD):
                    if analysis & MergeAnalysis.FASTFORWARD:
                        return self._merge_ff(rev, obj)

                    if analysis & MergeAnalysis.UNBORN:
                        self.repo.set_head(obj.id)  # type: ignore[attr-defined]
                        return str(obj.id)

                if ff_pref & MergePreference.FASTFORWARD_ONLY:
                    raise SCMError(f"Cannot fast-forward HEAD to '{rev}'")

                if commit:
                    return self._merge_commit(msg, obj)

                # --squash merge:
                # HEAD is not moved and merge changes stay in index
                return None
            finally:
                self.repo.state_cleanup()  # type: ignore[attr-defined]
                self.repo.index.write()  # type: ignore[attr-defined]

    def _merge_ff(self, rev: str, obj) -> str:
        if self.repo.head_is_detached:
            self.repo.set_head(obj.id)  # type: ignore[attr-defined]
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
        tree = self.repo.index.write_tree()  # type: ignore[attr-defined]
        merge_commit = self.repo.create_commit(
            "HEAD",
            self.author,
            self.committer,
            msg,
            tree,
            [self.repo.head.target, obj.id],
        )
        return str(merge_commit)

    def validate_git_remote(self, url: str, **kwargs):
        raise NotImplementedError

    def get_remote_url(self, remote: str) -> str:
        try:
            return self.repo.remotes[remote].url
        except KeyError as exc:
            raise InvalidRemote(remote) from exc

    def check_ref_format(self, refname: str):
        raise NotImplementedError

    def get_tag(self, name: str) -> Optional[Union[str, "GitTag"]]:
        from pygit2 import InvalidSpecError, Tag

        try:
            ref = self.repo.references.get(f"refs/tags/{name}")
        except InvalidSpecError:
            return None
        if not ref:
            return None
        try:
            tag = self.repo[ref.target]
            if isinstance(tag, Tag):
                return GitTag(
                    tag.name,
                    str(tag.id),
                    str(tag.target),
                    tag.tagger.name,
                    tag.tagger.email,
                    tag.tagger.time,
                    tag.tagger.offset * 60,
                    tag.message,
                )
        except KeyError:
            pass
        return str(ref.target)

    def get_config(self, path: Optional[str] = None) -> "Config":
        from pygit2.config import Config as _Pygit2Config

        if path:
            return Pygit2Config(_Pygit2Config(path))
        return Pygit2Config(self.repo.config)  # type: ignore[attr-defined]

    def check_attr(
        self,
        path: str,
        attr: str,
        source: Optional[str] = None,
    ) -> Optional[Union[bool, str]]:
        from pygit2 import GitError
        from pygit2.enums import AttrCheck

        commit: Optional[Commit] = None
        flags = AttrCheck.FILE_THEN_INDEX
        if source:
            try:
                commit, _ref = self._resolve_refish(source)
                flags = AttrCheck.INDEX_ONLY | AttrCheck.INCLUDE_COMMIT
            except (KeyError, GitError) as exc:
                raise SCMError(f"Invalid commit '{source}'") from exc
        try:
            return self.repo.get_attr(  # type: ignore[attr-defined]
                path, attr, flags=flags, commit=commit.id if commit else None
            )
        except GitError as exc:
            raise SCMError("Failed to check attribute") from exc
