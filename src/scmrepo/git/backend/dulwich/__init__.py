import fnmatch
import locale
import logging
import os
import re
import stat
from contextlib import closing
from functools import partial
from io import BytesIO, StringIO
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from funcy import cached_property, reraise

from scmrepo.exceptions import AuthError, CloneError, InvalidRemote, RevError, SCMError
from scmrepo.progress import GitProgressReporter
from scmrepo.utils import relpath

from ...objects import GitObject, GitTag
from ..base import BaseGitBackend, SyncStatus

if TYPE_CHECKING:
    from dulwich.client import SSHVendor
    from dulwich.repo import Repo

    from scmrepo.progress import GitProgressEvent

    from ...objects import GitCommit


logger = logging.getLogger(__name__)


class DulwichObject(GitObject):
    def __init__(self, repo, name, mode, sha):
        self.repo = repo
        self._name = name
        self._mode = mode
        self._sha = sha

    def open(self, mode: str = "r", encoding: str = None):
        if not encoding:
            encoding = locale.getpreferredencoding(False)
        # NOTE: we didn't load the object before as Dulwich will also try to
        # load the contents of it into memory, which will slow down Trie
        # building considerably.
        obj = self.repo[self._sha]
        data = obj.as_raw_string()
        if mode == "rb":
            return BytesIO(data)
        return StringIO(data.decode(encoding))

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> int:
        return self._mode

    def scandir(self) -> Iterable["DulwichObject"]:
        tree = self.repo[self._sha]
        for entry in tree.iteritems():  # noqa: B301
            yield DulwichObject(self.repo, entry.path.decode(), entry.mode, entry.sha)

    @cached_property
    def size(self) -> int:  # pylint: disable=invalid-overridden-method
        try:
            return self.repo[self._sha].raw_length()
        except KeyError:
            return 0

    @property
    def sha(self) -> str:
        return self._sha.decode("ascii")


class DulwichProgressReporter(GitProgressReporter):
    """Dulwich progress reporter.

    Works with both dulwich.porcelain methods which expect an 'errstream'
    stream object and internal dulwich methods which expect a 'progress' stream
    write method.
    """

    def write(self, msg: Union[str, bytes]) -> int:
        self(msg)
        return len(msg)


def _get_ssh_vendor() -> "SSHVendor":
    import shutil
    import sys

    from dulwich.client import SubprocessSSHVendor

    from .asyncssh_vendor import AsyncSSHVendor, get_unsupported_opts

    ssh_command = os.environ.get("GIT_SSH_COMMAND", os.environ.get("GIT_SSH"))
    if ssh_command:
        logger.debug("dulwich: Using environment GIT_SSH_COMMAND '%s'", ssh_command)
        return SubprocessSSHVendor()

    if sys.platform == "win32" and os.environ.get("MSYSTEM") and shutil.which("ssh"):
        # see https://github.com/iterative/dvc/issues/7702
        logger.debug(
            "dulwich: native win32 Python inside MSYS2/git-bash, using MSYS2 OpenSSH"
        )
        return SubprocessSSHVendor()

    default_config = os.path.expanduser(os.path.join("~", ".ssh", "config"))
    unsupported = list(get_unsupported_opts([default_config]))
    if unsupported and shutil.which("ssh"):
        logger.debug(
            "dulwich: unsupported SSH config option(s) '%s', using system OpenSSH",
            ", ".join(unsupported),
        )
        return SubprocessSSHVendor()
    return AsyncSSHVendor()


class DulwichBackend(BaseGitBackend):  # pylint:disable=abstract-method
    """Dulwich Git backend."""

    from dulwich import client

    from .client import GitCredentialsHTTPClient

    # monkeypatch dulwich client's default SSH vendor to use asyncssh
    client.get_ssh_vendor = _get_ssh_vendor  # type: ignore[assignment]
    # monkeypatch dulwich client's default HTTPClient to add support for
    # git credential helpers. See https://github.com/jelmer/dulwich/pull/976
    client.HttpGitClient = GitCredentialsHTTPClient  # type: ignore[assignment]

    # Dulwich progress will return messages equivalent to git CLI,
    # our pbars should just display the messages as formatted by dulwich
    BAR_FMT_NOTOTAL = "{desc}{bar:b}|{postfix[info]} [{elapsed}]"

    def __init__(  # pylint:disable=W0231
        self, root_dir=os.curdir, search_parent_directories=True
    ):
        from dulwich.errors import NotGitRepository
        from dulwich.repo import Repo

        try:
            if search_parent_directories:
                self.repo = Repo.discover(start=root_dir)
            else:
                self.repo = Repo(root_dir)
        except NotGitRepository as exc:
            raise SCMError(f"{root_dir} is not a git repository") from exc

        self._submodules: Dict[str, str] = self._find_submodules()
        self._stashes: dict = {}

    def _find_submodules(self) -> Dict[str, str]:
        """Return dict mapping submodule names to submodule paths.

        Submodule paths will be relative to Git repo root.
        """
        from dulwich.config import ConfigFile, parse_submodules

        submodules: Dict[str, str] = {}
        config_path = os.path.join(self.root_dir, ".gitmodules")
        if os.path.isfile(config_path):
            config = ConfigFile.from_path(config_path)
            for path, _url, section in parse_submodules(config):
                submodules[os.fsdecode(section)] = os.fsdecode(path)
        return submodules

    def close(self):
        self.repo.close()

    @property
    def root_dir(self) -> str:
        return self.repo.path

    @classmethod
    def clone(
        cls,
        url: str,
        to_path: str,
        shallow_branch: Optional[str] = None,
        progress: Callable[["GitProgressEvent"], None] = None,
        bare: bool = False,
        mirror: bool = False,
    ):
        from urllib.parse import urlparse

        from dulwich.porcelain import NoneStream
        from dulwich.porcelain import clone as git_clone

        if mirror:
            bare = True
        parsed = urlparse(url)
        try:
            clone_from = partial(
                git_clone,
                url,
                target=to_path,
                errstream=(
                    DulwichProgressReporter(progress) if progress else NoneStream()
                ),
                bare=bare,
            )
            if shallow_branch:
                # NOTE: dulwich only supports shallow/depth for non-local
                # clones. This differs from CLI git, where depth is used for
                # file:// URLs but not direct local paths
                if parsed.scheme in ("git", "git+ssh", "ssh", "http", "https"):
                    depth = 1
                else:
                    depth = 0
                repo = clone_from(depth=depth, branch=os.fsencode(shallow_branch))
            else:
                repo = clone_from()

            with closing(repo):
                if mirror:
                    cls._set_mirror(repo, progress=progress)
                else:
                    cls._set_default_tracking_branch(repo)
        except Exception as exc:
            raise CloneError(url, to_path) from exc

    @staticmethod
    def _set_default_tracking_branch(repo: "Repo"):
        from dulwich.refs import LOCAL_BRANCH_PREFIX, parse_symref_value

        try:
            ref = parse_symref_value(repo.refs.read_ref(b"HEAD"))
        except ValueError:
            return
        if ref.startswith(LOCAL_BRANCH_PREFIX):
            branch = ref[len(LOCAL_BRANCH_PREFIX) :]
            config = repo.get_config()
            section = ("branch", os.fsencode(branch))
            config.set(section, b"remote", b"origin")
            config.set(section, b"merge", ref)

    @staticmethod
    def _set_mirror(
        repo: "Repo", progress: Callable[["GitProgressEvent"], None] = None
    ):
        from dulwich.porcelain import NoneStream, fetch

        config = repo.get_config()
        section = config[(b"remote", b"origin")]
        try:
            del section[b"fetch"]
        except KeyError:
            pass
        section[b"fetch"] = b"+refs/*:refs/*"
        section[b"mirror"] = b"true"
        config.write_to_path()
        fetch(
            repo,
            remote_location=b"origin",
            errstream=(DulwichProgressReporter(progress) if progress else NoneStream()),
        )

    @staticmethod
    def init(path: str, bare: bool = False) -> None:
        from dulwich.porcelain import init

        init(path, bare=bare)

    @property
    def dir(self) -> str:
        return self.repo.commondir()

    def add(
        self,
        paths: Union[str, Iterable[str]],
        update: bool = False,
        force: bool = False,
    ):
        assert paths or update

        paths = [paths] if isinstance(paths, str) else list(paths)

        if update and not paths:
            self.repo.stage(list(self.repo.open_index()))
            return

        files: List[bytes] = [
            os.fsencode(fpath) for fpath in self._expand_paths(paths, force=force)
        ]
        if update:
            index = self.repo.open_index()
            if os.name == "nt":
                # NOTE: we need git/unix separator to compare against index
                # paths but repo.stage() expects to be called with OS paths
                self.repo.stage(
                    [fname for fname in files if fname.replace(b"\\", b"/") in index]
                )
            else:
                self.repo.stage([fname for fname in files if fname in index])
        else:
            self.repo.stage(files)

    def _expand_paths(self, paths: List[str], force: bool = False) -> Iterator[str]:
        for path in paths:
            if not os.path.isabs(path) and self._submodules:
                # NOTE: If path is inside a submodule, Dulwich expects the
                # staged paths to be relative to the submodule root (not the
                # parent git repo root). We append path to root_dir here so
                # that the result of relpath(path, root_dir) is actually the
                # path relative to the submodule root.
                fs_path = relpath(path, self.root_dir)
                for sm_path in self._submodules.values():
                    if fs_path.startswith(sm_path):
                        path = os.path.join(
                            self.root_dir,
                            relpath(fs_path, sm_path),
                        )
                        break
            if os.path.isdir(path):
                for root, _, fs in os.walk(path):
                    for fpath in fs:
                        rel = relpath(os.path.join(root, fpath), self.root_dir)
                        if force or not self.ignore_manager.is_ignored(rel):
                            yield rel
            else:
                rel = relpath(path, self.root_dir)
                if force or not self.ignore_manager.is_ignored(rel):
                    yield rel

    def commit(self, msg: str, no_verify: bool = False):
        from dulwich.errors import CommitError
        from dulwich.porcelain import Error, TimezoneFormatError, commit
        from dulwich.repo import InvalidUserIdentity

        with reraise((Error, CommitError), SCMError("Git commit failed")):
            try:
                commit(self.root_dir, message=msg, no_verify=no_verify)
            except InvalidUserIdentity as exc:
                raise SCMError("Git username and email must be configured") from exc
            except TimezoneFormatError as exc:
                raise SCMError("Invalid Git timestamp") from exc

    def checkout(
        self,
        branch: str,
        create_new: Optional[bool] = False,
        force: bool = False,
        **kwargs,
    ):
        raise NotImplementedError

    def fetch(
        self,
        remote: Optional[str] = None,
        force: bool = False,
        unshallow: bool = False,
    ):
        from dulwich.porcelain import Error, fetch
        from dulwich.protocol import DEPTH_INFINITE

        with reraise(Error, SCMError("Git fetch failed")):
            remote_b = os.fsencode(remote) if remote else b"origin"
            fetch(
                self.repo,
                remote_location=remote_b,
                force=force,
                depth=DEPTH_INFINITE if unshallow else None,
            )

    def pull(self, **kwargs):
        raise NotImplementedError

    def push(self):
        raise NotImplementedError

    def branch(self, branch: str):
        from dulwich.porcelain import Error, branch_create

        try:
            branch_create(self.root_dir, branch)
        except Error as exc:
            raise SCMError(f"Failed to create branch '{branch}'") from exc

    def tag(
        self,
        tag: str,
        target: Optional[str] = None,
        annotated: bool = False,
        message: Optional[str] = None,
    ):
        from dulwich.porcelain import Error, tag_create

        if annotated and not message:
            raise SCMError("message is required for annotated tag")
        with reraise(Error, SCMError("Failed to create tag")):
            tag_create(
                self.repo,
                os.fsencode(tag),
                objectish=target or "HEAD",
                annotated=annotated,
                message=message.encode("utf-8") if message else None,
            )

    def untracked_files(self) -> Iterable[str]:
        _staged, _unstaged, untracked = self.status()
        return untracked

    def is_tracked(self, path: str) -> bool:
        rel = relpath(path, self.root_dir).replace(os.path.sep, "/").encode()
        rel_dir = rel + b"/"
        for p in self.repo.open_index():
            if p == rel or p.startswith(rel_dir):
                return True
        return False

    def is_dirty(self, untracked_files: bool = False) -> bool:
        kwargs: Dict[str, Any] = {} if untracked_files else {"untracked_files": "no"}
        return any(self.status(**kwargs))

    def active_branch(self) -> str:
        raise NotImplementedError

    def list_branches(self) -> Iterable[str]:
        base = "refs/heads/"
        return sorted(ref[len(base) :] for ref in self.iter_refs(base))

    def list_tags(self) -> Iterable[str]:
        base = "refs/tags/"
        return sorted(ref[len(base) :] for ref in self.iter_refs(base))

    def list_all_commits(self) -> Iterable[str]:
        raise NotImplementedError

    def get_tree_obj(self, rev: str, **kwargs) -> DulwichObject:
        from dulwich.objectspec import parse_tree

        tree = parse_tree(self.repo, rev)
        return DulwichObject(self.repo, ".", stat.S_IFDIR, tree.id)

    def get_rev(self) -> str:
        rev = self.get_ref("HEAD")
        if rev:
            return rev
        raise SCMError("Empty git repo")

    def resolve_rev(self, rev: str) -> str:
        raise NotImplementedError

    def resolve_commit(self, rev: str) -> "GitCommit":
        raise NotImplementedError

    def _get_stash(self, ref: str):
        from dulwich.stash import Stash as DulwichStash

        if ref not in self._stashes:
            self._stashes[ref] = DulwichStash(self.repo, ref=os.fsencode(ref))
        return self._stashes[ref]

    @cached_property
    def ignore_manager(self):
        from dulwich.ignore import IgnoreFilterManager

        return IgnoreFilterManager.from_repo(self.repo)

    def is_ignored(self, path: "Union[str, os.PathLike[str]]") -> bool:
        # `is_ignored` returns `false` if excluded in `.gitignore` and
        # `None` if it's not mentioned at all. `True` if it is ignored.
        relative_path = relpath(path, self.root_dir)
        # if checking a directory, a trailing slash must be included
        if str(path)[-1] == os.sep:
            relative_path += os.sep
        return bool(self.ignore_manager.is_ignored(relative_path))

    def set_ref(
        self,
        name: str,
        new_ref: str,
        old_ref: Optional[str] = None,
        message: Optional[str] = None,
        symbolic: Optional[bool] = False,
    ):
        name_b = os.fsencode(name)
        new_ref_b = os.fsencode(new_ref)
        old_ref_b = os.fsencode(old_ref) if old_ref else None
        message_b = message.encode("utf-8") if message else None
        if symbolic:
            return self.repo.refs.set_symbolic_ref(name_b, new_ref_b, message=message_b)
        if not self.repo.refs.set_if_equals(
            name_b, old_ref_b, new_ref_b, message=message_b
        ):
            raise SCMError(f"Failed to set '{name}'")

    def get_ref(self, name, follow: bool = True) -> Optional[str]:
        from dulwich.objects import Tag
        from dulwich.refs import parse_symref_value

        name_b = os.fsencode(name)
        if follow:
            try:
                ref = self.repo.refs[name_b]
            except KeyError:
                ref = None
        else:
            ref = self.repo.refs.read_ref(name_b)
            try:
                if ref:
                    ref = parse_symref_value(ref)
            except ValueError:
                pass
        if ref:
            if ref in self.repo and isinstance(self.repo[ref], Tag):
                ref = self.repo.get_peeled(name_b)
            return os.fsdecode(ref)
        return None

    def remove_ref(self, name: str, old_ref: Optional[str] = None):
        name_b = name.encode("utf-8")
        old_ref_b = old_ref.encode("utf-8") if old_ref else None
        if not self.repo.refs.remove_if_equals(name_b, old_ref_b):
            raise SCMError(f"Failed to remove '{name}'")

    def iter_refs(self, base: Optional[str] = None):
        base_b = os.fsencode(base) if base else None
        for key in self.repo.refs.keys(base=base_b):
            if base:
                if base.endswith("/"):
                    base = base[:-1]
                yield "/".join([base, os.fsdecode(key)])
            else:
                yield os.fsdecode(key)

    def iter_remote_refs(self, url: str, base: Optional[str] = None, **kwargs):
        from dulwich.client import HTTPUnauthorized, get_transport_and_path
        from dulwich.errors import NotGitRepository
        from dulwich.porcelain import get_remote_repo

        try:
            _remote, location = get_remote_repo(self.repo, url)
            client, path = get_transport_and_path(location, **kwargs)
        except Exception as exc:
            raise InvalidRemote(url) from exc

        try:
            if base:
                yield from (
                    os.fsdecode(ref)
                    for ref in client.get_refs(path)
                    if ref.startswith(os.fsencode(base))
                )
            else:
                yield from (os.fsdecode(ref) for ref in client.get_refs(path))
        except NotGitRepository as exc:
            raise InvalidRemote(url) from exc
        except HTTPUnauthorized as exc:
            raise AuthError(url) from exc

    def get_refs_containing(self, rev: str, pattern: Optional[str] = None):
        raise NotImplementedError

    def push_refspecs(  # noqa: C901
        self,
        url: str,
        refspecs: Union[str, Iterable[str]],
        force: bool = False,
        on_diverged: Optional[Callable[[str, str], bool]] = None,
        progress: Callable[["GitProgressEvent"], None] = None,
        **kwargs,
    ) -> Mapping[str, SyncStatus]:
        from dulwich.client import HTTPUnauthorized, get_transport_and_path
        from dulwich.errors import NotGitRepository, SendPackError
        from dulwich.objectspec import parse_reftuples
        from dulwich.porcelain import DivergedBranches, check_diverged, get_remote_repo

        try:
            _remote, location = get_remote_repo(self.repo, url)
            client, path = get_transport_and_path(location, **kwargs)
        except Exception as exc:
            raise SCMError(f"'{url}' is not a valid Git remote or URL") from exc

        change_result = {}
        selected_refs = []

        def update_refs(refs):
            from dulwich.objects import ZERO_SHA

            selected_refs.extend(
                parse_reftuples(self.repo.refs, refs, refspecs, force=force)
            )
            new_refs = {}
            for lh, rh, _ in selected_refs:
                refname = os.fsdecode(rh)
                if rh in refs and lh is not None:
                    if refs[rh] == self.repo.refs[lh]:
                        change_result[refname] = SyncStatus.UP_TO_DATE
                        continue
                    try:
                        check_diverged(self.repo, refs[rh], self.repo.refs[lh])
                    except DivergedBranches:
                        if not force:
                            overwrite = (
                                on_diverged(os.fsdecode(lh), os.fsdecode(refs[rh]))
                                if on_diverged
                                else False
                            )
                            if not overwrite:
                                change_result[refname] = SyncStatus.DIVERGED
                                continue

                if lh is None:
                    value = ZERO_SHA
                else:
                    value = self.repo.refs[lh]

                new_refs[rh] = value
                change_result[refname] = SyncStatus.SUCCESS

            return new_refs

        try:
            result = client.send_pack(
                path,
                update_refs,
                generate_pack_data=self.repo.generate_pack_data,
                progress=(DulwichProgressReporter(progress) if progress else None),
            )
        except (NotGitRepository, SendPackError) as exc:
            src = [lh for (lh, _, _) in selected_refs]
            raise SCMError(f"Git failed to push '{src}' to '{url}'") from exc
        except HTTPUnauthorized as exc:
            raise AuthError(url) from exc
        if result.ref_status and any(
            (value is not None) for value in result.ref_status.values()
        ):
            reasons = ", ".join(
                (
                    f"{os.fsdecode(ref)}: {reason}"
                    for ref, reason in result.ref_status.items()
                    if reason is not None
                )
            )
            raise SCMError(f"Git failed to push some refs to '{url}' ({reasons})")
        return change_result

    def fetch_refspecs(
        self,
        url: str,
        refspecs: Union[str, Iterable[str]],
        force: bool = False,
        on_diverged: Optional[Callable[[str, str], bool]] = None,
        progress: Callable[["GitProgressEvent"], None] = None,
        **kwargs,
    ) -> Mapping[str, SyncStatus]:
        from dulwich.client import get_transport_and_path
        from dulwich.errors import NotGitRepository
        from dulwich.objectspec import parse_reftuples
        from dulwich.porcelain import DivergedBranches, check_diverged, get_remote_repo
        from dulwich.refs import DictRefsContainer

        fetch_refs = []

        def determine_wants(
            remote_refs: Dict[bytes, bytes],
            depth: Optional[int] = None,  # pylint: disable=unused-argument
        ) -> List[bytes]:
            fetch_refs.extend(
                parse_reftuples(
                    DictRefsContainer(remote_refs),
                    self.repo.refs,
                    os.fsencode(refspecs)
                    if isinstance(refspecs, str)
                    else [os.fsencode(refspec) for refspec in refspecs],
                    force=force,
                )
            )
            return [
                remote_refs[lh]
                for (lh, _, _) in fetch_refs
                if remote_refs[lh] not in self.repo.object_store
            ]

        with reraise(Exception, SCMError(f"'{url}' is not a valid Git remote or URL")):
            _remote, location = get_remote_repo(self.repo, url)
            client, path = get_transport_and_path(location, **kwargs)

        with reraise(
            (NotGitRepository, KeyError),
            SCMError(f"Git failed to fetch ref from '{url}'"),
        ):
            fetch_result = client.fetch(
                path,
                self.repo,
                progress=DulwichProgressReporter(progress) if progress else None,
                determine_wants=determine_wants,
            )

            result = {}

            for lh, rh, _ in fetch_refs:
                refname = os.fsdecode(rh)
                if rh in self.repo.refs:
                    if self.repo.refs[rh] == fetch_result.refs[lh]:
                        result[refname] = SyncStatus.UP_TO_DATE
                        continue
                    try:
                        check_diverged(
                            self.repo,
                            self.repo.refs[rh],
                            fetch_result.refs[lh],
                        )
                    except DivergedBranches:
                        if not force:
                            overwrite = (
                                on_diverged(
                                    os.fsdecode(rh),
                                    os.fsdecode(fetch_result.refs[lh]),
                                )
                                if on_diverged
                                else False
                            )
                            if not overwrite:
                                result[refname] = SyncStatus.DIVERGED
                                continue

                self.repo.refs[rh] = fetch_result.refs[lh]
                result[refname] = SyncStatus.SUCCESS
        return result

    def _stash_iter(self, ref: str):
        stash = self._get_stash(ref)
        yield from stash.stashes()

    def _stash_push(
        self,
        ref: str,
        message: Optional[str] = None,
        include_untracked: bool = False,
    ) -> Tuple[Optional[str], bool]:
        from dulwich.repo import InvalidUserIdentity

        from scmrepo.git import Stash

        # dulwich will silently generate an empty stash commit if there is
        # nothing to stash, we check status here to get consistent behavior
        # across backends
        if not self.is_dirty(untracked_files=include_untracked):
            return None, False

        if include_untracked or ref == Stash.DEFAULT_STASH:
            # dulwich stash.push does not support include_untracked and does
            # not touch working tree
            raise NotImplementedError

        stash = self._get_stash(ref)
        message_b = message.encode("utf-8") if message else None
        try:
            rev = stash.push(message=message_b)
        except InvalidUserIdentity as exc:
            raise SCMError("Git username and email must be configured") from exc
        return os.fsdecode(rev), True

    def _stash_apply(
        self,
        rev: str,
        reinstate_index: bool = False,
        skip_conflicts: bool = False,
        **kwargs,
    ):
        raise NotImplementedError

    def _stash_drop(self, ref: str, index: int):
        from scmrepo.git import Stash

        if ref == Stash.DEFAULT_STASH:
            raise NotImplementedError

        stash = self._get_stash(ref)
        try:
            stash.drop(index)
        except ValueError as exc:
            raise SCMError("Failed to drop stash entry") from exc

    def _describe(
        self,
        revs: Iterable[str],
        base: Optional[str] = None,
        match: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> Mapping[str, Optional[str]]:
        if not base:
            base = "refs/tags"
        rev_mapping: Dict[str, Optional[str]] = {}
        results: Dict[str, Optional[str]] = {}
        for ref in self.iter_refs(base=base):
            if (match and not fnmatch.fnmatch(ref, match)) or (
                exclude and fnmatch.fnmatch(ref, exclude)
            ):
                continue
            revision = self.get_ref(ref, follow=False)
            if revision and revision not in rev_mapping:
                rev_mapping[revision] = ref
        for rev in revs:
            results[rev] = rev_mapping.get(rev, None)
        return results

    def diff(self, rev_a: str, rev_b: str, binary=False) -> str:
        from dulwich.patch import write_tree_diff

        try:
            commit_a = self.repo[os.fsencode(rev_a)]
            commit_b = self.repo[os.fsencode(rev_b)]
        except KeyError as exc:
            raise RevError("Invalid revision") from exc

        buf = BytesIO()
        write_tree_diff(buf, self.repo.object_store, commit_a.tree, commit_b.tree)
        return buf.getvalue().decode("utf-8")

    def reset(self, hard: bool = False, paths: Iterable[str] = None):
        raise NotImplementedError

    def checkout_index(
        self,
        paths: Optional[Iterable[str]] = None,
        force: bool = False,
        ours: bool = False,
        theirs: bool = False,
    ):
        raise NotImplementedError

    def status(
        self, ignored: bool = False, untracked_files: str = "all"
    ) -> Tuple[Mapping[str, Iterable[str]], Iterable[str], Iterable[str]]:
        from dulwich.porcelain import Error
        from dulwich.porcelain import status as git_status

        with reraise(Error, SCMError("Git status failed")):
            staged, unstaged, untracked = git_status(
                self.root_dir, ignored=ignored, untracked_files=untracked_files
            )

        return (
            {
                status: [os.fsdecode(name) for name in paths]
                for status, paths in staged.items()
                if paths
            },
            [os.fsdecode(name) for name in unstaged],
            [os.fsdecode(name) for name in untracked],
        )

    def _reset(self) -> None:
        self.__dict__.pop("ignore_manager", None)

    def merge(
        self,
        rev: str,
        commit: bool = True,
        msg: Optional[str] = None,
        squash: bool = False,
    ) -> Optional[str]:
        raise NotImplementedError

    def validate_git_remote(self, url: str, **kwargs):
        from dulwich.client import LocalGitClient, get_transport_and_path
        from dulwich.porcelain import get_remote_repo

        try:
            _, location = get_remote_repo(self.repo, url)
            client, path = get_transport_and_path(location, **kwargs)
        except Exception as exc:
            raise InvalidRemote(url) from exc
        if isinstance(client, LocalGitClient) and not os.path.exists(
            os.path.join("", path)
        ):
            raise InvalidRemote(url)

    def check_ref_format(self, refname: str) -> bool:
        from dulwich.refs import check_ref_format

        return check_ref_format(refname.encode())

    def get_tag(self, name: str) -> Optional[Union[str, "GitTag"]]:
        from dulwich.objects import Tag

        name_b = os.fsencode(f"refs/tags/{name}")
        try:
            ref = self.repo.refs[name_b]
        except KeyError:
            return None
        if ref in self.repo and isinstance(self.repo[ref], Tag):
            tag = self.repo[ref]
            _typ, target_sha = tag.object
            tagger_name, tagger_email = _parse_identity(tag.tagger.decode("utf-8"))
            return GitTag(
                os.fsdecode(tag.name),
                tag.id,
                target_sha.decode("ascii"),
                tagger_name,
                tagger_email,
                tag.tag_time,
                tag.tag_timezone,
                tag.message.decode("utf-8"),
            )
        return os.fsdecode(ref)


_IDENTITY_RE = re.compile(r"(?P<name>.+)\s+<(?P<email>.+)>")


def _parse_identity(identity: str) -> Tuple[str, str]:
    m = _IDENTITY_RE.match(identity)
    if not m:
        raise SCMError("Could not parse tagger identity '{identity}'")
    return m.group("name"), m.group("email")
