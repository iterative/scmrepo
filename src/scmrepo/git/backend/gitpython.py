import io
import locale
import logging
import os
import sys
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from funcy import ignore

from scmrepo.exceptions import (
    CloneError,
    MergeConflictError,
    RevError,
    SCMError,
    UnsupportedIndexFormat,
)
from scmrepo.utils import relpath

from ..objects import GitCommit, GitObject, GitTag
from .base import BaseGitBackend, SyncStatus

if TYPE_CHECKING:
    from scmrepo.progress import GitProgressEvent


logger = logging.getLogger(__name__)


# NOTE: Check if we are in a bundle
# https://pythonhosted.org/PyInstaller/runtime-information.html
def is_binary() -> bool:
    return getattr(sys, "frozen", False)


def fix_env(env: Dict[str, str] = None) -> Dict[str, str]:
    if env is None:
        environ = os.environ.copy()
    else:
        environ = env.copy()

    if is_binary():
        lp_key = "LD_LIBRARY_PATH"
        lp_orig = environ.get(lp_key + "_ORIG", None)
        if lp_orig is not None:
            environ[lp_key] = lp_orig
        else:
            environ.pop(lp_key, None)
    return environ


class GitPythonObject(GitObject):
    def __init__(self, obj):
        self.obj = obj

    def open(self, mode: str = "r", encoding: str = None):
        if not encoding:
            encoding = locale.getpreferredencoding(False)
        # GitPython's obj.data_stream is a fragile thing, it is better to
        # read it immediately, also it needs to be to decoded if we follow
        # the `open()` behavior (since data_stream.read() returns bytes,
        # and `open` with default "r" mode returns str)
        data = self.obj.data_stream.read()
        if mode == "rb":
            return io.BytesIO(data)
        return io.StringIO(data.decode(encoding))

    @property
    def name(self) -> str:
        # NOTE: `obj.name` is not always a basename. See [1] for more details.
        #
        # [1] https://github.com/iterative/dvc/issues/3481
        return os.path.basename(self.obj.path)

    @property
    def mode(self) -> int:
        return self.obj.mode

    def scandir(self) -> Iterable["GitPythonObject"]:
        for obj in self.obj:
            yield GitPythonObject(obj)

    @property
    def size(self) -> int:
        return self.obj.size

    @property
    def sha(self) -> str:
        return self.obj.hexsha


class GitPythonBackend(BaseGitBackend):  # pylint:disable=abstract-method
    """git-python Git backend."""

    def __init__(  # pylint:disable=W0231
        self, root_dir=os.curdir, search_parent_directories=True
    ):
        import git
        from git.exc import InvalidGitRepositoryError

        try:
            self.repo = git.Repo(
                root_dir, search_parent_directories=search_parent_directories
            )
        except InvalidGitRepositoryError:
            msg = "{} is not a git repository"
            raise SCMError(msg.format(root_dir))

        # NOTE: fixing LD_LIBRARY_PATH for binary built by PyInstaller.
        # http://pyinstaller.readthedocs.io/en/stable/runtime-information.html
        env = fix_env()
        libpath = env.get("LD_LIBRARY_PATH", None)
        self.repo.git.update_environment(LD_LIBRARY_PATH=libpath)

    def close(self):
        self.repo.close()

    @property
    def git(self):
        return self.repo.git

    def is_ignored(self, path: "Union[str, os.PathLike[str]]") -> bool:
        from git.exc import GitCommandError

        func = ignore(GitCommandError)(self.repo.git.check_ignore)
        return bool(func(str(path)))

    @property
    def root_dir(self) -> str:
        return self.repo.working_tree_dir

    @staticmethod
    def clone(
        url: str,
        to_path: str,
        shallow_branch: Optional[str] = None,
        progress: Callable[["GitProgressEvent"], None] = None,
        bare: bool = False,
        mirror: bool = False,
    ):
        from git import Repo
        from git.exc import GitCommandError

        ld_key = "LD_LIBRARY_PATH"

        env = fix_env()
        if is_binary() and ld_key not in env.keys():
            # In fix_env, we delete LD_LIBRARY_PATH key if it was empty before
            # PyInstaller modified it. GitPython, in git.Repo.clone_from, uses
            # env to update its own internal state. When there is no key in
            # env, this value is not updated and GitPython re-uses
            # LD_LIBRARY_PATH that has been set by PyInstaller.
            # See [1] for more info.
            # [1] https://github.com/gitpython-developers/GitPython/issues/924
            env[ld_key] = ""

        try:
            if shallow_branch is not None and os.path.exists(url):
                # git disables --depth for local clones unless file:// url
                # scheme is used
                url = f"file://{url}"

            from scmrepo.progress import GitProgressReporter

            clone_from = partial(
                Repo.clone_from,
                url,
                to_path,
                env=env,  # needed before we can fix it in __init__
                no_single_branch=True,
                progress=GitProgressReporter.wrap_fn(progress) if progress else None,
                bare=bare,
                mirror=mirror,
            )
            if shallow_branch is None:
                tmp_repo = clone_from()
            else:
                tmp_repo = clone_from(branch=shallow_branch, depth=1)
            tmp_repo.close()
        except GitCommandError as exc:  # pylint: disable=no-member
            raise CloneError(url, to_path) from exc

    @staticmethod
    def init(path: str, bare: bool = False) -> None:
        from funcy import retry
        from git import Repo
        from git.exc import GitCommandNotFound

        # NOTE: handles EAGAIN error on BSD systems (osx in our case).
        # Otherwise when running tests you might get this exception:
        #
        #    GitCommandNotFound: Cmd('git') not found due to:
        #        OSError('[Errno 35] Resource temporarily unavailable')
        method = retry(5, GitCommandNotFound)(Repo.init)
        git = method(path, bare=bare)
        git.close()

    @staticmethod
    def is_sha(rev):
        import git

        return rev and git.Repo.re_hexsha_shortened.search(rev)

    @property
    def dir(self) -> str:
        return self.repo.git_dir

    def add(
        self,
        paths: Union[str, Iterable[str]],
        update: bool = False,
        force: bool = False,
    ):
        try:
            if update or not force:
                # NOTE: git-python index.add() defines force parameter but
                # ignores it (index.add() behavior is always force=True)
                kwargs: Dict[str, Any] = {}
                if update:
                    kwargs["update"] = True
                if isinstance(paths, str):
                    paths = [paths]
                if not force:
                    paths = [path for path in paths if not self.is_ignored(path)]
                self.git.add(*paths, **kwargs)
            else:
                self.repo.index.add(paths)
        except AssertionError as exc:
            # NOTE: GitPython is not currently able to handle index version >= 3.
            # See https://github.com/iterative/dvc/issues/610 for more details.
            raise UnsupportedIndexFormat from exc

    def commit(self, msg: str, no_verify: bool = False):
        from git.exc import HookExecutionError

        try:
            self.repo.index.commit(msg, skip_hooks=no_verify)
        except HookExecutionError as exc:
            raise SCMError("Git pre-commit hook failed") from exc

    def checkout(
        self,
        branch: str,
        create_new: Optional[bool] = False,
        force: bool = False,
        **kwargs,
    ):
        if create_new:
            self.repo.git.checkout("HEAD", b=branch, force=force, **kwargs)
        else:
            self.repo.git.checkout(branch, force=force, **kwargs)

    def fetch(
        self,
        remote: Optional[str] = None,
        force: bool = False,
        unshallow: bool = False,
    ):
        if not remote:
            remote = "origin"
        kwargs = {}
        if force:
            kwargs["force"] = True
        if unshallow:
            kwargs["unshallow"] = True
        infos = self.repo.remote(name=remote).fetch(**kwargs)
        for info in infos:
            if info.flags & info.ERROR:
                raise SCMError(f"fetch failed: {info.note}")

    def pull(self, **kwargs):
        infos = self.repo.remote().pull(**kwargs)
        for info in infos:
            if info.flags & info.ERROR:
                raise SCMError(f"pull failed: {info.note}")

    def push(self):
        infos = self.repo.remote().push()
        for info in infos:
            if info.flags & info.ERROR:
                raise SCMError(f"push failed: {info.summary}")

    def branch(self, branch):
        self.repo.git.branch(branch)

    def tag(
        self,
        tag: str,
        target: Optional[str] = None,
        annotated: bool = False,
        message: Optional[str] = None,
    ):
        if annotated and not message:
            raise SCMError("message is required for annotated tag")
        self.repo.git.tag(tag, target or "HEAD", a=annotated, m=message)

    def untracked_files(self):
        files = self.repo.untracked_files
        return [os.path.join(self.repo.working_dir, fname) for fname in files]

    def is_tracked(self, path):
        return bool(self.repo.git.ls_files(path))

    def is_dirty(self, untracked_files: bool = False) -> bool:
        return self.repo.is_dirty(untracked_files=untracked_files)

    def active_branch(self):
        try:
            return self.repo.active_branch.name
        except TypeError as exc:
            raise SCMError("No active branch") from exc

    def list_branches(self):
        return [h.name for h in self.repo.heads]

    def list_tags(self):
        return [t.name for t in self.repo.tags]

    def list_all_commits(self):
        head = self.get_ref("HEAD")
        if not head:
            # Empty repo
            return []

        return [
            c.hexsha
            for c in self.repo.iter_commits(
                rev=head, branches=True, tags=True, remotes=True
            )
        ]

    def get_tree_obj(self, rev: str, **kwargs) -> GitPythonObject:
        tree = self.repo.tree(rev)
        return GitPythonObject(tree)

    def get_rev(self):
        return self.repo.rev_parse("HEAD").hexsha

    def resolve_rev(self, rev):
        from contextlib import suppress

        from git.exc import BadName, GitCommandError

        def _resolve_rev(name):
            with suppress(BadName, GitCommandError):
                try:
                    # Try python implementation of rev-parse first, it's faster
                    return self.repo.rev_parse(name).hexsha
                except NotImplementedError:
                    # Fall back to `git rev-parse` for advanced features
                    return self.repo.git.rev_parse(name)
                except ValueError:
                    raise RevError(f"unknown Git revision '{name}'")

        # Resolve across local names
        sha = _resolve_rev(rev)
        if sha:
            return sha

        # Try all the remotes and if it resolves unambiguously then take it
        if not self.is_sha(rev):
            shas = {
                _resolve_rev(f"{remote.name}/{rev}") for remote in self.repo.remotes
            } - {None}
            if len(shas) > 1:
                raise RevError(f"ambiguous Git revision '{rev}'")
            if len(shas) == 1:
                return shas.pop()

        raise RevError(f"unknown Git revision '{rev}'")

    def resolve_commit(self, rev: str) -> "GitCommit":
        """Return Commit object for the specified revision."""
        from git.exc import BadName, GitCommandError
        from git.objects.tag import TagObject
        from gitdb.exc import BadObject

        try:
            commit = self.repo.rev_parse(rev)
        except (BadName, BadObject, GitCommandError) as exc:
            raise SCMError(f"Invalid commit '{rev}'") from exc
        if isinstance(commit, TagObject):
            commit = commit.object
        return GitCommit(
            commit.hexsha,
            commit.committed_date,
            commit.committer_tz_offset,
            commit.message,
            [str(parent) for parent in commit.parents],
            commit.committer.name,
            commit.committer.email,
            commit.author.name,
            commit.author.email,
            commit.authored_date,
            commit.author_tz_offset,
        )

    def set_ref(
        self,
        name: str,
        new_ref: str,
        old_ref: Optional[str] = None,
        message: Optional[str] = None,
        symbolic: Optional[bool] = False,
    ):
        from git.exc import GitCommandError

        if old_ref and self.get_ref(name) != old_ref:
            raise SCMError(f"Failed to set ref '{name}'")
        try:
            if symbolic:
                if message:
                    self.git.symbolic_ref(name, new_ref, m=message)
                else:
                    self.git.symbolic_ref(name, new_ref)
            else:
                args = [name, new_ref]
                if old_ref:
                    args.append(old_ref)
                if message:
                    self.git.update_ref(*args, m=message, create_reflog=True)
                else:
                    self.git.update_ref(*args)
        except GitCommandError as exc:
            raise SCMError(f"Failed to set ref '{name}'") from exc

    def get_ref(self, name: str, follow: bool = True) -> Optional[str]:
        from git.exc import GitCommandError

        if name == "HEAD":
            try:
                if follow or self.repo.head.is_detached:
                    return self.repo.head.commit.hexsha
                return f"refs/heads/{self.repo.active_branch}"
            except (GitCommandError, ValueError):
                return None
        elif name.startswith("refs/heads/"):
            name = name[11:]
            if name in self.repo.heads:
                return self.repo.heads[name].commit.hexsha
        elif name.startswith("refs/tags/"):
            name = name[10:]
            if name in self.repo.tags:
                return self.repo.tags[name].commit.hexsha
        else:
            if not follow:
                try:
                    rev = self.git.symbolic_ref(name).strip()
                    return rev if rev else None
                except GitCommandError:
                    pass
            try:
                rev = self.git.show_ref(name, hash=True).strip()
                return rev if rev else None
            except GitCommandError:
                pass
        return None

    def remove_ref(self, name: str, old_ref: Optional[str] = None):
        from git.exc import GitCommandError

        if old_ref and self.get_ref(name) != old_ref:
            raise SCMError(f"Failed to set ref '{name}'")
        try:
            args = [name]
            if old_ref:
                args.append(old_ref)
            self.git.update_ref(*args, d=True)
        except GitCommandError as exc:
            raise SCMError(f"Failed to set ref '{name}'") from exc

    def iter_refs(self, base: Optional[str] = None):
        from git import Reference

        for ref in Reference.iter_items(self.repo, common_path=base):
            yield ref.path

    def iter_remote_refs(self, url: str, base: Optional[str] = None, **kwargs):
        raise NotImplementedError

    def get_refs_containing(self, rev: str, pattern: Optional[str] = None):
        from git.exc import GitCommandError

        try:
            if pattern:
                args = [pattern]
            else:
                args = []
            for line in self.git.for_each_ref(
                *args, contains=rev, format=r"%(refname)"
            ).splitlines():
                line = line.strip()
                if line:
                    yield line
        except GitCommandError:
            pass

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

    def fetch_refspecs(
        self,
        url: str,
        refspecs: Union[str, Iterable[str]],
        force: bool = False,
        on_diverged: Optional[Callable[[str, str], bool]] = None,
        progress: Callable[["GitProgressEvent"], None] = None,
        **kwargs,
    ) -> Mapping[str, SyncStatus]:
        raise NotImplementedError

    def _stash_iter(self, ref: str):
        raise NotImplementedError

    def _stash_push(
        self,
        ref: str,
        message: Optional[str] = None,
        include_untracked: bool = False,
    ) -> Tuple[Optional[str], bool]:
        from scmrepo.git import Stash

        if not self.is_dirty(untracked_files=include_untracked):
            return None, False

        args = ["push"]
        if message:
            args.extend(["-m", message])
        if include_untracked:
            args.append("--include-untracked")
        self.git.stash(*args)
        commit = self.resolve_commit("stash@{0}")
        if ref != Stash.DEFAULT_STASH:
            # `git stash` CLI doesn't support using custom refspecs,
            # so we push a commit onto refs/stash, make our refspec
            # point to the new commit, then pop it from refs/stash
            # `git stash create` is intended to be used for this kind of
            # behavior but it doesn't support --include-untracked so we need to
            # use push
            self.set_ref(ref, commit.hexsha, message=commit.message)
            self.git.stash("drop")
        return commit.hexsha, False

    def _stash_apply(
        self,
        rev: str,
        reinstate_index: bool = False,
        skip_conflicts: bool = False,
        **kwargs,
    ):
        from git.exc import GitCommandError

        if skip_conflicts:
            raise NotImplementedError
        try:
            args = ["apply"]
            if reinstate_index:
                args.append("--index")
            args.append(rev)
            self.git.stash(args)
        except GitCommandError as exc:
            out = str(exc)
            if "CONFLICT" in out or "already exists" in out:
                raise MergeConflictError(
                    "Stash apply resulted in merge conflicts"
                ) from exc
            raise SCMError("Could not apply stash") from exc

    def _stash_drop(self, ref: str, index: int):
        from git.exc import GitCommandError

        from scmrepo.git import Stash

        if ref == Stash.DEFAULT_STASH:
            self.git.stash("drop", index)
            return

        self.git.reflog("delete", "--updateref", "--rewrite", f"{ref}@{{{index}}}")
        try:
            self.git.reflog("exists", ref)
        except GitCommandError:
            self.remove_ref(ref)

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
        if paths:
            paths_list: Optional[List[str]] = [
                relpath(path, self.root_dir) for path in paths
            ]
            if os.name == "nt":
                paths_list = [
                    path.replace("\\", "/")
                    for path in paths_list  # type: ignore[union-attr]
                ]
        else:
            paths_list = None
        self.repo.head.reset(index=True, working_tree=hard, paths=paths_list)

    def checkout_index(
        self,
        paths: Optional[Iterable[str]] = None,
        force: bool = False,
        ours: bool = False,
        theirs: bool = False,
    ):
        """Checkout the specified paths from HEAD index."""
        assert not (ours and theirs)
        if ours or theirs:
            args = ["--ours"] if ours else ["--theirs"]
            if force:
                args.append("--force")
            args.append("--")
            if paths:
                args.extend(list(paths))
            else:
                args.append(".")
            self.repo.git.checkout(*args)
        else:
            if paths:
                paths_list: Optional[List[str]] = [
                    relpath(path, self.root_dir) for path in paths
                ]
                if os.name == "nt":
                    paths_list = [
                        path.replace("\\", "/")
                        for path in paths_list  # type: ignore[union-attr]
                    ]
            else:
                paths_list = None
            self.repo.index.checkout(paths=paths_list, force=force)

    def status(
        self, ignored: bool = False, untracked_files: str = "all"
    ) -> Tuple[Mapping[str, Iterable[str]], Iterable[str], Iterable[str]]:
        raise NotImplementedError

    def merge(
        self,
        rev: str,
        commit: bool = True,
        msg: Optional[str] = None,
        squash: bool = False,
    ) -> Optional[str]:
        from git.exc import GitCommandError

        if commit and squash:
            raise SCMError("Cannot merge with 'squash' and 'commit'")

        if commit and not msg:
            raise SCMError("Merge commit message is required")

        merge = partial(self.git.merge, rev)
        try:
            if commit:
                merge(m=msg)
                return self.get_rev()
            merge(no_commit=True, squash=True)
        except GitCommandError as exc:
            if "CONFLICT" in str(exc):
                raise MergeConflictError("Merge contained conflicts") from exc
            raise SCMError("Merge failed") from exc
        return None

    def validate_git_remote(self, url: str, **kwargs):
        raise NotImplementedError

    def check_ref_format(self, refname: str):
        raise NotImplementedError

    def get_tag(self, name: str) -> Optional[Union[str, "GitTag"]]:
        try:
            ref = self.repo.tags[name]
            if not ref.tag:
                return ref.commit.hexsha
            tag = ref.tag
            return GitTag(
                tag.tag,
                tag.hexsha,
                tag.object.hexsha,
                tag.tagger.name,
                tag.tagger.email,
                tag.tagged_date,
                tag.tagger_tz_offset,
                tag.message,
            )
        except IndexError:
            pass
        return None
