"""Manages Git."""

import logging
import os
import re
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from functools import partial
from typing import Dict, Iterable, Optional, Tuple, Type, Union

from funcy import cached_property, first
from pathspec.patterns import GitWildMatchPattern

from scmrepo.base import Base
from scmrepo.exceptions import (
    FileNotInRepoError,
    GitHookAlreadyExists,
    RevError,
)
from scmrepo.utils import relpath

from .backend.base import BaseGitBackend, NoGitBackendError
from .backend.dulwich import DulwichBackend
from .backend.gitpython import GitPythonBackend
from .backend.pygit2 import Pygit2Backend
from .stash import Stash

logger = logging.getLogger(__name__)

BackendCls = Type[BaseGitBackend]


_LOW_PRIO_BACKENDS = ("gitpython",)


class from_backend:
    def __init__(self, *prio) -> None:
        self.prio = prio
        self.name = None
        self.func = None

    def __set_name__(self, _, name):
        self.name = name

    def __get__(self, obj, cls):
        assert self.name
        if self.func:
            return self.func
        if obj is None:
            return partial(cls._backend_func_cls, self.name, self.prio)
        return partial(obj._backend_func, self.name, self.prio)

    def __set__(self, _, func) -> None:
        self.func = func


class GitBackends(Mapping):
    DEFAULT: Dict[str, BackendCls] = {
        "dulwich": DulwichBackend,
        "pygit2": Pygit2Backend,
        "gitpython": GitPythonBackend,
    }

    def __getitem__(self, key: str) -> BaseGitBackend:
        """Lazily initialize backends and cache it afterwards"""
        initialized = self.initialized.get(key)
        if not initialized:
            backend = self.backends[key]
            initialized = backend(*self.args, **self.kwargs)
            self.initialized[key] = initialized
        return initialized

    def __init__(
        self, selected: Optional[Iterable[str]], *args, **kwargs
    ) -> None:
        selected = selected or list(self.DEFAULT)
        self.backends = OrderedDict(
            ((key, self.DEFAULT[key]) for key in selected)
        )

        self.initialized: Dict[str, BaseGitBackend] = {}

        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter(self.backends)

    def __len__(self) -> int:
        return len(self.backends)

    def close_initialized(self) -> None:
        for backend in self.initialized.values():
            backend.close()

    def reset_all(self) -> None:
        for backend in self.initialized.values():
            backend._reset()  # pylint: disable=protected-access

    def move_to_end(self, key: str, last: bool = True):
        if key not in _LOW_PRIO_BACKENDS:
            self.backends.move_to_end(key, last=last)


class Git(Base):
    """Class for managing Git."""

    GITIGNORE = ".gitignore"
    GIT_DIR = ".git"
    LOCAL_BRANCH_PREFIX = "refs/heads/"
    RE_HEXSHA = re.compile(r"^[0-9A-Fa-f]{4,40}$")
    BAD_REF_CHARS_RE = re.compile("[\177\\s~^:?*\\[]")
    default_backends = GitBackends.DEFAULT

    def __init__(
        self, *args, backends: Optional[Iterable[str]] = None, **kwargs
    ):
        self.backends = GitBackends(backends, *args, **kwargs)
        first_ = first(self.backends.values())
        super().__init__(first_.root_dir)
        self._last_backend: Optional[str] = None

    @property
    def dir(self):
        for backend in self.backends.values():
            try:
                return backend.dir
            except NotImplementedError:
                pass
        raise NotImplementedError

    @cached_property
    def hooks_dir(self):
        from pathlib import Path

        return Path(self.root_dir) / self.GIT_DIR / "hooks"

    @property
    def gitpython(self):
        return self.backends["gitpython"]

    @property
    def dulwich(self):
        return self.backends["dulwich"]

    @property
    def pygit2(self):
        return self.backends["pygit2"]

    @cached_property
    def stash(self):
        return Stash(self)

    @classmethod
    def clone(
        cls,
        url: str,
        to_path: str,
        rev: Optional[str] = None,
        **kwargs,
    ):
        cls._backend_func_cls("clone", (), url, to_path, **kwargs)
        repo = cls(to_path)
        if rev:
            repo.checkout(rev)
        return repo

    @classmethod
    def is_sha(cls, rev):
        return rev and cls.RE_HEXSHA.search(rev)

    @classmethod
    def split_ref_pattern(cls, ref: str) -> Tuple[str, str]:
        name = cls.BAD_REF_CHARS_RE.split(ref, maxsplit=1)[0]
        return name, ref[len(name) :]

    @staticmethod
    def _get_git_dir(root_dir):
        return os.path.join(root_dir, Git.GIT_DIR)

    @property
    def ignore_file(self):
        return self.GITIGNORE

    def _get_gitignore(self, path):
        ignore_file_dir = os.path.dirname(path)

        assert os.path.isabs(path)
        assert os.path.isabs(ignore_file_dir)

        entry = relpath(path, ignore_file_dir).replace(os.sep, "/")
        # NOTE: using '/' prefix to make path unambiguous
        if len(entry) > 0 and entry[0] != "/":
            entry = "/" + entry

        gitignore = os.path.join(ignore_file_dir, self.GITIGNORE)

        if not os.path.realpath(gitignore).startswith(self.root_dir + os.sep):
            raise FileNotInRepoError(
                f"'{path}' is outside of git repository '{self.root_dir}'"
            )

        return entry, gitignore

    def ignore(self, path: str) -> Optional[str]:
        entry, gitignore = self._get_gitignore(path)
        if self.is_ignored(path):
            return None

        self._add_entry_to_gitignore(entry, gitignore)
        return gitignore

    def _add_entry_to_gitignore(self, entry, gitignore):
        entry = GitWildMatchPattern.escape(entry)

        with open(gitignore, "a+", encoding="utf-8") as fobj:
            unique_lines = set(fobj.readlines())
            fobj.seek(0, os.SEEK_END)
            if fobj.tell() == 0:
                # Empty file
                prefix = ""
            else:
                fobj.seek(fobj.tell() - 1, os.SEEK_SET)
                last = fobj.read(1)
                prefix = "" if last == "\n" else "\n"
            new_entry = f"{prefix}{entry}\n"
            if new_entry not in unique_lines:
                fobj.write(new_entry)

    def ignore_remove(self, path: str) -> Optional[str]:
        entry, gitignore = self._get_gitignore(path)

        if not os.path.exists(gitignore):
            return None

        with open(gitignore, encoding="utf-8") as fobj:
            lines = fobj.readlines()

        filtered = list(filter(lambda x: x.strip() != entry.strip(), lines))

        if not filtered:
            os.unlink(gitignore)
            return None

        with open(gitignore, "w", encoding="utf-8") as fobj:
            fobj.writelines(filtered)
        return gitignore

    def verify_hook(self, name):
        if (self.hooks_dir / name).exists():
            raise GitHookAlreadyExists(name)

    def install_hook(
        self, name: str, script: str, interpreter: str = "sh"
    ) -> None:
        import shutil

        self.hooks_dir.mkdir(exist_ok=True)
        hook = self.hooks_dir / name

        directive = f"#!{shutil.which(interpreter) or '/bin/sh' }"
        hook.write_text(f"{directive}\n{script}\n", encoding="utf-8")
        hook.chmod(0o777)

    def install_merge_driver(
        self, name: str, description: str, driver: str
    ) -> None:
        self.gitpython.repo.git.config(f"merge.{name}.name", description)
        self.gitpython.repo.git.config(f"merge.{name}.driver", driver)

    def belongs_to_scm(self, path):
        basename = os.path.basename(path)
        path_parts = os.path.normpath(path).split(os.path.sep)
        return basename == self.ignore_file or Git.GIT_DIR in path_parts

    def has_rev(self, rev):
        try:
            self.resolve_rev(rev)
            return True
        except RevError:
            return False

    def close(self):
        self.backends.close_initialized()

    @property
    def no_commits(self):
        return not bool(self.get_ref("HEAD"))

    # Prefer re-using the most recently used backend when possible. When
    # changing backends (due to unimplemented calls), we close the previous
    # backend to release any open git files/contexts that may cause conflicts
    # with the new backend.
    #
    # See:
    # https://github.com/iterative/dvc/issues/5641
    # https://github.com/iterative/dvc/issues/7458
    def _backend_func(self, name, prio, *args, **kwargs):
        backends = prio or self.backends.keys()
        for key in backends:
            if self._last_backend is not None and key != self._last_backend:
                self.backends[self._last_backend].close()
                self._last_backend = None
            try:
                backend = self.backends[key]
                func = getattr(backend, name)
                result = func(*args, **kwargs)
                self._last_backend = key
                self.backends.move_to_end(key, last=False)
                return result
            except NotImplementedError:
                pass
        raise NoGitBackendError(name)

    @classmethod
    def _backend_func_cls(cls, name, prio, *args, **kwargs):
        backends = prio or cls.default_backends.keys()
        for key in backends:
            try:
                backend = cls.default_backends[key]
                func = getattr(backend, name)
                result = func(*args, **kwargs)
                return result
            except NotImplementedError:
                pass
        raise NoGitBackendError(name)

    def get_fs(self, rev: str):
        from scmrepo.fs import GitFileSystem

        return GitFileSystem(scm=self, rev=rev)

    @classmethod
    def init(
        cls, path: str, bare: bool = False, _backend: str = None
    ) -> "Git":
        backends = (_backend,) if _backend else ()
        cls._backend_func_cls("init", backends, path, bare=bare)
        return cls(path)

    def add_commit(
        self,
        paths: Union[str, Iterable[str]],
        message: str,
    ) -> None:
        self.add(paths)
        self.commit(msg=message)

    is_ignored = from_backend()
    add = from_backend()
    commit = from_backend()
    checkout = from_backend()
    fetch = from_backend()
    pull = from_backend()
    push = from_backend()
    branch = from_backend()
    tag = from_backend()
    untracked_files = from_backend()
    is_tracked = from_backend()
    is_dirty = from_backend()
    active_branch = from_backend()
    list_branches = from_backend()
    list_tags = from_backend()
    list_all_commits = from_backend()
    get_rev = from_backend()
    resolve_rev = from_backend()
    resolve_commit = from_backend()
    set_ref = from_backend()
    get_ref = from_backend()
    remove_ref = from_backend()
    iter_refs = from_backend()
    iter_remote_refs = from_backend()
    get_refs_containing = from_backend()
    push_refspecs = from_backend()
    fetch_refspecs = from_backend()
    _stash_iter = from_backend()
    _stash_push = from_backend()
    _stash_apply = from_backend()
    _stash_drop = from_backend()
    _describe = from_backend()
    diff = from_backend()
    reset = from_backend()
    checkout_index = from_backend()
    status = from_backend()
    merge = from_backend()
    validate_git_remote = from_backend()
    check_ref_format = from_backend()
    get_tree_obj = from_backend()

    def branch_revs(
        self, branch: str, end_rev: Optional[str] = None
    ) -> Iterable[str]:
        """Iterate over revisions in a given branch (from newest to oldest).

        If end_rev is set, iterator will stop when the specified revision is
        reached.
        """
        commit = self.resolve_commit(branch)
        while commit is not None:
            yield commit.hexsha
            parent = first(commit.parents)
            if parent is None or parent == end_rev:
                return
            commit = self.resolve_commit(parent)

    @contextmanager
    def detach_head(
        self,
        rev: Optional[str] = None,
        force: bool = False,
        client: str = "scm",
    ):
        """Context manager for performing detached HEAD SCM operations.

        Detaches and restores HEAD similar to interactive git rebase.
        Restore is equivalent to 'reset --soft', meaning the caller is
        is responsible for preserving & restoring working tree state
        (i.e. via stash) when applicable.

        Yields revision of detached head.
        """
        if not rev:
            rev = "HEAD"
        orig_head = self.get_ref("HEAD", follow=False)
        logger.debug("Detaching HEAD at '%s'", rev)
        self.checkout(rev, detach=True, force=force)
        try:
            yield self.get_ref("HEAD")
        finally:
            prefix = self.LOCAL_BRANCH_PREFIX
            if orig_head.startswith(prefix):
                symbolic = True
                name = orig_head[len(prefix) :]
            else:
                symbolic = False
                name = orig_head
            self.set_ref(
                "HEAD",
                orig_head,
                symbolic=symbolic,
                message=f"{client}: Restore HEAD to '{name}'",
            )
            logger.debug("Restore HEAD to '%s'", name)
            self.reset()

    @contextmanager
    def stash_workspace(self, **kwargs):
        """Stash and restore any workspace changes.

        Yields revision of the stash commit. Yields None if there were no
        changes to stash.
        """
        logger.debug("Stashing workspace")
        rev = self.stash.push(**kwargs)
        try:
            yield rev
        finally:
            if rev:
                logger.debug("Restoring stashed workspace")
                self.stash.pop()

    def _reset(self) -> None:
        self.backends.reset_all()

    def describe(
        self,
        rev: str,
        base: Optional[str] = None,
        match: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> Optional[str]:
        if (
            base == "refs/heads"
            and self.get_rev() == rev
            and self.get_ref("HEAD", follow=False).startswith(base)
        ):
            return self.get_ref("HEAD", follow=False)
        return self._describe(rev, base, match, exclude)
