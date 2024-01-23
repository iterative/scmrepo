import fnmatch
import io
import os
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Optional, Set

from scmrepo.exceptions import InvalidRemote, SCMError

from .pointer import HEADERS, Pointer

if TYPE_CHECKING:
    from scmrepo.git import Git
    from scmrepo.git.config import Config
    from scmrepo.progress import GitProgressEvent


def fetch(
    scm: "Git",
    revs: Optional[List[str]] = None,
    remote: Optional[str] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    progress: Optional[Callable[["GitProgressEvent"], None]] = None,
):
    # NOTE: This currently does not support fetching objects from the worktree
    if not revs:
        revs = ["HEAD"]
    objects: Set[Pointer] = set()
    for rev in revs:
        objects.update(
            pointer
            for pointer in _collect_objects(scm, rev, include, exclude)
            if not scm.lfs_storage.exists(pointer)
        )
    if not objects:
        return
    try:
        url = get_fetch_url(scm, remote=remote)
    except InvalidRemote:
        if remote:
            # treat remote as a raw Git remote
            url = remote
        else:
            raise
    scm.lfs_storage.fetch(url, objects, progress=progress)


def get_fetch_url(scm: "Git", remote: Optional[str] = None):  # noqa: C901,PLR0912
    """Return LFS fetch URL for the specified repository."""
    git_config = scm.get_config()

    # check lfs.url (can be set in git config and .lfsconfig)
    try:
        return git_config.get(("lfs",), "url")
    except KeyError:
        pass
    try:
        lfs_config: Optional["Config"] = scm.get_config(
            os.path.join(scm.root_dir, ".lfsconfig")
        )
    except FileNotFoundError:
        lfs_config = None
    if lfs_config:
        try:
            return lfs_config.get(("lfs",), "url")
        except KeyError:
            pass

    # use:
    #   current tracking-branch remote
    #   or remote.lfsdefault  (can only be set in git config)
    #   or "origin"
    # in that order
    if not remote:
        try:
            remote = scm.active_branch_remote()
        except SCMError:
            pass
    if not remote:
        try:
            remote = git_config.get(("remote",), "lfsdefault")
        except KeyError:
            remote = "origin"

    # check remote.*.lfsurl (can be set in git config and .lfsconfig)
    assert remote is not None
    try:
        return git_config.get(("remote", remote), "lfsurl")
    except KeyError:
        pass
    if lfs_config:
        try:
            return lfs_config.get(("remote", remote), "lfsurl")
        except KeyError:
            pass

    # return default Git fetch URL for this remote
    return scm.get_remote_url(remote)


def _collect_objects(
    scm: "Git",
    rev: str,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
) -> Iterator[Pointer]:
    fs = scm.get_fs(rev)
    for path in _filter_paths(fs.find("/"), include, exclude):
        check_path = path.lstrip("/")
        if scm.check_attr(check_path, "filter", source=rev) == "lfs":
            try:
                with fs.open(path, "rb", raw=True) as fobj:
                    with io.BufferedReader(fobj) as reader:
                        data = reader.peek(100)
                        if any(data.startswith(header) for header in HEADERS):
                            yield Pointer.load(reader)
            except (ValueError, OSError):
                pass


def _filter_paths(
    paths: Iterable[str], include: Optional[List[str]], exclude: Optional[List[str]]
) -> Iterator[str]:
    filtered = set()
    if include:
        for pattern in include:
            filtered.update(fnmatch.filter(paths, pattern))
    else:
        filtered.update(paths)
    if exclude:
        for pattern in exclude:
            filtered.difference_update(fnmatch.filter(paths, pattern))
    yield from filtered


if __name__ == "__main__":
    # Minimal `git lfs fetch` CLI implementation
    import argparse

    from scmrepo.git import Git

    parser = argparse.ArgumentParser(
        description=(
            "Download Git LFS objects at the given refs from the specified remote."
        ),
    )
    parser.add_argument(
        "remote",
        nargs="?",
        default="origin",
        help="Remote to fetch from. Defaults to 'origin'.",
    )
    parser.add_argument(
        "refs",
        nargs="*",
        default=["HEAD"],
        help="Refs or commits to fetch. Defaults to 'HEAD'.",
    )
    args = parser.parse_args()
    with Git(".") as scm_:  # pylint: disable=E0601
        fetch(scm_, revs=args.refs, remote=args.remote)
