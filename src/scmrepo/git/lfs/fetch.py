import fnmatch
import io
from typing import TYPE_CHECKING, Iterable, Iterator, List, Optional

from scmrepo.exceptions import InvalidRemote

from .pointer import HEADERS, Pointer

if TYPE_CHECKING:
    from scmrepo.git import Git


def fetch(
    scm: "Git",
    revs: Optional[List[str]] = None,
    remote: Optional[str] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
):
    # NOTE: This currently does not support fetching objects from the worktree
    if not revs:
        revs = ["HEAD"]
    if not remote:
        remote = "origin"
    objects = set()
    for rev in revs:
        objects.update(
            pointer
            for pointer in _collect_objects(scm, rev, include, exclude)
            if not scm.lfs_storage.exists(pointer)
        )
    try:
        url = scm.get_remote_url(remote)
    except InvalidRemote:
        # treat remote as a raw git URL
        url = remote
    scm.lfs_storage.fetch(url, objects)


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
    import sys

    from scmrepo.git import Git  # noqa: F811

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
    with Git(".") as scm:
        print("fetch: fetching reference", ", ".join(args.refs), file=sys.stderr)
        fetch(scm, revs=args.refs, remote=args.remote)
