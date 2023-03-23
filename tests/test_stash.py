import sys
from typing import Optional

import pytest
from pytest_test_utils import TmpDir

from scmrepo.git import Git, Stash


def test_git_stash_workspace(tmp_dir: TmpDir, scm: Git):
    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    tmp_dir.gen("file", "1")

    with scm.stash_workspace():
        assert not scm.is_dirty()
        assert (tmp_dir / "file").read_text() == "0"
    assert scm.is_dirty()
    assert (tmp_dir / "file").read_text() == "1"


def test_git_stash_workspace_reinstate_index(tmp_dir: TmpDir, scm: Git):
    tmp_dir.gen({"modified": "init", "deleted": "deleted"})
    scm.add_commit(["modified", "deleted"], "init")

    tmp_dir.gen({"newfile": "nefile"})
    scm.add("newfile")
    tmp_dir.gen({"modified": "modified"})
    scm.add("modified")
    (tmp_dir / "deleted").unlink()
    scm.add("deleted")

    prev_status = scm.status()
    with scm.stash_workspace(reinstate_index=True):
        pass
    assert scm.status() == prev_status


@pytest.mark.parametrize(
    "ref, include_untracked",
    [
        (None, True),
        (None, False),
        ("refs/foo/stash", True),
        ("refs/foo/stash", False),
    ],
)
def test_git_stash_push(
    tmp_dir: TmpDir, scm: Git, ref: Optional[str], include_untracked: bool
):
    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    tmp_dir.gen({"file": "1", "untracked": "0"})

    stash = Stash(scm, ref=ref)
    rev = stash.push(include_untracked=include_untracked)
    assert rev == scm.get_ref(stash.ref)
    assert (tmp_dir / "file").read_text() == "0"
    assert include_untracked != (tmp_dir / "untracked").exists()
    assert len(stash) == 1

    stash.apply(rev)
    assert (tmp_dir / "file").read_text() == "1"
    assert (tmp_dir / "untracked").read_text() == "0"

    parts = list(stash.ref.split("/"))
    assert (tmp_dir / ".git").joinpath(*parts).exists()
    assert (tmp_dir / ".git" / "logs").joinpath(*parts).exists()


@pytest.mark.parametrize("ref", [None, "refs/foo/stash"])
def test_git_stash_drop(tmp_dir: TmpDir, scm: Git, ref: Optional[str]):
    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    tmp_dir.gen("file", "1")

    stash = Stash(scm, ref=ref)
    stash.push()

    tmp_dir.gen("file", "2")
    expected = stash.push()

    stash.drop(1)
    assert expected == scm.get_ref(stash.ref)
    assert len(stash) == 1


reason = """libgit2 stash_save() is flaky on linux when run inside pytest
    https://github.com/iterative/dvc/pull/5286#issuecomment-792574294"""


@pytest.mark.xfail(
    sys.platform in ("linux", "win32"),
    raises=AssertionError,
    strict=False,
    reason=reason,
)
@pytest.mark.parametrize("ref", [None, "refs/foo/stash"])
def test_git_stash_pop(tmp_dir: TmpDir, scm: Git, ref: Optional[str]):
    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    tmp_dir.gen("file", "1")

    stash = Stash(scm, ref=ref)
    first = stash.push()

    tmp_dir.gen("file", "2")
    second = stash.push()

    assert second == stash.pop()
    assert len(stash) == 1
    assert first == scm.get_ref(stash.ref)
    assert (tmp_dir / "file").read_text() == "2"


@pytest.mark.parametrize("ref", [None, "refs/foo/stash"])
def test_git_stash_clear(tmp_dir: TmpDir, scm: Git, ref: Optional[str]):
    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    tmp_dir.gen("file", "1")

    stash = Stash(scm, ref=ref)
    stash.push()

    tmp_dir.gen("file", "2")
    stash.push()

    stash.clear()
    assert len(stash) == 0

    parts = list(stash.ref.split("/"))
    assert not (tmp_dir / ".git").joinpath(*parts).exists()

    reflog_file = (tmp_dir / ".git" / "logs").joinpath(*parts)
    # NOTE: some backends will completely remove reflog file on clear, some
    # will only truncate it, either case means an empty stash
    assert not reflog_file.exists() or not reflog_file.cat()


@pytest.mark.skip_git_backend("dulwich")
def test_git_stash_apply_index(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
):
    tmp_dir.gen("file", "0")
    scm.add_commit("file", message="init")
    tmp_dir.gen("file", "1")
    scm.add("file")
    scm.stash.push()
    rev = scm.resolve_rev(r"stash@{0}")

    stash = Stash(git)
    stash.apply(rev, reinstate_index=True)

    assert (tmp_dir / "file").read_text() == "1"
    staged, unstaged, untracked = scm.status()
    assert dict(staged) == {"modify": ["file"]}
    assert not dict(unstaged)
    assert not dict(untracked)


def test_git_stash_push_clean_workspace(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
):
    tmp_dir.gen("file", "0")
    scm.add_commit("file", message="init")
    assert git._stash_push("refs/stash") == (  # pylint: disable=protected-access
        None,
        False,
    )
