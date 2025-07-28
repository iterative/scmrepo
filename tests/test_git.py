import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import pytest
from asyncssh import SFTPClient
from asyncssh.connection import SSHClientConnection
from dulwich.client import LocalGitClient
from git import Repo as GitPythonRepo
from proxy import TestCase as ProxyTestCase
from pygit2 import GitError
from pygit2.remotes import Remote
from pytest_mock import MockerFixture
from pytest_test_utils import TempDirFactory, TmpDir
from pytest_test_utils.matchers import Matcher

from scmrepo.exceptions import (
    InvalidRemote,
    MergeConflictError,
    RevError,
    SCMError,
)
from scmrepo.git import Git, GitBackends
from scmrepo.git.objects import GitTag

from .conftest import backends

# pylint: disable=redefined-outer-name,unused-argument,protected-access


BAD_PROXY_CONFIG = """[http]
proxy = "http://bad-proxy.dvc.org"
[https]
proxy = "http://bad-proxy.dvc.org"
"""


@pytest.fixture
def submodule_dir(tmp_dir: TmpDir, scm: Git):
    scm.commit("init")

    subrepo = GitPythonRepo.init()
    subrepo_path = "subrepo"
    subrepo.create_submodule(subrepo_path, subrepo_path, subrepo.git_dir)
    subrepo.close()

    return tmp_dir / subrepo_path


def test_git_init(tmp_dir: TmpDir, git_backend: str):
    Git.init(".", _backend=git_backend)
    assert (tmp_dir / ".git").is_dir()

    for backend in backends:
        Git(tmp_dir, backends=[backend])


def test_git_init_bare(tmp_dir: TmpDir, git_backend: str):
    Git.init(".", bare=True, _backend=git_backend)
    assert list(tmp_dir.iterdir())

    for backend in backends:
        Git(tmp_dir, backends=[backend])


def test_git_submodule(submodule_dir: TmpDir, git_backend: str):
    git = Git(backends=[git_backend])
    git.close()

    git = Git(submodule_dir, backends=[git_backend])
    git.close()


@pytest.mark.skip_git_backend("pygit2")
def test_commit(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"foo": "foo"})
    git.add(["foo"])
    git.commit("add")
    assert "foo" in scm.gitpython.git.ls_files()


@pytest.mark.skip_git_backend("pygit2")
def test_commit_in_root_repo_with_submodule(
    tmp_dir: TmpDir, scm: Git, git: Git, submodule_dir: TmpDir
):
    tmp_dir.gen("foo", "foo")
    git.add(["foo"])
    git.commit("add foo")
    assert "foo" in scm.gitpython.git.ls_files()


@pytest.mark.parametrize(
    "path, expected",
    [
        (os.path.join("path", "to", ".gitignore"), True),
        (os.path.join("path", "to", ".git", "internal", "file"), True),
        (os.path.join("some", "non-.git", "file"), False),
    ],
    ids=["gitignore_file", "git_internal_file", "non_git_file"],
)
def test_belongs_to_scm(scm: Git, git: Git, path: str, expected: str):
    assert git.belongs_to_scm(path) == expected


@pytest.mark.skip_git_backend("pygit2")
def test_is_tracked(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen(
        {
            "tracked": "tracked",
            "dir": {"data": "data", "subdir": {"subdata": "subdata"}},
        },
    )
    scm.add_commit(["tracked", "dir"], message="add dirs and files")
    tmp_dir.gen({"untracked": "untracked", "dir": {"untracked": "untracked"}})

    # sanity check
    assert (tmp_dir / "untracked").exists()
    assert (tmp_dir / "tracked").exists()
    assert (tmp_dir / "dir" / "untracked").exists()
    assert (tmp_dir / "dir" / "data").exists()
    assert (tmp_dir / "dir" / "subdir" / "subdata").exists()

    assert not git.is_tracked("untracked")
    assert not git.is_tracked(os.path.join("dir", "untracked"))

    assert git.is_tracked("tracked")
    assert git.is_tracked("dir")
    assert git.is_tracked(os.path.join("dir", "data"))
    assert git.is_tracked(os.path.join("dir", "subdir"))
    assert git.is_tracked(os.path.join("dir", "subdir", "subdata"))


@pytest.mark.skip_git_backend("pygit2")
def test_is_tracked_unicode(tmp_dir: TmpDir, scm: Git, git: Git):
    files = tmp_dir.gen("ṭṝḁḉḵḗḋ", "tracked")
    scm.add_commit(files, message="add unicode")
    tmp_dir.gen("ṳṋṭṝḁḉḵḗḋ", "untracked")

    assert git.is_tracked("ṭṝḁḉḵḗḋ")
    assert not git.is_tracked("ṳṋṭṝḁḉḵḗḋ")


@pytest.mark.skip_git_backend("pygit2")
def test_is_tracked_func(tmp_dir, scm, git):
    tmp_dir.gen({"foo": "foo", "тест": "проверка"})
    scm.add(["foo", "тест"])

    abs_foo = os.path.abspath("foo")
    assert git.is_tracked(abs_foo)
    assert git.is_tracked("foo")
    assert git.is_tracked("тест")

    git.commit("add")
    assert git.is_tracked(abs_foo)
    assert git.is_tracked("foo")

    scm.gitpython.repo.index.remove(["foo"], working_tree=True)
    assert not git.is_tracked(abs_foo)
    assert not git.is_tracked("foo")
    assert not git.is_tracked("not-existing-file")


@pytest.mark.skip_git_backend("pygit2")
def test_no_commits(tmp_dir: TmpDir, scm: Git, git: Git):
    assert git.no_commits

    tmp_dir.gen("foo", "foo")
    scm.add_commit(["foo"], message="foo")

    assert not git.no_commits


@pytest.mark.skip_git_backend("dulwich")
def test_branch_revs(tmp_dir: TmpDir, scm: Git, git: Git):
    def _gen(i: int):
        tmp_dir.gen({"file": f"{i}"})
        scm.add_commit("file", message=f"{i}")
        return scm.get_rev()

    base, *others = (_gen(i) for i in range(5))
    branch_revs = list(git.branch_revs("master", base))[::-1]
    assert branch_revs == others


def test_set_ref(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen("file", "0")
    scm.add_commit("file", message="init")
    init_rev = scm.get_rev()

    tmp_dir.gen({"file": "1"})
    scm.add_commit("file", message="commit")
    commit_rev = scm.get_rev()

    git.set_ref("refs/foo/bar", init_rev)
    assert init_rev == (tmp_dir / ".git" / "refs" / "foo" / "bar").read_text().strip()

    with pytest.raises(SCMError):
        git.set_ref("refs/foo/bar", commit_rev, old_ref=commit_rev)
    git.set_ref("refs/foo/bar", commit_rev, old_ref=init_rev)
    assert commit_rev == (tmp_dir / ".git" / "refs" / "foo" / "bar").read_text().strip()

    git.set_ref("refs/foo/baz", "refs/heads/master", symbolic=True)
    assert (
        tmp_dir / ".git" / "refs" / "foo" / "baz"
    ).read_text().strip() == "ref: refs/heads/master"


def test_get_ref(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    init_rev = scm.get_rev()
    tmp_dir.gen(
        {
            os.path.join(".git", "refs", "foo", "bar"): init_rev,
            os.path.join(".git", "refs", "foo", "baz"): "ref: refs/heads/master",
        }
    )
    scm.tag("annotated", annotated=True, message="Annotated Tag")

    assert init_rev == git.get_ref("refs/foo/bar")
    assert init_rev == git.get_ref("refs/foo/baz")
    assert init_rev == git.get_ref("refs/tags/annotated", follow=True)
    assert init_rev == git.get_ref("refs/tags/annotated", follow=False)
    assert git.get_ref("refs/foo/baz", follow=False) == "refs/heads/master"
    assert git.get_ref("refs/foo/qux") is None


def test_remove_ref(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    init_rev = scm.get_rev()

    # remove nonexistent ref should silently pass when old_ref is None)
    git.remove_ref("refs/foo/bar", old_ref=None)

    tmp_dir.gen(os.path.join(".git", "refs", "foo", "bar"), init_rev)
    tmp_dir.gen({"file": "1"})
    scm.add_commit("file", message="commit")
    commit_rev = scm.get_rev()

    with pytest.raises(SCMError):
        git.remove_ref("refs/foo/bar", old_ref=commit_rev)
    git.remove_ref("refs/foo/bar", old_ref=init_rev)
    assert not (tmp_dir / ".git" / "refs" / "foo" / "bar").exists()


@pytest.mark.skip_git_backend("dulwich")
def test_refs_containing(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    init_rev = scm.get_rev()
    tmp_dir.gen(
        {
            os.path.join(".git", "refs", "foo", "bar"): init_rev,
            os.path.join(".git", "refs", "foo", "baz"): init_rev,
        }
    )

    expected = {"refs/foo/bar", "refs/foo/baz", "refs/heads/master"}
    assert expected == set(git.get_refs_containing(init_rev))


@pytest.mark.skip_git_backend("pygit2", "gitpython")
@pytest.mark.parametrize("use_url", [True, False])
def test_push_refspecs(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
    remote_git_dir: TmpDir,
    use_url: str,
):
    from scmrepo.git.backend.dulwich import SyncStatus

    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    init_rev = scm.get_rev()
    scm.add_commit("file", message="bar")
    bar_rev = scm.get_rev()
    scm.checkout(init_rev)
    scm.add_commit("file", message="baz")
    baz_rev = scm.get_rev()
    tmp_dir.gen(
        {
            os.path.join(".git", "refs", "foo", "bar"): bar_rev,
            os.path.join(".git", "refs", "foo", "baz"): baz_rev,
        }
    )

    url = f"file://{remote_git_dir.resolve().as_posix()}"
    remote_scm = Git(remote_git_dir)
    scm.gitpython.repo.create_remote("origin", url)

    with pytest.raises(SCMError):
        git.push_refspecs("bad-remote", "refs/foo/bar:refs/foo/bar")

    remote = url if use_url else "origin"
    assert git.push_refspecs(remote, "refs/foo/bar:refs/foo/bar") == {
        "refs/foo/bar": SyncStatus.SUCCESS
    }
    assert bar_rev == remote_scm.get_ref("refs/foo/bar")

    remote_scm.checkout("refs/foo/bar")
    assert bar_rev == remote_scm.get_rev()
    assert (remote_git_dir / "file").read_text() == "0"

    assert git.push_refspecs(
        remote, ["refs/foo/bar:refs/foo/bar", "refs/foo/baz:refs/foo/baz"]
    ) == {
        "refs/foo/bar": SyncStatus.UP_TO_DATE,
        "refs/foo/baz": SyncStatus.SUCCESS,
    }
    assert baz_rev == remote_scm.get_ref("refs/foo/baz")

    assert git.push_refspecs(remote, ["refs/foo/bar:refs/foo/baz"]) == {
        "refs/foo/baz": SyncStatus.DIVERGED
    }
    assert baz_rev == remote_scm.get_ref("refs/foo/baz")

    assert git.push_refspecs(remote, ":refs/foo/baz") == {
        "refs/foo/baz": SyncStatus.SUCCESS
    }
    assert remote_scm.get_ref("refs/foo/baz") is None


@pytest.mark.skip_git_backend("gitpython")
@pytest.mark.parametrize("use_url", [True, False])
def test_fetch_refspecs(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
    remote_git_dir: TmpDir,
    use_url: bool,
    mocker: MockerFixture,
):
    from scmrepo.git.backend.dulwich import SyncStatus

    url = f"file://{remote_git_dir.resolve().as_posix()}"
    scm.gitpython.repo.create_remote("origin", url)
    remote_scm = Git(remote_git_dir)
    remote_git_dir.gen("file", "0")

    remote_scm.add_commit("file", message="init")
    init_rev = remote_scm.get_rev()
    remote_scm.add_commit("file", message="bar")
    bar_rev = remote_scm.get_rev()
    remote_scm.checkout(init_rev)
    remote_scm.add_commit("file", message="baz")
    baz_rev = remote_scm.get_rev()
    remote_git_dir.gen(
        {
            os.path.join(".git", "refs", "foo", "bar"): bar_rev,
            os.path.join(".git", "refs", "foo", "baz"): baz_rev,
        }
    )

    with pytest.raises(SCMError):
        git.fetch_refspecs("bad-remote", "refs/foo/bar:refs/foo/bar")

    remote = url if use_url else "origin"
    assert git.fetch_refspecs(remote, "refs/foo/bar:refs/foo/bar") == {
        "refs/foo/bar": SyncStatus.SUCCESS
    }
    assert bar_rev == scm.get_ref("refs/foo/bar")

    assert git.fetch_refspecs(
        remote, ["refs/foo/bar:refs/foo/bar", "refs/foo/baz:refs/foo/baz"]
    ) == {
        "refs/foo/bar": SyncStatus.UP_TO_DATE,
        "refs/foo/baz": SyncStatus.SUCCESS,
    }
    assert baz_rev == scm.get_ref("refs/foo/baz")

    assert git.fetch_refspecs(remote, ["refs/foo/bar:refs/foo/baz"]) == {
        "refs/foo/baz": SyncStatus.DIVERGED
    }
    assert baz_rev == scm.get_ref("refs/foo/baz")

    mocker.patch.object(LocalGitClient, "fetch", side_effect=KeyError)
    mocker.patch.object(Remote, "fetch", side_effect=GitError)
    with pytest.raises(SCMError):
        git.fetch_refspecs(remote, "refs/foo/bar:refs/foo/bar")

    assert len(scm.pygit2.repo.remotes) == 1


@pytest.mark.skip_git_backend("pygit2", "gitpython")
@pytest.mark.parametrize("use_url", [True, False])
def test_iter_remote_refs(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
    remote_git_dir: TmpDir,
    tmp_dir_factory: TempDirFactory,
    use_url: bool,
):
    url = f"file://{remote_git_dir.resolve().as_posix()}"

    scm.gitpython.repo.create_remote("origin", url)
    remote_scm = Git(remote_git_dir)
    remote_git_dir.gen("file", "0")
    remote_scm.add_commit("file", message="init")
    remote_scm.branch("new-branch")
    remote_scm.add_commit("file", message="bar")
    remote_scm.add_commit("file", message="baz")
    remote_scm.tag("a-tag")

    with pytest.raises(InvalidRemote):
        set(git.iter_remote_refs("bad-remote"))

    tmp_directory = tmp_dir_factory.mktemp("not_a_git_repo")
    remote = f"file://{tmp_directory.as_posix()}"
    with pytest.raises(InvalidRemote):
        set(git.iter_remote_refs(remote))

    remote = url if use_url else "origin"
    assert {
        "refs/heads/master",
        "HEAD",
        "refs/heads/new-branch",
        "refs/tags/a-tag",
    } == set(git.iter_remote_refs(remote))


def _gen(scm: Git, s: str, commit_timestamp: Optional[float] = None) -> str:
    with open(s, mode="w") as f:
        f.write(s)
    scm.dulwich.add([s])
    scm.dulwich.repo.do_commit(
        message=s.encode("utf-8"), commit_timestamp=commit_timestamp
    )
    return scm.get_rev()


def test_list_all_commits(tmp_dir: TmpDir, scm: Git, git: Git, matcher: type[Matcher]):
    assert git.list_all_commits() == []
    # https://github.com/libgit2/libgit2/issues/6336
    now = time.time()

    rev_a = _gen(scm, "a", commit_timestamp=now - 10)
    rev_b = _gen(scm, "b", commit_timestamp=now - 8)
    rev_c = _gen(scm, "c", commit_timestamp=now - 5)
    rev_d = _gen(scm, "d", commit_timestamp=now - 2)

    assert git.list_all_commits() == [rev_d, rev_c, rev_b, rev_a]

    scm.gitpython.git.reset(rev_b, hard=True)
    assert git.list_all_commits() == [rev_b, rev_a]


def test_list_all_commits_branch(
    tmp_dir: TmpDir, scm: Git, git: Git, matcher: type[Matcher]
):
    revs = {}
    now = time.time()

    revs["1"] = _gen(scm, "a", commit_timestamp=now - 10)

    scm.checkout("branch", create_new=True)
    revs["3"] = _gen(scm, "c", commit_timestamp=now - 9)

    scm.checkout("master")
    revs["2"] = _gen(scm, "b", commit_timestamp=now - 7)

    scm.checkout("branch")
    revs["5"] = _gen(scm, "e", commit_timestamp=now - 6)

    scm.checkout("master")
    revs["4"] = _gen(scm, "d", commit_timestamp=now - 5)

    scm.checkout("branch")
    revs["6"] = _gen(scm, "f", commit_timestamp=now - 4)

    scm.checkout("master")
    revs["7"] = _gen(scm, "g", commit_timestamp=now - 3)
    revs["8"] = scm.merge("branch", msg="merge branch")

    inv_map = {v: k for k, v in revs.items()}
    assert [inv_map[k] for k in git.list_all_commits()] == [
        "8",
        "7",
        "6",
        "4",
        "5",
        "2",
        "3",
        "1",
    ]


def test_list_all_tags(tmp_dir: TmpDir, scm: Git, git: Git, matcher: type[Matcher]):
    rev_a = _gen(scm, "a")
    scm.tag("tag")
    rev_b = _gen(scm, "b")
    scm.tag("annotated", annotated=True, message="Annotated Tag")
    rev_c = _gen(scm, "c")
    rev_d = _gen(scm, "d")
    assert git.list_all_commits() == matcher.unordered(rev_d, rev_c, rev_b, rev_a)

    rev_e = _gen(scm, "e")
    scm.tag(
        "annotated2",
        target="refs/tags/annotated",
        annotated=True,
        message="Annotated Tag",
    )
    assert git.list_all_commits() == matcher.unordered(
        rev_e, rev_d, rev_c, rev_b, rev_a
    )

    rev_f = _gen(scm, "f")
    scm.tag(
        "annotated3",
        target="refs/tags/annotated2",
        annotated=True,
        message="Annotated Tag 3",
    )
    assert git.list_all_commits() == matcher.unordered(
        rev_f, rev_e, rev_d, rev_c, rev_b, rev_a
    )

    scm.gitpython.git.reset(rev_a, hard=True)
    assert git.list_all_commits() == matcher.unordered(rev_b, rev_a)


def test_list_all_commits_dangling_annotated_tag(tmp_dir: TmpDir, scm: Git, git: Git):
    rev_a = _gen(scm, "a")
    scm.tag("annotated", annotated=True, message="Annotated Tag")

    _gen(scm, "b")

    # Delete branch pointing to rev_a
    scm.checkout(rev_a)
    scm.gitpython.repo.delete_head("master", force=True)

    assert git.list_all_commits() == [rev_a]  # Only reachable via the tag


def test_list_all_commits_orphan(
    tmp_dir: TmpDir, scm: Git, git: Git, matcher: type[Matcher]
):
    rev_a = _gen(scm, "a")

    # Make an orphan branch
    scm.gitpython.git.checkout("--orphan", "orphan-branch")
    rev_orphan = _gen(scm, "orphanfile")

    assert rev_orphan != rev_a
    assert git.list_all_commits() == matcher.unordered(rev_orphan, rev_a)


def test_list_all_commits_refs(
    tmp_dir: TmpDir, scm: Git, git: Git, matcher: type[Matcher]
):
    assert git.list_all_commits() == []

    rev_a = _gen(scm, "a")

    assert git.list_all_commits() == [rev_a]
    rev_b = _gen(scm, "b")
    scm.set_ref("refs/remotes/origin/feature", rev_b)
    assert git.list_all_commits() == matcher.unordered(rev_b, rev_a)

    # also add refs/exps/foo/bar
    rev_c = _gen(scm, "c")
    scm.set_ref("refs/exps/foo/bar", rev_c)
    assert git.list_all_commits() == matcher.unordered(rev_c, rev_b, rev_a)

    # Dangling/broken ref ---
    scm.set_ref("refs/heads/bad-ref", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    with pytest.raises(Exception):  # noqa: B017, PT011
        git.list_all_commits()
    scm.remove_ref("refs/heads/bad-ref")

    scm.gitpython.git.reset(rev_a, hard=True)
    assert git.list_all_commits() == matcher.unordered(rev_b, rev_a)


def test_list_all_commits_detached_head(
    tmp_dir: TmpDir, scm: Git, git: Git, matcher: type[Matcher]
):
    rev_a = _gen(scm, "a")
    rev_b = _gen(scm, "b")
    rev_c = _gen(scm, "c")
    scm.checkout(rev_b)

    assert scm.pygit2.repo.head_is_detached
    assert git.list_all_commits() == matcher.unordered(rev_c, rev_b, rev_a)


@pytest.mark.skip_git_backend("pygit2")
def test_ignore_remove_empty(tmp_dir: TmpDir, scm: Git, git: Git):
    test_entries = [
        {"entry": "/foo1", "path": f"{tmp_dir}/foo1"},
        {"entry": "/foo2", "path": f"{tmp_dir}/foo2"},
    ]

    path_to_gitignore = tmp_dir / ".gitignore"

    with open(path_to_gitignore, "a", encoding="utf-8") as f:
        for entry in test_entries:
            f.write(entry["entry"] + "\n")

    assert path_to_gitignore.exists()

    git.ignore_remove(test_entries[0]["path"])
    assert path_to_gitignore.exists()

    git.ignore_remove(test_entries[1]["path"])
    assert not path_to_gitignore.exists()


@pytest.mark.skip_git_backend("pygit2")
@pytest.mark.skipif(os.name == "nt", reason="Git hooks not supported on Windows")
@pytest.mark.parametrize("hook", ["pre-commit", "commit-msg"])
def test_commit_no_verify(tmp_dir: TmpDir, scm: Git, git: Git, hook: str):
    import stat

    hook_file = os.path.join(".git", "hooks", hook)
    tmp_dir.gen(hook_file, "#!/usr/bin/env python\nimport sys\nsys.exit(1)")
    os.chmod(hook_file, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)

    tmp_dir.gen("foo", "foo")
    git.add(["foo"])
    with pytest.raises(SCMError):
        git.commit("commit foo")
    git.commit("commit foo", no_verify=True)


@pytest.mark.skip_git_backend("dulwich")
@pytest.mark.parametrize("squash", [True, False])
def test_merge(tmp_dir: TmpDir, scm: Git, git: Git, squash: bool):
    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="init")
    init_rev = scm.get_rev()

    scm.checkout("branch", create_new=True)
    tmp_dir.gen("foo", "bar")
    scm.add_commit("foo", message="bar")
    branch = scm.resolve_rev("branch")

    scm.checkout("master")

    tmp_dir.gen("foo", "baz")
    scm.add_commit("foo", message="baz")
    with pytest.raises(MergeConflictError):
        git.merge(branch, commit=not squash, squash=squash, msg="merge")

    scm.gitpython.git.reset(init_rev, hard=True)
    merge_rev = git.merge(branch, commit=not squash, squash=squash, msg="merge")
    assert (tmp_dir / "foo").read_text() == "bar"
    if squash:
        assert merge_rev is None
        assert scm.get_rev() == init_rev
    else:
        assert scm.get_rev() == merge_rev


@pytest.mark.skip_git_backend("dulwich")
def test_checkout_index(tmp_dir: TmpDir, scm: Git, git: Git):
    files = tmp_dir.gen({"foo": "foo", "bar": "bar", "dir": {"baz": "baz"}})
    scm.add_commit(files, message="init")
    tmp_dir.gen({"foo": "baz", "dir": {"baz": "foo"}})

    with (tmp_dir / "dir").chdir():
        git.checkout_index([os.path.join("..", "foo"), "baz"], force=True)
    assert (tmp_dir / "foo").read_text() == "foo"
    assert (tmp_dir / "dir" / "baz").read_text() == "baz"

    tmp_dir.gen({"foo": "baz", "bar": "baz", "dir": {"baz": "foo"}})
    git.checkout_index(force=True)
    assert (tmp_dir / "foo").read_text() == "foo"
    assert (tmp_dir / "bar").read_text() == "bar"
    assert (tmp_dir / "dir" / "baz").read_text() == "baz"


@pytest.mark.skip_git_backend("dulwich")
@pytest.mark.parametrize("strategy, expected", [("ours", "baz"), ("theirs", "bar")])
def test_checkout_index_conflicts(
    tmp_dir: TmpDir, scm: Git, git: Git, strategy: str, expected: str
):
    tmp_dir.gen({"file": "foo"})
    scm.add_commit("file", message="init")

    scm.checkout("branch", create_new=True)
    tmp_dir.gen({"file": "bar"})
    scm.add_commit("file", message="bar")
    rev_bar = scm.get_rev()

    scm.checkout("master")
    tmp_dir.gen({"file": "baz"})
    scm.add_commit("file", message="baz")

    with pytest.raises(MergeConflictError):
        git.merge(rev_bar, commit=False, squash=True)

    git.checkout_index(
        ours=strategy == "ours",
        theirs=strategy == "theirs",
    )
    assert (tmp_dir / "file").read_text() == expected


@pytest.mark.skip_git_backend("dulwich")
def test_resolve_rev(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
    remote_git_dir: TmpDir,
):
    url = f"file://{remote_git_dir.resolve().as_posix()}"
    scm.gitpython.repo.create_remote("origin", url)
    scm.gitpython.repo.create_remote("upstream", url)

    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    init_rev = scm.get_rev()

    tmp_dir.gen({"file": "1"})
    scm.add_commit("file", message="1")
    rev = scm.get_rev()

    scm.checkout("branch", create_new=True)
    tmp_dir.gen(
        {
            os.path.join(".git", "refs", "foo"): rev,
            os.path.join(".git", "refs", "remotes", "origin", "bar"): rev,
            os.path.join(".git", "refs", "remotes", "origin", "baz"): rev,
            os.path.join(".git", "refs", "remotes", "upstream", "baz"): init_rev,
        }
    )

    assert git.resolve_rev(rev) == rev
    assert git.resolve_rev(rev[:7]) == rev
    assert git.resolve_rev("HEAD") == rev
    assert git.resolve_rev("branch") == rev
    assert git.resolve_rev("refs/foo") == rev
    assert git.resolve_rev("bar") == rev
    assert git.resolve_rev("origin/baz") == rev
    assert git.resolve_rev("HEAD~1") == init_rev

    with pytest.raises(RevError):
        git.resolve_rev("qux")

    with pytest.raises(RevError):
        git.resolve_rev("baz")

    with pytest.raises(RevError):
        git.resolve_rev("HEAD~3")


@pytest.mark.skip_git_backend("dulwich")
def test_checkout(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"foo": "foo"})
    scm.add_commit("foo", message="foo")
    foo_rev = scm.get_rev()

    tmp_dir.gen("foo", "bar")
    scm.add_commit("foo", message="bar")
    bar_rev = scm.get_rev()

    git.checkout("branch", create_new=True)
    assert git.get_ref("HEAD", follow=False) == "refs/heads/branch"
    assert (tmp_dir / "foo").read_text() == "bar"

    git.checkout("master", detach=True)
    assert git.get_ref("HEAD", follow=False) == bar_rev

    git.checkout("master")
    assert git.get_ref("HEAD", follow=False) == "refs/heads/master"

    git.checkout(foo_rev[:7])
    assert git.get_ref("HEAD", follow=False) == foo_rev
    assert (tmp_dir / "foo").read_text() == "foo"


@pytest.mark.skip_git_backend("dulwich")
def test_reset(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"foo": "foo", "dir": {"baz": "baz"}})
    scm.add_commit(["foo", "dir"], message="init")

    tmp_dir.gen({"foo": "bar", "dir": {"baz": "bar"}})
    scm.add(["foo", os.path.join("dir", "baz")])
    git.reset()
    assert (tmp_dir / "foo").read_text() == "bar"
    assert (tmp_dir / "dir" / "baz").read_text() == "bar"
    staged, unstaged, _ = scm.status()
    assert len(staged) == 0
    assert set(unstaged) == {"foo", "dir/baz"}

    scm.add(["foo", os.path.join("dir", "baz")])
    git.reset(hard=True)
    assert (tmp_dir / "foo").read_text() == "foo"
    assert (tmp_dir / "dir" / "baz").read_text() == "baz"
    staged, unstaged, _ = scm.status()
    assert len(staged) == 0
    assert len(unstaged) == 0

    tmp_dir.gen({"foo": "bar", "bar": "bar", "dir": {"baz": "bar"}})
    scm.add(["foo", "bar", os.path.join("dir", "baz")])
    with (tmp_dir / "dir").chdir():
        git.reset(paths=[os.path.join("..", "foo"), os.path.join("baz")])
    assert (tmp_dir / "foo").read_text() == "bar"
    assert (tmp_dir / "bar").read_text() == "bar"
    assert (tmp_dir / "dir" / "baz").read_text() == "bar"
    staged, unstaged, _ = scm.status()
    assert len(staged) == 1
    assert len(unstaged) == 2


@pytest.mark.skip_git_backend("pygit2")
def test_add(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"foo": "foo", "bar": "bar", "dir": {"baz": "baz"}})
    git.add(["foo", "dir"])
    staged, unstaged, untracked = scm.status()
    assert set(staged["add"]) == {"foo", "dir/baz"}
    assert len(unstaged) == 0
    assert len(untracked) == 1

    scm.commit("commit")
    tmp_dir.gen({"foo": "bar", "dir": {"baz": "bar"}})
    git.add([], update=True)
    staged, unstaged, _ = scm.status()
    assert set(staged["modify"]) == {"foo", "dir/baz"}
    assert len(unstaged) == 0
    assert len(untracked) == 1

    scm.reset()
    git.add(["dir"], update=True)
    staged, unstaged, _ = scm.status()
    assert set(staged["modify"]) == {"dir/baz"}
    assert len(unstaged) == 1
    assert len(untracked) == 1


@pytest.mark.skip_git_backend("pygit2")
def test_add_force(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"foo": "foo", "bar": "bar", "dir": {"baz": "baz"}})
    tmp_dir.gen({".gitignore": "foo\ndir/"})

    git.add(["foo", "bar", "dir"])
    staged, _unstaged, untracked = scm.status()
    assert set(staged["add"]) == {"bar"}
    assert set(untracked) == {".gitignore"}

    git.add(["foo", "bar", "dir"], force=True)
    staged, _unstaged, untracked = scm.status()
    assert set(staged["add"]) == {"foo", "bar", "dir/baz"}
    assert set(untracked) == {".gitignore"}


@pytest.mark.skip_git_backend("dulwich", "gitpython")
def test_checkout_subdir(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="init")
    rev = scm.get_rev()

    tmp_dir.gen({"dir": {"bar": "bar"}})
    scm.add_commit("dir", message="dir")

    with (tmp_dir / "dir").chdir():
        git.checkout(rev)
        assert not (tmp_dir / "dir" / "bar").exists()


@pytest.mark.skip_git_backend("pygit2", "gitpython")
def test_describe(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"foo": "foo"})
    scm.add_commit("foo", message="foo")
    rev_foo = scm.get_rev()

    tmp_dir.gen({"foo": "bar"})
    scm.add_commit("foo", message="bar")
    rev_bar = scm.get_rev()

    assert git.describe([rev_foo], "refs/heads") == {rev_foo: None}

    scm.checkout("branch", create_new=True)
    assert git.describe([rev_bar], "refs/heads") == {rev_bar: "refs/heads/branch"}

    tmp_dir.gen({"foo": "foobar"})
    scm.add_commit("foo", message="foobar")
    rev_foobar = scm.get_rev()

    scm.checkout("master")
    assert git.describe([rev_bar, rev_foobar], "refs/heads") == {
        rev_bar: "refs/heads/master",
        rev_foobar: "refs/heads/branch",
    }

    scm.tag("tag")
    assert git.describe([rev_bar, "nonexist"]) == {
        rev_bar: "refs/tags/tag",
        "nonexist": None,
    }


def test_ignore(tmp_dir: TmpDir, scm: Git, git: Git):
    file = os.fspath(tmp_dir / "foo")

    git.ignore(file)
    assert (tmp_dir / ".gitignore").cat() == "/foo\n"

    git._reset()
    git.ignore(file)
    assert (tmp_dir / ".gitignore").cat() == "/foo\n"

    git._reset()
    git.ignore_remove(file)
    assert not (tmp_dir / ".gitignore").exists()


def test_ignored(tmp_dir: TmpDir, scm: Git, git: Git, git_backend: str):
    tmp_dir.gen({"dir1": {"file1.jpg": "cont", "file2.txt": "cont"}})
    tmp_dir.gen({".gitignore": "dir1/*.jpg"})

    git._reset()

    assert git.is_ignored(tmp_dir / "dir1" / "file1.jpg")
    assert not git.is_ignored(tmp_dir / "dir1" / "file2.txt")


@pytest.mark.skip_git_backend("pygit2", "gitpython")
def test_ignored_dir_unignored_subdirs(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({".gitignore": "data/**\n!data/**/\n!data/**/*.csv"})
    scm.add([".gitignore"])
    tmp_dir.gen(
        {
            os.path.join("data", "raw", "tracked.csv"): "cont",
            os.path.join("data", "raw", "not_tracked.json"): "cont",
        }
    )

    git._reset()

    assert not git.is_ignored(tmp_dir / "data" / "raw" / "tracked.csv")
    assert git.is_ignored(tmp_dir / "data" / "raw" / "not_tracked.json")
    assert not git.is_ignored(tmp_dir / "data" / "raw" / "non_existent.csv")
    assert git.is_ignored(tmp_dir / "data" / "raw" / "non_existent.json")
    assert not git.is_ignored(tmp_dir / "data" / "non_existent.csv")
    assert git.is_ignored(tmp_dir / "data" / "non_existent.json")

    assert not git.is_ignored(f"data{os.sep}")
    # git check-ignore would now mark "data/raw" as ignored
    # after detecting it's a directory in the file system;
    # instead, we rely on the trailing separator to determine if handling a
    # a directory - for consistency between existent and non-existent paths
    assert git.is_ignored(os.path.join("data", "raw"))
    assert not git.is_ignored(os.path.join("data", f"raw{os.sep}"))

    assert git.is_ignored(os.path.join("data", "non_existent"))
    assert not git.is_ignored(os.path.join("data", f"non_existent{os.sep}"))


def test_get_gitignore(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"file1": "contents", "dir": {}})

    data_dir = os.fspath(tmp_dir / "file1")
    entry, gitignore = git._get_gitignore(data_dir)
    assert entry == "/file1"
    assert gitignore == os.fspath(tmp_dir / ".gitignore")

    data_dir = os.fspath(tmp_dir / "dir")
    entry, gitignore = git._get_gitignore(data_dir)

    assert entry == "/dir"
    assert gitignore == os.fspath(tmp_dir / ".gitignore")


def test_get_gitignore_symlink(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"dir": {"subdir": {"data": "contents"}}})
    link = tmp_dir / "link"
    link.symlink_to(tmp_dir / "dir" / "subdir" / "data")

    entry, gitignore = git._get_gitignore(os.fspath(link))
    assert entry == "/link"
    assert gitignore == os.fspath(tmp_dir / ".gitignore")


def test_get_gitignore_subdir(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"dir1": {"file1": "cont", "dir2": {}}})

    data_dir = os.fspath(tmp_dir / "dir1" / "file1")
    entry, gitignore = git._get_gitignore(data_dir)
    assert entry == "/file1"
    assert gitignore == os.fspath(tmp_dir / "dir1" / ".gitignore")

    data_dir = os.fspath(tmp_dir / "dir1" / "dir2")
    entry, gitignore = git._get_gitignore(data_dir)
    assert entry == "/dir2"
    assert gitignore == os.fspath(tmp_dir / "dir1" / ".gitignore")


def test_gitignore_should_append_newline_to_gitignore(
    tmp_dir: TmpDir, scm: Git, git: Git
):
    tmp_dir.gen({"foo": "foo", "bar": "bar"})

    bar_path = os.fspath(tmp_dir / "bar")
    gitignore = tmp_dir / ".gitignore"

    gitignore.write_text("/foo")
    assert not gitignore.read_text().endswith("\n")

    git.ignore(bar_path)
    contents = gitignore.read_text()
    assert gitignore.read_text().endswith("\n")

    assert contents.splitlines() == ["/foo", "/bar"]


@pytest.mark.skip_git_backend("dulwich")
def test_git_detach_head(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    init_rev = scm.get_rev()

    with git.detach_head() as rev:
        assert init_rev == rev
        assert init_rev == (tmp_dir / ".git" / "HEAD").read_text().strip()
    assert (tmp_dir / ".git" / "HEAD").read_text().strip() == "ref: refs/heads/master"


@pytest.mark.skip_git_backend("pygit2", "gitpython")
@pytest.mark.slow
@pytest.mark.asyncio
async def test_git_ssh(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
    sftp: SFTPClient,
    ssh_connection: SSHClientConnection,
    ssh_conn_info: dict[str, Any],
):
    host, port = ssh_conn_info["host"], ssh_conn_info["port"]
    key_filename = ssh_conn_info["client_keys"]
    user = ssh_conn_info["username"]
    url = f"ssh://{user}@{host}:{port}/~/test-repo.git"

    result = await ssh_connection.run("git init --bare test-repo.git")
    assert result.returncode == 0

    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="init")
    rev = scm.get_rev()

    git.push_refspecs(
        url,
        "refs/heads/master:refs/heads/master",
        force=True,
        key_filename=key_filename,
    )

    async with sftp.open("test-repo.git/refs/heads/master") as fobj:
        assert (await fobj.read(-1, 0)).strip() == rev

    shutil.rmtree(tmp_dir / ".git")
    (tmp_dir / "foo").unlink()
    scm = Git.init(str(tmp_dir))

    git.fetch_refspecs(
        url,
        ["refs/heads/master"],
        force=True,
        key_filename=key_filename,
    )
    assert git.get_ref("refs/heads/master") == rev
    scm.checkout("master")
    assert (tmp_dir / "foo").read_text() == "foo"


@pytest.mark.parametrize("scheme", ["", "file://"])
@pytest.mark.parametrize("shallow_branch", [None, "master"])
@pytest.mark.parametrize("bare", [True, False])
def test_clone(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
    tmp_dir_factory: TempDirFactory,
    scheme: str,
    shallow_branch: Optional[str],
    bare: bool,
):
    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="init")
    rev = scm.get_rev()

    target_dir = tmp_dir_factory.mktemp("git-clone")
    git.clone(
        f"{scheme}{tmp_dir}",
        str(target_dir),
        shallow_branch=shallow_branch,
        bare=bare,
    )
    target = Git(str(target_dir))
    assert target.get_rev() == rev
    if bare:
        assert not (target_dir / "foo").exists()
    else:
        assert (target_dir / "foo").read_text() == "foo"
        assert (target_dir / "foo").read_text() == "foo"
    fs = target.get_fs(rev)
    with fs.open("foo", mode="r") as fobj:
        assert fobj.read().strip() == "foo"


@pytest.fixture
def proxy_server():
    class _ProxyServer(ProxyTestCase):
        pass

    _ProxyServer.setUpClass()
    assert _ProxyServer.PROXY
    yield f"http://{_ProxyServer.PROXY.flags.hostname}:{_ProxyServer.PROXY.flags.port}"
    _ProxyServer.tearDownClass()


def test_clone_proxy_server(proxy_server: str, scm: Git, git: Git, tmp_dir: TmpDir):
    url = "https://github.com/iterative/dvcyaml-schema"

    p = (
        Path(os.environ["HOME"] if "HOME" in os.environ else os.environ["USERPROFILE"])
        / ".gitconfig"
    )
    p.write_text(BAD_PROXY_CONFIG)
    with pytest.raises(Exception):  # noqa: PT011, B017
        git.clone(url, "dir")

    mock_config_content = f"""[http]\n
proxy = {proxy_server}
[https]
proxy = {proxy_server}
"""

    p.write_text(mock_config_content)
    git.clone(url, "dir")


def test_iter_remote_refs_proxy_server(proxy_server: str, scm: Git, tmp_dir: TmpDir):
    url = "https://github.com/iterative/dvcyaml-schema"
    git = GitBackends.DEFAULT["dulwich"](".")

    p = (
        Path(os.environ["HOME"] if "HOME" in os.environ else os.environ["USERPROFILE"])
        / ".gitconfig"
    )
    p.write_text(BAD_PROXY_CONFIG)
    with pytest.raises(Exception):  # noqa: PT011, B017
        list(git.iter_remote_refs(url))

    mock_config_content = f"""[http]
proxy = {proxy_server}
[https]
proxy = {proxy_server}
"""

    p.write_text(mock_config_content)
    res = list(git.iter_remote_refs(url))
    assert res


@pytest.mark.skip_git_backend("gitpython")
def test_fetch_refspecs_proxy_server(
    proxy_server: str, scm: Git, git: Git, tmp_dir: TmpDir
):
    url = "https://github.com/iterative/dvcyaml-schema"

    p = (
        Path(os.environ["HOME"] if "HOME" in os.environ else os.environ["USERPROFILE"])
        / ".gitconfig"
    )
    p.write_text(BAD_PROXY_CONFIG)
    with pytest.raises(Exception):  # noqa: PT011, B017
        git.fetch_refspecs(url, ["refs/heads/master:refs/heads/master"])

    mock_config_content = f"""[http]
proxy = {proxy_server}
[https]
proxy = {proxy_server}
"""

    p.write_text(mock_config_content)
    git.fetch_refspecs(url, "refs/heads/master:refs/heads/master")


@pytest.mark.skip_git_backend("pygit2")
def test_fetch(tmp_dir: TmpDir, scm: Git, git: Git, tmp_dir_factory: TempDirFactory):
    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="init")

    target_dir = tmp_dir_factory.mktemp("git-clone")
    git.clone(str(tmp_dir), target_dir)
    target = Git(str(target_dir))

    scm.add_commit("bar", message="update")
    rev = scm.get_rev()

    target.fetch()
    assert target.get_ref("refs/remotes/origin/master") == rev


@pytest.mark.skip_git_backend("gitpython")
@pytest.mark.parametrize("untracked_files", ["all", "no", "normal"])
@pytest.mark.parametrize("ignored", [False, True])
def test_status(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
    git_backend,
    tmp_dir_factory: TempDirFactory,
    untracked_files: str,
    ignored: bool,
):
    if untracked_files == "normal" and git_backend == "dulwich":
        pytest.skip(
            "untracked_files=normal is not implemented for dulwich",
        )

    tmp_dir.gen(
        {
            "foo": "foo",
            "bar": "bar",
            ".gitignore": "ignored",
            "ignored": "ignored",
        }
    )
    scm.add_commit(["foo", "bar", ".gitignore"], message="init")

    staged, unstaged, untracked = git.status(ignored, untracked_files)

    assert not staged
    assert not unstaged
    if ignored and untracked_files != "no":
        assert untracked == ["ignored"]
    else:
        assert not untracked

    with (tmp_dir / "foo").open("a") as fobj:
        fobj.write("modified")

    with (tmp_dir / "bar").open("a") as fobj:
        fobj.write("modified")

    tmp_dir.gen({"untracked_dir": {"subfolder": {"subfile": "subfile"}}})
    expected_untracked = []
    if ignored and untracked_files != "no":
        expected_untracked.append("ignored")
    if untracked_files == "all":
        expected_untracked.append("untracked_dir/subfolder/subfile")
    elif untracked_files == "normal":
        expected_untracked.append("untracked_dir/")

    scm.add("foo")
    staged, unstaged, untracked = git.status(ignored, untracked_files)

    assert staged["modify"] == ["foo"]
    assert unstaged == ["bar"]
    assert untracked == expected_untracked


@pytest.mark.skip_git_backend("pygit2")
def test_is_dirty_empty(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
):
    assert not git.is_dirty()


@pytest.mark.skip_git_backend("pygit2")
def test_is_dirty_added(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
):
    tmp_dir.gen("foo", "foo")
    scm.add("foo")

    assert git.is_dirty()
    scm.commit("commit")
    assert not git.is_dirty()


@pytest.mark.skip_git_backend("pygit2")
def test_is_dirty_modified(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
):
    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="add foo")

    tmp_dir.gen("foo", "modified")
    assert git.is_dirty()
    scm.add_commit("foo", message="modified")
    assert not git.is_dirty()


@pytest.mark.skip_git_backend("pygit2")
def test_is_dirty_deleted(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
):
    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="add foo")

    assert not git.is_dirty()
    os.unlink(tmp_dir / "foo")
    assert git.is_dirty()
    scm.add("foo")
    assert git.is_dirty()
    scm.commit("deleted")
    assert not git.is_dirty()


@pytest.mark.skip_git_backend("pygit2")
def test_is_dirty_untracked(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
):
    tmp_dir.gen("untracked", "untracked")
    assert git.is_dirty(untracked_files=True)
    assert not git.is_dirty(untracked_files=False)


@pytest.mark.parametrize(
    "backends", [["gitpython", "dulwich"], ["dulwich", "gitpython"]]
)
def test_backend_func(
    tmp_dir: TmpDir,
    scm: Git,
    backends: list[str],
    mocker: MockerFixture,
):
    from functools import partial

    scm.add = partial(scm._backend_func, "add", backends=backends)
    tmp_dir.gen({"foo": "foo"})
    backend = getattr(scm, backends[0])
    mock = mocker.spy(backend, "add")
    scm.add(["foo"])
    mock.assert_called_once_with(["foo"])


def test_tag(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="init")
    rev = scm.get_rev()

    git.tag("lightweight")
    assert scm.resolve_rev("lightweight") == rev

    git.tag("annotated", annotated=True, message="message")
    assert scm.resolve_rev("annotated") == rev


def test_get_tag(tmp_dir, scm: Git, git: Git):
    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="init")
    rev = scm.get_rev()

    assert git.get_tag("nonexistent") is None

    scm.tag("lightweight")
    assert git.get_tag("lightweight") == rev

    scm.tag("annotated", annotated=True, message="message")
    tag = git.get_tag("annotated")
    assert isinstance(tag, GitTag)
    assert tag.target == rev
    assert tag.message.strip() == "message"


@pytest.mark.skip_git_backend("gitpython")
def test_config(tmp_dir, scm: Git, git: Git):
    tmp_dir.gen({".git": {"config": "[test]\nfoo = true\n"}})
    tmp_dir.gen(".otherconfig", "[test]\nfoo = false\n")
    config = git.get_config()
    assert config.get(("test",), "foo") == "true"
    assert config.get_bool(("test",), "foo") is True

    config = git.get_config(".otherconfig")
    assert config.get(("test",), "foo") == "false"
    assert config.get_bool(("test",), "foo") is False


@pytest.mark.skip_git_backend("dulwich")
def test_check_attr(tmp_dir, scm: Git, git: Git):
    tmp_dir.gen("foo.txt", "foo")
    tmp_dir.gen(".gitattributes", "*.txt text")
    scm.add_commit([".gitattributes", "foo"], message="init")
    rev = scm.get_rev()
    assert git.check_attr("foo.txt", "text") is True
    assert git.check_attr("foo.txt", "filter") is None

    tmp_dir.gen(".gitattributes", "*.txt -text filter=lfs")
    assert git.check_attr("foo.txt", "text") is False
    assert git.check_attr("foo.txt", "filter") == "lfs"
    assert git.check_attr("foo.txt", "text", source=rev) is True
    assert git.check_attr("foo.txt", "filter", source=rev) is None
