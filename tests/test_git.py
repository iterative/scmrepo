import os
from typing import Iterator, Type

import pytest
from pytest_test_utils import TempDirFactory, TmpDir
from pytest_test_utils.matchers import Matcher

from scmrepo.exceptions import MergeConflictError, RevError, SCMError
from scmrepo.git import Git

# pylint: disable=redefined-outer-name,unused-argument


@pytest.fixture(params=["gitpython", "dulwich", "pygit2"])
def git_backend(request) -> str:
    marker = request.node.get_closest_marker("skip_git_backend")
    to_skip = marker.args if marker else []

    backend = request.param
    if backend in to_skip:
        pytest.skip()
    return backend


@pytest.fixture
def git(tmp_dir: TmpDir, git_backend: str) -> Iterator[Git]:
    git_ = Git(tmp_dir, backends=[git_backend])
    yield git_
    git_.close()


@pytest.fixture
def remote_git_dir(tmp_dir_factory: TempDirFactory):
    git_dir = tmp_dir_factory.mktemp("git-remote")
    remote_git = Git.init(git_dir)
    remote_git.close()
    return git_dir


@pytest.mark.parametrize("backend", ["gitpython", "dulwich", "pygit2"])
def test_git_init(tmp_dir: TmpDir, backend: str):
    Git.init(".", _backend=backend)
    assert (tmp_dir / ".git").is_dir()
    Git(tmp_dir)


@pytest.mark.parametrize("backend", ["gitpython", "dulwich", "pygit2"])
def test_git_init_bare(tmp_dir: TmpDir, backend: str):
    Git.init(".", bare=True, _backend=backend)
    assert list(tmp_dir.iterdir())
    Git(tmp_dir)


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


def test_walk_with_submodules(
    tmp_dir: Git,
    scm: Git,
    remote_git_dir: TmpDir,
):
    remote_git = Git(remote_git_dir)
    remote_git_dir.gen({"foo": "foo", "bar": "bar", "dir": {"data": "data"}})
    remote_git.add_commit(["foo", "bar", "dir"], message="add dir and files")
    scm.gitpython.repo.create_submodule(
        "submodule", "submodule", url=os.fspath(remote_git_dir)
    )
    scm.commit("added submodule")

    files = []
    dirs = []
    fs = scm.get_fs("HEAD")
    for _, dnames, fnames in fs.walk("."):
        dirs.extend(dnames)
        files.extend(fnames)

    # currently we don't walk through submodules
    assert not dirs
    assert set(files) == {".gitmodules", "submodule"}


def test_walk_onerror(tmp_dir: TmpDir, scm: Git):
    def onerror(exc):
        raise exc

    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="init")

    fs = scm.get_fs("HEAD")

    # path does not exist
    for _ in fs.walk("dir"):
        pass
    with pytest.raises(OSError):
        for _ in fs.walk("dir", onerror=onerror):
            pass

    # path is not a directory
    for _ in fs.walk("foo"):
        pass
    with pytest.raises(OSError):
        for _ in fs.walk("foo", onerror=onerror):
            pass


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

    base, *others = [_gen(i) for i in range(5)]
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
    assert (
        init_rev
        == (tmp_dir / ".git" / "refs" / "foo" / "bar").read_text().strip()
    )

    with pytest.raises(SCMError):
        git.set_ref("refs/foo/bar", commit_rev, old_ref=commit_rev)
    git.set_ref("refs/foo/bar", commit_rev, old_ref=init_rev)
    assert (
        commit_rev
        == (tmp_dir / ".git" / "refs" / "foo" / "bar").read_text().strip()
    )

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
            os.path.join(
                ".git", "refs", "foo", "baz"
            ): "ref: refs/heads/master",
        }
    )

    assert init_rev == git.get_ref("refs/foo/bar")
    assert init_rev == git.get_ref("refs/foo/baz")
    assert git.get_ref("refs/foo/baz", follow=False) == "refs/heads/master"
    assert git.get_ref("refs/foo/qux") is None


def test_remove_ref(tmp_dir: TmpDir, scm: Git, git: Git):
    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    init_rev = scm.get_rev()

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
def test_push_refspec(
    tmp_dir: TmpDir,
    scm: Git,
    git: Git,
    remote_git_dir: TmpDir,
    use_url: str,
):
    tmp_dir.gen({"file": "0"})
    scm.add_commit("file", message="init")
    init_rev = scm.get_rev()
    tmp_dir.gen(
        {
            os.path.join(".git", "refs", "foo", "bar"): init_rev,
            os.path.join(".git", "refs", "foo", "baz"): init_rev,
        }
    )

    url = f"file://{remote_git_dir.resolve().as_posix()}"
    remote_scm = Git(remote_git_dir)
    scm.gitpython.repo.create_remote("origin", url)

    with pytest.raises(SCMError):
        git.push_refspec("bad-remote", "refs/foo/bar", "refs/foo/bar")

    remote = url if use_url else "origin"
    git.push_refspec(remote, "refs/foo/bar", "refs/foo/bar")
    assert init_rev == remote_scm.get_ref("refs/foo/bar")

    remote_scm.checkout("refs/foo/bar")
    assert init_rev == remote_scm.get_rev()
    assert (remote_git_dir / "file").read_text() == "0"

    git.push_refspec(remote, "refs/foo/", "refs/foo/")
    assert init_rev == remote_scm.get_ref("refs/foo/baz")

    git.push_refspec(remote, None, "refs/foo/baz")
    assert remote_scm.get_ref("refs/foo/baz") is None


@pytest.mark.skip_git_backend("pygit2", "gitpython")
def test_fetch_refspecs(
    scm: Git,
    git: Git,
    remote_git_dir: TmpDir,
):
    url = f"file://{remote_git_dir.resolve().as_posix()}"

    remote_scm = Git(remote_git_dir)
    remote_git_dir.gen("file", "0")
    remote_scm.add_commit("file", message="init")

    init_rev = remote_scm.get_rev()

    remote_git_dir.gen(
        {
            os.path.join(".git", "refs", "foo", "bar"): init_rev,
            os.path.join(".git", "refs", "foo", "baz"): init_rev,
        }
    )

    git.fetch_refspecs(
        url, ["refs/foo/bar:refs/foo/bar", "refs/foo/baz:refs/foo/baz"]
    )
    assert init_rev == scm.get_ref("refs/foo/bar")
    assert init_rev == scm.get_ref("refs/foo/baz")

    remote_scm.checkout("refs/foo/bar")
    assert init_rev == remote_scm.get_rev()
    assert (remote_git_dir / "file").read_text() == "0"


@pytest.mark.skip_git_backend("dulwich", "pygit2")
def test_list_all_commits(
    tmp_dir: TmpDir, scm: Git, git: Git, matcher: Type[Matcher]
):
    def _gen(s):
        tmp_dir.gen(s, s)
        scm.add_commit(s, message=s)
        return scm.get_rev()

    rev_a = _gen("a")
    rev_b = _gen("b")
    scm.tag("tag")
    rev_c = _gen("c")
    scm.gitpython.git.reset(rev_a, hard=True)
    scm.set_ref("refs/foo/bar", rev_c)

    assert git.list_all_commits() == matcher.unordered(rev_a, rev_b)


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
@pytest.mark.skipif(
    os.name == "nt", reason="Git hooks not supported on Windows"
)
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

    with pytest.raises(MergeConflictError):
        tmp_dir.gen("foo", "baz")
        scm.add_commit("foo", message="baz")
        git.merge(branch, commit=not squash, squash=squash, msg="merge")

    scm.gitpython.git.reset(init_rev, hard=True)
    merge_rev = git.merge(
        branch, commit=not squash, squash=squash, msg="merge"
    )
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
@pytest.mark.parametrize(
    "strategy, expected", [("ours", "baz"), ("theirs", "bar")]
)
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
            os.path.join(
                ".git", "refs", "remotes", "upstream", "baz"
            ): init_rev,
        }
    )

    assert git.resolve_rev(rev) == rev
    assert git.resolve_rev(rev[:7]) == rev
    assert git.resolve_rev("HEAD") == rev
    assert git.resolve_rev("branch") == rev
    assert git.resolve_rev("refs/foo") == rev
    assert git.resolve_rev("bar") == rev
    assert git.resolve_rev("origin/baz") == rev

    with pytest.raises(RevError):
        git.resolve_rev("qux")

    with pytest.raises(RevError):
        git.resolve_rev("baz")


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


@pytest.mark.skip_git_backend("dulwich", "gitpython")
@pytest.mark.skipif(os.name != "nt", reason="Windows only")
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

    assert git.describe(rev_foo, "refs/heads") is None

    scm.checkout("branch", create_new=True)
    assert git.describe(rev_bar, "refs/heads") == "refs/heads/branch"

    tmp_dir.gen({"foo": "foobar"})
    scm.add_commit("foo", message="foobar")
    rev_foobar = scm.get_rev()

    scm.checkout("master")
    assert git.describe(rev_bar, "refs/heads") == "refs/heads/master"
    assert git.describe(rev_foobar, "refs/heads") == "refs/heads/branch"

    scm.tag("tag")
    assert git.describe(rev_bar) == "refs/tags/tag"
