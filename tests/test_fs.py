import os

import pytest
from pytest_test_utils import TmpDir

from scmrepo.git import Git


@pytest.fixture(name="make_fs")
def fixture_make_fs(scm: Git, git: Git):
    def _make_fs(rev=None):
        from scmrepo.fs import GitFileSystem
        from scmrepo.git.objects import GitTrie

        # NOTE: not all git backends have `resolve_rev` implemented,
        # so we are using whichever works.
        resolved = scm.resolve_rev(rev or "HEAD")
        tree = git.get_tree_obj(rev=resolved)
        trie = GitTrie(tree, resolved)
        return GitFileSystem(trie=trie)

    return _make_fs


@pytest.mark.parametrize("raw", [True, False])
def test_open(tmp_dir: TmpDir, scm: Git, make_fs, raw: bool, git_backend: str):
    if not raw and git_backend != "pygit2":
        pytest.skip()

    files = tmp_dir.gen({"foo": "foo", "тест": "проверка", "data": {"lorem": "ipsum"}})
    scm.add_commit(files, message="add")

    fs = make_fs()
    with fs.open("foo", mode="r", encoding="utf-8", raw=raw) as fobj:
        assert fobj.read() == "foo"
    with fs.open("тест", mode="r", encoding="utf-8", raw=raw) as fobj:
        assert fobj.read() == "проверка"
    with pytest.raises(IOError):  # noqa: PT011
        fs.open("not-existing-file", raw=raw)
    with pytest.raises(IOError):  # noqa: PT011
        fs.open("data", raw=raw)


def test_exists(tmp_dir: TmpDir, scm: Git, make_fs):
    scm.commit("init")
    files = tmp_dir.gen({"foo": "foo", "тест": "проверка", "data": {"lorem": "ipsum"}})

    fs = make_fs()

    assert fs.exists("/")
    assert fs.exists(".")
    assert not fs.exists("foo")
    assert not fs.exists("тест")
    assert not fs.exists("data")
    assert not fs.exists("data/lorem")

    scm.add_commit(files, message="add")

    fs = make_fs()
    assert fs.exists("/")
    assert fs.exists(".")
    assert fs.exists("foo")
    assert fs.exists("тест")
    assert fs.exists("data")
    assert fs.exists("data/lorem")
    assert not fs.exists("non-existing-file")


def test_isdir(tmp_dir: TmpDir, scm: Git, make_fs):
    tmp_dir.gen({"foo": "foo", "тест": "проверка", "data": {"lorem": "ipsum"}})
    scm.add_commit(["foo", "data"], message="add")

    fs = make_fs()

    assert fs.isdir("/")
    assert fs.isdir(".")
    assert fs.isdir("data")
    assert not fs.isdir("foo")
    assert not fs.isdir("non-existing-file")


def test_isfile(tmp_dir: TmpDir, scm: Git, make_fs):
    tmp_dir.gen({"foo": "foo", "тест": "проверка", "data": {"lorem": "ipsum"}})
    scm.add_commit(["foo", "data"], message="add")

    fs = make_fs()
    assert not fs.isfile("/")
    assert not fs.isfile(".")
    assert fs.isfile("foo")
    assert not fs.isfile("data")
    assert not fs.isfile("not-existing-file")


def test_walk(tmp_dir: TmpDir, scm: Git, make_fs):
    tmp_dir.gen(
        {
            "foo": "foo",
            "тест": "проверка",
            "data": {"lorem": "ipsum", "subdir": {"sub": "sub"}},
        }
    )
    scm.add_commit("data/subdir", message="add")
    fs = make_fs()

    def convert_to_sets(walk_results):
        return [(root, set(dirs), set(nondirs)) for root, dirs, nondirs in walk_results]

    assert convert_to_sets(fs.walk("/")) == convert_to_sets(
        [
            ("/", ["data"], []),
            ("/data", ["subdir"], []),
            (
                "/data/subdir",
                [],
                ["sub"],
            ),
        ]
    )

    assert convert_to_sets(fs.walk("data/subdir")) == convert_to_sets(
        [
            (
                "data/subdir",
                [],
                ["sub"],
            )
        ]
    )


def test_walk_with_submodules(
    scm: Git,
    remote_git_dir: TmpDir,
    make_fs,
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
    fs = make_fs()
    for _, dnames, fnames in fs.walk(""):
        dirs.extend(dnames)
        files.extend(fnames)

    # currently we don't walk through submodules
    assert not dirs
    assert set(files) == {".gitmodules", "submodule"}


def test_ls(tmp_dir: TmpDir, scm: Git, make_fs):
    files = tmp_dir.gen(
        {
            "foo": "foo",
            "тест": "проверка",
            "data": {"lorem": "ipsum", "subdir": {"sub": "sub"}},
        }
    )
    scm.add_commit(files, message="add")
    fs = make_fs()

    assert fs.ls("/", detail=False) == ["/data", "/foo", "/тест"]
    assert fs.ls("/") == [
        {
            "mode": 16384,
            "name": "/data",
            "sha": "f5d6ac1955c85410b71bb6e35e4c57c54e2ad524",
            "size": 66,
            "type": "directory",
        },
        {
            "mode": 33188,
            "name": "/foo",
            "sha": "19102815663d23f8b75a47e7a01965dcdc96468c",
            "size": 3,
            "type": "file",
        },
        {
            "mode": 33188,
            "name": "/тест",
            "sha": "eeeba1738f4c12844163b89112070c6e57eb764e",
            "size": 16,
            "type": "file",
        },
    ]
