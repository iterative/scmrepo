import os

import pytest
from pytest_test_utils import TmpDir

from scmrepo.git import Git


def test_open(tmp_dir: TmpDir, scm: Git):
    files = tmp_dir.gen(
        {"foo": "foo", "тест": "проверка", "data": {"lorem": "ipsum"}}
    )
    scm.add_commit(files, message="add")

    fs = scm.get_fs("master")
    with fs.open("foo", mode="r", encoding="utf-8") as fobj:
        assert fobj.read() == "foo"
    with fs.open("тест", mode="r", encoding="utf-8") as fobj:
        assert fobj.read() == "проверка"
    with pytest.raises(IOError):
        fs.open("not-existing-file")
    with pytest.raises(IOError):
        fs.open("data")


def test_exists(tmp_dir: TmpDir, scm: Git):
    scm.commit("init")
    files = tmp_dir.gen(
        {"foo": "foo", "тест": "проверка", "data": {"lorem": "ipsum"}}
    )

    fs = scm.get_fs("master")

    assert not fs.exists("foo")
    assert not fs.exists("тест")
    assert not fs.exists("data")
    assert not fs.exists(os.path.join("data", "lorem"))

    scm.add_commit(files, message="add")

    fs = scm.get_fs("master")
    assert fs.exists("foo")
    assert fs.exists("тест")
    assert fs.exists("data")
    assert fs.exists(os.path.join("data", "lorem"))
    assert not fs.exists("non-existing-file")


def test_isdir(tmp_dir: TmpDir, scm: Git):
    tmp_dir.gen({"foo": "foo", "тест": "проверка", "data": {"lorem": "ipsum"}})
    scm.add_commit(["foo", "data"], message="add")

    fs = scm.get_fs("master")

    assert fs.isdir("data")
    assert not fs.isdir("foo")
    assert not fs.isdir("non-existing-file")


def test_isfile(tmp_dir: TmpDir, scm: Git):
    tmp_dir.gen({"foo": "foo", "тест": "проверка", "data": {"lorem": "ipsum"}})
    scm.add_commit(["foo", "data"], message="add")

    fs = scm.get_fs("master")
    assert fs.isfile("foo")
    assert not fs.isfile("data")
    assert not fs.isfile("not-existing-file")


def test_walk(tmp_dir: TmpDir, scm: Git):
    tmp_dir.gen(
        {
            "foo": "foo",
            "тест": "проверка",
            "data": {"lorem": "ipsum", "subdir": {"sub": "sub"}},
        }
    )
    scm.add_commit(os.path.join("data", "subdir"), message="add")
    fs = scm.get_fs("master")

    def convert_to_sets(walk_results):
        return [
            (root, set(dirs), set(nondirs))
            for root, dirs, nondirs in walk_results
        ]

    assert convert_to_sets(fs.walk(".")) == convert_to_sets(
        [
            (scm.root_dir, ["data"], []),
            (os.path.join(scm.root_dir, "data"), ["subdir"], []),
            (
                os.path.join(scm.root_dir, "data", "subdir"),
                [],
                ["sub"],
            ),
        ]
    )

    assert convert_to_sets(
        fs.walk(os.path.join("data", "subdir"))
    ) == convert_to_sets(
        [
            (
                os.path.join(scm.root_dir, "data", "subdir"),
                [],
                ["sub"],
            )
        ]
    )


def test_ls(tmp_dir: TmpDir, scm: Git):
    files = tmp_dir.gen(
        {
            "foo": "foo",
            "тест": "проверка",
            "data": {"lorem": "ipsum", "subdir": {"sub": "sub"}},
        }
    )
    scm.add_commit(files, message="add")
    fs = scm.get_fs("master")

    assert fs.ls(".", detail=False) == ["foo", "тест", "data"]
    assert fs.ls(".") == {
        "data": {
            "mode": 16384,
            "name": str(tmp_dir / "data"),
            "sha": "f5d6ac1955c85410b71bb6e35e4c57c54e2ad524",
            "size": 66,
            "type": "directory",
        },
        "foo": {
            "mode": 33188,
            "name": str(tmp_dir / "foo"),
            "sha": "19102815663d23f8b75a47e7a01965dcdc96468c",
            "size": 3,
            "type": "file",
        },
        "тест": {
            "mode": 33188,
            "name": str(tmp_dir / "тест"),
            "sha": "eeeba1738f4c12844163b89112070c6e57eb764e",
            "size": 16,
            "type": "file",
        },
    }
