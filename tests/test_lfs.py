# pylint: disable=redefined-outer-name
import io

import pytest
from pytest_mock import MockerFixture
from pytest_test_utils import TempDirFactory, TmpDir

from scmrepo.git import Git
from scmrepo.git.lfs import LFSStorage, Pointer, smudge

FOO_OID = "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
FOO_POINTER = (
    f"version https://git-lfs.github.com/spec/v1\noid sha256:{FOO_OID}\nsize 3\n"
).encode()


@pytest.fixture
def storage(tmp_dir_factory: TempDirFactory) -> LFSStorage:
    storage_path = tmp_dir_factory.mktemp("lfs")
    return LFSStorage(storage_path)


@pytest.fixture
def lfs(tmp_dir: TmpDir, scm: Git) -> None:
    tmp_dir.gen(".gitattributes", "*.lfs filter=lfs diff=lfs merge=lfs -text")
    scm.add([".gitattributes"])
    scm.commit("init lfs attributes")


@pytest.fixture
def lfs_objects(tmp_dir: TmpDir) -> TmpDir:
    objects = tmp_dir / ".git" / "lfs" / "objects"
    objects.mkdir(parents=True)
    return objects


def test_pointer_build(tmp_dir: TmpDir):
    tmp_dir.gen("foo", "foo")
    with open(tmp_dir / "foo", "rb") as fobj:
        pointer = Pointer.build(fobj)

    assert pointer.dump() == FOO_POINTER.decode("utf-8")


def test_pointer_load(tmp_dir: TmpDir):
    tmp_dir.gen("foo.lfs", FOO_POINTER)
    with open(tmp_dir / "foo.lfs", "rb") as fobj:
        pointer = Pointer.load(fobj)
    assert pointer.oid == FOO_OID
    assert pointer.size == 3


def test_smudge(tmp_dir: TmpDir, storage: LFSStorage, mocker: MockerFixture):
    tmp_dir.gen("foo.lfs", FOO_POINTER)
    with open(tmp_dir / "foo.lfs", "rb") as fobj:
        assert smudge(storage, fobj).read() == FOO_POINTER

    mocker.patch.object(storage, "open", return_value=io.BytesIO(b"foo"))
    with open(tmp_dir / "foo.lfs", "rb") as fobj:
        assert smudge(storage, fobj).read() == b"foo"


@pytest.mark.usefixtures("lfs")
def test_lfs(tmp_dir: TmpDir, scm: Git, lfs_objects: TmpDir):
    # NOTE: scmrepo does not currently support LFS clean (writes), this writes
    # the pointer to the git odb (simulating an actual LFS clean)
    tmp_dir.gen("foo.lfs", FOO_POINTER)
    scm.add(["foo.lfs"])
    scm.commit("add foo")
    lfs_objects.gen({FOO_OID[0:2]: {FOO_OID[2:4]: {FOO_OID: "foo"}}})

    fs = scm.get_fs("HEAD")
    with fs.open("foo.lfs", "rb", raw=True) as fobj:
        assert fobj.read() == FOO_POINTER
    with fs.open("foo.lfs", "rb", raw=False) as fobj:
        assert fobj.read() == b"foo"
