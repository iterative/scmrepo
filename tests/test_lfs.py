# pylint: disable=redefined-outer-name
import io
from collections import defaultdict
from collections.abc import Sequence
from http import HTTPStatus
from time import time
from typing import Callable

import pytest
from aiohttp import ClientResponseError
from aioresponses import CallbackResult, aioresponses
from pytest_mock import MockerFixture
from pytest_test_utils import TempDirFactory, TmpDir
from yarl import URL

from scmrepo.git import Git
from scmrepo.git.lfs import LFSClient, LFSStorage, Pointer, smudge

FOO_OID = "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
FOO_SIZE = 3
FOO_POINTER = (
    f"version https://git-lfs.github.com/spec/v1\n"
    f"oid sha256:{FOO_OID}\n"
    f"size {FOO_SIZE}\n"
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


class CallbackResultRecorder:
    def __init__(self) -> None:
        self._results: dict[str, list[CallbackResult]] = defaultdict(list)

    def record(self, result: CallbackResult) -> Callable[..., CallbackResult]:
        def _callback(url: URL, **_) -> CallbackResult:
            self._results[str(url)].append(result)
            return result

        return _callback

    def __getitem__(self, url: str) -> Sequence[CallbackResult]:
        return self._results[url]


class LFSServerMock:
    def __init__(
        self,
        mocker: aioresponses,
        recorder: CallbackResultRecorder,
        batch_url: str,
        objects_url: str,
    ) -> None:
        self._mocker = mocker
        self._recorder = recorder
        self.batch_url = batch_url
        self.objects_url = objects_url

    def get_object_url(self, oid: str) -> str:
        return f"{self.objects_url}/{oid}"

    def mock_batch_200(self, pointer: Pointer) -> None:
        self._mocker.post(
            self.batch_url,
            callback=self._recorder.record(
                CallbackResult(
                    status=HTTPStatus.OK,
                    headers={"Content-Type": "application/vnd.git-lfs+json"},
                    payload={
                        "transfer": "basic",
                        "objects": [
                            {
                                "oid": pointer.oid,
                                "size": pointer.size,
                                "authenticated": True,
                                "actions": {
                                    "download": {
                                        "href": self.get_object_url(pointer.oid),
                                    }
                                },
                            }
                        ],
                        "hash_algo": "sha256",
                    },
                )
            ),
        )

    def mock_batch_429(
        self, header: str, value: Callable[[], str], *, repeat: bool = False
    ) -> None:
        self._mocker.post(
            self.batch_url,
            callback=self._recorder.record(
                CallbackResult(
                    status=HTTPStatus.TOO_MANY_REQUESTS,
                    headers={header: value()},
                    reason="Too many requests",
                )
            ),
            repeat=repeat,
        )

    def mock_object_200(self, oid: str) -> None:
        self._mocker.get(
            self.get_object_url(oid),
            callback=self._recorder.record(
                CallbackResult(
                    status=HTTPStatus.OK,
                    body=f"object {oid} data",
                )
            ),
        )

    def mock_object_429(
        self,
        oid: str,
        header: str,
        value: Callable[[], str],
        *,
        repeat: bool = False,
    ) -> None:
        self._mocker.get(
            self.get_object_url(oid),
            callback=self._recorder.record(
                CallbackResult(
                    status=HTTPStatus.TOO_MANY_REQUESTS,
                    headers={header: value()},
                    reason="Too many requests",
                )
            ),
            repeat=repeat,
        )


@pytest.mark.parametrize(
    "rate_limit_header, rate_limit_value",
    [
        ("Retry-After", lambda: "1"),
        ("RateLimit-Reset", lambda: f"{int(time()) + 1}"),
        ("X-RateLimit-Reset", lambda: f"{int(time()) + 1}"),
    ],
)
def test_rate_limit_retry(
    storage: LFSStorage, rate_limit_header: str, rate_limit_value: Callable[[], str]
):
    client = LFSClient.from_git_url("http://git.example.com/namespace/project.git")
    recorder = CallbackResultRecorder()

    with aioresponses() as m:
        lfs_server = LFSServerMock(
            m, recorder, f"{client.url}/objects/batch", "http://git-lfs.example.com"
        )
        lfs_server.mock_batch_429(rate_limit_header, rate_limit_value)
        lfs_server.mock_batch_200(Pointer(FOO_OID, FOO_SIZE))
        lfs_server.mock_object_429(FOO_OID, rate_limit_header, rate_limit_value)
        lfs_server.mock_object_200(FOO_OID)

        client.download(storage, [Pointer(oid=FOO_OID, size=FOO_SIZE)])

        results = recorder[lfs_server.batch_url]
        assert [r.status for r in results] == [429, 200]

        results = recorder[lfs_server.get_object_url(FOO_OID)]
        assert [r.status for r in results] == [429, 200]


@pytest.mark.parametrize(
    "rate_limit_header, rate_limit_value",
    [
        ("Retry-After", lambda: "1"),
        ("RateLimit-Reset", lambda: f"{int(time()) + 1}"),
        ("X-RateLimit-Reset", lambda: f"{int(time()) + 1}"),
    ],
)
def test_rate_limit_max_retries_batch(
    storage: LFSStorage, rate_limit_header: str, rate_limit_value: Callable[[], str]
):
    client = LFSClient.from_git_url("http://git.example.com/namespace/project.git")
    recorder = CallbackResultRecorder()

    with aioresponses() as m:
        lfs_server = LFSServerMock(
            m, recorder, f"{client.url}/objects/batch", "http://git-lfs.example.com"
        )
        lfs_server.mock_batch_429(rate_limit_header, rate_limit_value, repeat=True)

        with pytest.raises(ClientResponseError, match="Too many requests"):
            client.download(storage, [Pointer(oid=FOO_OID, size=FOO_SIZE)])

        results = recorder[lfs_server.batch_url]
        assert [r.status for r in results] == [429] * 5


@pytest.mark.parametrize(
    "rate_limit_header, rate_limit_value",
    [
        ("Retry-After", lambda: "1"),
        ("RateLimit-Reset", lambda: f"{int(time()) + 1}"),
        ("X-RateLimit-Reset", lambda: f"{int(time()) + 1}"),
    ],
)
def test_rate_limit_max_retries_objects(
    storage: LFSStorage, rate_limit_header: str, rate_limit_value: Callable[[], str]
):
    client = LFSClient.from_git_url("http://git.example.com/namespace/project.git")
    recorder = CallbackResultRecorder()

    with aioresponses() as m:
        lfs_server = LFSServerMock(
            m, recorder, f"{client.url}/objects/batch", "http://git-lfs.example.com"
        )
        lfs_server.mock_batch_200(Pointer(FOO_OID, FOO_SIZE))
        lfs_server.mock_object_429(
            FOO_OID, rate_limit_header, rate_limit_value, repeat=True
        )

        with pytest.raises(ClientResponseError, match="Too many requests"):
            client.download(storage, [Pointer(oid=FOO_OID, size=FOO_SIZE)])

        results = recorder[lfs_server.batch_url]
        assert [r.status for r in results] == [200]

        results = recorder[lfs_server.get_object_url(FOO_OID)]
        assert [r.status for r in results] == [429] * 5
