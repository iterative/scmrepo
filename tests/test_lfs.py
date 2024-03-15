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
def lfs(tmp_dir: TmpDir, scm: Git) -> None:  # noqa: PT004
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


@pytest.mark.parametrize(
    "rate_limit_header",
    [
        lambda: {"Retry-After": "1"},
        lambda: {"RateLimit-Reset": f"{int(time()) + 1}"},
        lambda: {"X-RateLimit-Reset": f"{int(time()) + 1}"},
    ],
)
def test_rate_limit_retry(
    storage: LFSStorage, rate_limit_header: Callable[[], dict[str, str]]
):
    client = LFSClient.from_git_url("http://git.example.com/namespace/project.git")
    lfs_batch_url = f"{client.url}/objects/batch"
    lfs_object_url = f"http://git-lfs.example.com/{FOO_OID}"
    recorder = CallbackResultRecorder()

    with aioresponses() as m:
        m.post(
            lfs_batch_url,
            callback=recorder.record(
                CallbackResult(
                    status=HTTPStatus.TOO_MANY_REQUESTS,
                    headers=rate_limit_header(),
                    reason="Too many requests",
                )
            ),
        )
        m.post(
            lfs_batch_url,
            callback=recorder.record(
                CallbackResult(
                    status=HTTPStatus.OK,
                    headers={"Content-Type": "application/vnd.git-lfs+json"},
                    payload={
                        "transfer": "basic",
                        "objects": [
                            {
                                "oid": FOO_OID,
                                "size": FOO_SIZE,
                                "authenticated": True,
                                "actions": {
                                    "download": {
                                        "href": lfs_object_url,
                                    }
                                },
                            }
                        ],
                        "hash_algo": "sha256",
                    },
                )
            ),
        )
        m.get(
            lfs_object_url,
            callback=recorder.record(
                CallbackResult(
                    status=HTTPStatus.TOO_MANY_REQUESTS,
                    headers=rate_limit_header(),
                    reason="Too many requests",
                )
            ),
        )
        m.get(
            lfs_object_url,
            callback=recorder.record(
                CallbackResult(
                    status=HTTPStatus.OK,
                    body="lfs data",
                )
            ),
        )

        client.download(storage, [Pointer(oid=FOO_OID, size=FOO_SIZE)])

        results = recorder[lfs_batch_url]
        assert [r.status for r in results] == [429, 200]

        results = recorder[lfs_object_url]
        assert [r.status for r in results] == [429, 200]


@pytest.mark.parametrize(
    "rate_limit_header",
    [
        lambda: {"Retry-After": "1"},
        lambda: {"RateLimit-Reset": f"{int(time()) + 1}"},
        lambda: {"X-RateLimit-Reset": f"{int(time()) + 1}"},
    ],
)
def test_rate_limit_max_retries_batch(
    storage: LFSStorage, rate_limit_header: Callable[[], dict[str, str]]
):
    client = LFSClient.from_git_url("http://git.example.com/namespace/project.git")
    recorder = CallbackResultRecorder()

    with aioresponses() as m:
        m.post(
            f"{client.url}/objects/batch",
            callback=recorder.record(
                CallbackResult(
                    status=HTTPStatus.TOO_MANY_REQUESTS,
                    headers=rate_limit_header(),
                    reason="Too many requests",
                )
            ),
            repeat=True,
        )

        with pytest.raises(ClientResponseError, match="Too many requests"):
            client.download(storage, [Pointer(oid=FOO_OID, size=FOO_SIZE)])

        results = recorder[f"{client.url}/objects/batch"]
        assert [r.status for r in results] == [429] * 5


@pytest.mark.parametrize(
    "rate_limit_header",
    [
        lambda: {"Retry-After": "1"},
        lambda: {"RateLimit-Reset": f"{int(time()) + 1}"},
        lambda: {"X-RateLimit-Reset": f"{int(time()) + 1}"},
    ],
)
def test_rate_limit_max_retries_objects(
    storage: LFSStorage, rate_limit_header: Callable[[], dict[str, str]]
):
    client = LFSClient.from_git_url("http://git.example.com/namespace/project.git")
    lfs_batch_url = f"{client.url}/objects/batch"
    lfs_object_url = f"http://git-lfs.example.com/{FOO_OID}"
    recorder = CallbackResultRecorder()

    with aioresponses() as m:
        m.post(
            lfs_batch_url,
            callback=recorder.record(
                CallbackResult(
                    status=HTTPStatus.OK,
                    headers={"Content-Type": "application/vnd.git-lfs+json"},
                    payload={
                        "transfer": "basic",
                        "objects": [
                            {
                                "oid": FOO_OID,
                                "size": FOO_SIZE,
                                "authenticated": True,
                                "actions": {
                                    "download": {
                                        "href": lfs_object_url,
                                    }
                                },
                            }
                        ],
                        "hash_algo": "sha256",
                    },
                )
            ),
        )
        m.get(
            lfs_object_url,
            callback=recorder.record(
                CallbackResult(
                    status=HTTPStatus.TOO_MANY_REQUESTS,
                    headers=rate_limit_header(),
                    reason="Too many requests",
                ),
            ),
            repeat=True,
        )

        with pytest.raises(ClientResponseError, match="Too many requests"):
            client.download(storage, [Pointer(oid=FOO_OID, size=FOO_SIZE)])

        results = recorder[lfs_batch_url]
        assert [r.status for r in results] == [200]

        results = recorder[lfs_object_url]
        assert [r.status for r in results] == [429] * 5
