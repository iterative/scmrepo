import logging
import os
import shutil
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Optional

import aiohttp
from aiohttp_retry import ExponentialRetry, RetryClient
from fsspec.asyn import _run_coros_in_chunks, sync_wrapper
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.implementations.http import HTTPFileSystem
from funcy import cached_property

from scmrepo.git.credentials import Credential, CredentialNotFoundError

from .exceptions import LFSError
from .pointer import Pointer

if TYPE_CHECKING:
    from fsspec.callbacks import Callback

    from .storage import LFSStorage

logger = logging.getLogger(__name__)


class LFSClient(AbstractContextManager):
    """Naive read-only LFS HTTP client."""

    JSON_CONTENT_TYPE = "application/vnd.git-lfs+json"

    _REQUEST_TIMEOUT = 60
    _SESSION_RETRIES = 5
    _SESSION_BACKOFF_FACTOR = 0.1

    def __init__(
        self,
        url: str,
        git_url: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
    ):
        """
        Args:
            url: LFS server URL.
        """
        self.url = url
        self.git_url = git_url
        self.headers: dict[str, str] = headers or {}

    def __exit__(self, *args, **kwargs):
        self.close()

    @cached_property
    def _fs(self) -> HTTPFileSystem:
        async def get_client(**kwargs):
            return RetryClient(
                connector=aiohttp.TCPConnector(
                    # Force cleanup of closed SSL transports.
                    # See https://github.com/iterative/dvc/issues/7414
                    enable_cleanup_closed=True,
                ),
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    connect=self._REQUEST_TIMEOUT,
                    sock_connect=self._REQUEST_TIMEOUT,
                    sock_read=self._REQUEST_TIMEOUT,
                ),
                retry_options=ExponentialRetry(
                    attempts=self._SESSION_RETRIES,
                    factor=self._SESSION_BACKOFF_FACTOR,
                    max_timeout=self._REQUEST_TIMEOUT,
                    exceptions={aiohttp.ClientError},
                ),
                **kwargs,
            )

        return HTTPFileSystem(get_client=get_client)

    @property
    def loop(self):
        return self._fs.loop

    @classmethod
    def from_git_url(cls, git_url: str) -> "LFSClient":
        if git_url.endswith(".git"):
            url = f"{git_url}/info/lfs"
        else:
            url = f"{git_url}.git/info/lfs"
        return cls(url, git_url=git_url)

    def close(self):
        pass

    def _get_auth(self) -> Optional[aiohttp.BasicAuth]:
        try:
            creds = Credential(url=self.git_url).fill()
            if creds.username and creds.password:
                return aiohttp.BasicAuth(creds.username, creds.password)
        except CredentialNotFoundError:
            pass
        return None

    async def _batch_request(
        self,
        objects: Iterable[Pointer],
        upload: bool = False,
        ref: Optional[str] = None,
        hash_algo: str = "sha256",
    ) -> dict[str, Any]:
        """Send LFS API /objects/batch request."""
        url = f"{self.url}/objects/batch"
        body: dict[str, Any] = {
            "operation": "upload" if upload else "download",
            "transfers": ["basic"],
            "objects": [{"oid": obj.oid, "size": obj.size} for obj in objects],
            "hash_algo": hash_algo,
        }
        if ref:
            body["ref"] = [{"name": ref}]
        session = await self._fs.set_session()
        headers = dict(self.headers)
        headers["Accept"] = self.JSON_CONTENT_TYPE
        headers["Content-Type"] = self.JSON_CONTENT_TYPE
        try:
            async with session.post(
                url,
                headers=headers,
                json=body,
                raise_for_status=True,
            ) as resp:
                data = await resp.json()
        except aiohttp.ClientResponseError as exc:
            if exc.status != 401:
                raise
            auth = self._get_auth()
            if auth is None:
                raise
            async with session.post(
                url,
                auth=auth,
                headers=headers,
                json=body,
                raise_for_status=True,
            ) as resp:
                data = await resp.json()
        return data

    async def _download(
        self,
        storage: "LFSStorage",
        objects: Iterable[Pointer],
        callback: "Callback" = DEFAULT_CALLBACK,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        async def _get_one(from_path: str, to_path: str, **kwargs):
            with _as_atomic(to_path, create_parents=True) as tmp_file:
                with callback.branched(from_path, tmp_file) as child:
                    await self._fs._get_file(
                        from_path, tmp_file, callback=child, **kwargs
                    )
                    callback.relative_update()

        resp_data = await self._batch_request(objects, **kwargs)
        if resp_data.get("transfer", "basic") != "basic":
            raise LFSError("Unsupported LFS transfer type")
        coros = []
        for data in resp_data.get("objects", []):
            obj = Pointer(data["oid"], data["size"])
            download = data.get("actions", {}).get("download", {})
            url = download.get("href")
            if not url:
                logger.debug("No download URL for LFS object '%s'", obj)
                continue
            headers = download.get("header", {})
            to_path = storage.oid_to_path(obj.oid)
            coros.append(_get_one(url, to_path, headers=headers))
        for result in await _run_coros_in_chunks(
            coros, batch_size=batch_size, return_exceptions=True
        ):
            if isinstance(result, BaseException):
                raise result

    download = sync_wrapper(_download)


@contextmanager
def _as_atomic(to_info: str, create_parents: bool = False) -> Iterator[str]:
    parent = os.path.dirname(to_info)
    if create_parents:
        os.makedirs(parent, exist_ok=True)

    tmp_file = NamedTemporaryFile(dir=parent, delete=False)
    tmp_file.close()
    try:
        yield tmp_file.name
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(tmp_file.name)
        raise
    else:
        shutil.move(tmp_file.name, to_info)
