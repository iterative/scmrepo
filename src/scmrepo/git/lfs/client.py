import logging
from contextlib import AbstractContextManager
from functools import wraps
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Iterable, Optional

import aiohttp
from dvc_http import HTTPFileSystem
from dvc_http.retry import ReadOnlyRetryClient
from dvc_objects.executors import batch_coros
from dvc_objects.fs import localfs
from dvc_objects.fs.callbacks import DEFAULT_CALLBACK
from dvc_objects.fs.utils import as_atomic
from fsspec.asyn import sync_wrapper
from funcy import cached_property

from ..credentials import Credential, CredentialNotFoundError
from .exceptions import LFSError
from .pointer import Pointer

if TYPE_CHECKING:
    from dvc_objects.fs.callbacks import Callback

    from .storage import LFSStorage

logger = logging.getLogger(__name__)


class _LFSClient(ReadOnlyRetryClient):
    async def _request(self, *args, **kwargs):
        return await super()._request(*args, **kwargs)  # pylint: disable=no-member


# pylint: disable=abstract-method
class _LFSFileSystem(HTTPFileSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_credentials(self, **config):
        return {}

    async def get_client(self, **kwargs):
        from aiohttp_retry import ExponentialRetry
        from dvc_http import make_context

        kwargs["retry_options"] = ExponentialRetry(
            attempts=self.SESSION_RETRIES,
            factor=self.SESSION_BACKOFF_FACTOR,
            max_timeout=self.REQUEST_TIMEOUT,
            exceptions={aiohttp.ClientError},
        )

        # The default total timeout for an aiohttp request is 300 seconds
        # which is too low for DVC's interactions when dealing with large
        # data blobs. We remove the total timeout, and only limit the time
        # that is spent when connecting to the remote server and waiting
        # for new data portions.
        connect_timeout = kwargs.pop("connect_timeout")
        kwargs["timeout"] = aiohttp.ClientTimeout(
            total=None,
            connect=connect_timeout,
            sock_connect=connect_timeout,
            sock_read=kwargs.pop("read_timeout"),
        )

        kwargs["connector"] = aiohttp.TCPConnector(
            # Force cleanup of closed SSL transports.
            # See https://github.com/iterative/dvc/issues/7414
            enable_cleanup_closed=True,
            ssl=make_context(kwargs.pop("ssl_verify", None)),
        )

        return ReadOnlyRetryClient(**kwargs)


def _authed(f: Callable[..., Awaitable]):
    """Set credentials and retry the given coroutine if needed."""

    # pylint: disable=protected-access
    @wraps(f)  # type: ignore[arg-type]
    async def wrapper(self, *args, **kwargs):
        try:
            return await f(self, *args, **kwargs)
        except aiohttp.ClientResponseError as exc:
            if exc.status != 401:
                raise
            session = await self._set_session()
            if session.auth:
                raise
            auth = self._get_auth()
            if auth is None:
                raise
            self._session._auth = auth
        return await f(self, *args, **kwargs)

    return wrapper


class LFSClient(AbstractContextManager):
    """Naive read-only LFS HTTP client."""

    JSON_CONTENT_TYPE = "application/vnd.git-lfs+json"

    def __init__(
        self,
        url: str,
        git_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            url: LFS server URL.
        """
        self.url = url
        self.git_url = git_url
        self.headers: Dict[str, str] = headers or {}

    def __exit__(self, *args, **kwargs):
        self.close()

    @cached_property
    def fs(self) -> "_LFSFileSystem":
        return _LFSFileSystem()

    @property
    def httpfs(self) -> "HTTPFileSystem":
        return self.fs.fs

    @property
    def loop(self):
        return self.httpfs.loop

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

    async def _set_session(self) -> aiohttp.ClientSession:
        return await self.fs.fs.set_session()

    @_authed
    async def _batch_request(
        self,
        objects: Iterable[Pointer],
        upload: bool = False,
        ref: Optional[str] = None,
        hash_algo: str = "sha256",
    ) -> Dict[str, Any]:
        """Send LFS API /objects/batch request."""
        url = f"{self.url}/objects/batch"
        body: Dict[str, Any] = {
            "operation": "upload" if upload else "download",
            "transfers": ["basic"],
            "objects": [{"oid": obj.oid, "size": obj.size} for obj in objects],
            "hash_algo": hash_algo,
        }
        if ref:
            body["ref"] = [{"name": ref}]
        session = await self._set_session()
        headers = dict(self.headers)
        headers["Content-Type"] = self.JSON_CONTENT_TYPE
        async with session.post(
            url,
            headers=headers,
            json=body,
        ) as resp:
            data = await resp.json()
        return data

    @_authed
    async def _download(
        self,
        storage: "LFSStorage",
        objects: Iterable[Pointer],
        callback: "Callback" = DEFAULT_CALLBACK,
        **kwargs,
    ):
        async def _get_one(from_path: str, to_path: str, **kwargs):
            with as_atomic(localfs, to_path, create_parents=True) as tmp_file:
                with callback.branch(from_path, tmp_file, kwargs):
                    await self.httpfs._get_file(from_path, tmp_file, **kwargs)  # pylint: disable=protected-access
                    callback.relative_update()

        resp_data = await self._batch_request(objects, **kwargs)
        if resp_data.get("transfer") != "basic":
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
        for result in await batch_coros(
            coros, batch_size=self.fs.jobs, return_exceptions=True
        ):
            if isinstance(result, BaseException):
                raise result

    download = sync_wrapper(_download)
