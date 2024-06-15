from collections.abc import Iterator
from typing import Optional, Union

from dulwich.client import HTTPUnauthorized, Urllib3HttpGitClient

from scmrepo.git.credentials import Credential, CredentialNotFoundError


class GitCredentialsHTTPClient(Urllib3HttpGitClient):  # pylint: disable=abstract-method
    def __init__(
        self,
        base_url,
        username=None,
        password=None,
        config=None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            username=username,
            password=password,
            config=config,
            **kwargs,
        )
        self._store_credentials: Optional[Credential] = None

    def _http_request(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        data: Optional[Union[bytes, Iterator[bytes]]] = None,
    ):
        cached_chunks: list[bytes] = []

        def _cached_data() -> Iterator[bytes]:
            assert data is not None
            if isinstance(data, bytes):
                yield data
                return

            if cached_chunks:
                yield from cached_chunks
                return

            for chunk in data:
                cached_chunks.append(chunk)
                yield chunk

        try:
            result = super()._http_request(
                url, headers=headers, data=None if data is None else _cached_data()
            )
        except HTTPUnauthorized:
            auth_header = self._get_auth()
            if not auth_header:
                raise
            if headers:
                headers.update(auth_header)
            else:
                headers = auth_header
            result = super()._http_request(
                url, headers=headers, data=None if data is None else _cached_data()
            )
        if self._store_credentials is not None:
            self._store_credentials.approve()
        return result

    def _get_auth(self) -> dict[str, str]:
        from urllib3.util import make_headers

        try:
            base_url = self._base_url.rstrip("/")
            creds = Credential(username=self._username, url=base_url).fill()
            self._store_credentials = creds
            return make_headers(basic_auth=f"{creds.username}:{creds.password}")
        except CredentialNotFoundError:
            return {}
