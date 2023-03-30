import os
from typing import Any, Dict, Optional

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
        self._store_credentials: Optional["Credential"] = None

    def _http_request(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Any = None,
    ):
        try:
            result = super()._http_request(url, headers=headers, data=data)
        except HTTPUnauthorized:
            auth_header = self._get_auth()
            if not auth_header:
                raise
            if headers:
                headers.update(auth_header)
            else:
                headers = auth_header
            result = super()._http_request(url, headers=headers, data=data)
        if self._store_credentials is not None:
            self._store_credentials.approve()
        return result

    def _get_auth(self) -> Dict[str, str]:
        from getpass import getpass

        from urllib3.util import make_headers

        try:
            creds = Credential(username=self._username, url=self._base_url).fill()
            self._store_credentials = creds
            return make_headers(basic_auth=f"{creds.username}:{creds.password}")
        except CredentialNotFoundError:
            pass

        if os.environ.get("GIT_TERMINAL_PROMPT") == "0":
            return {}

        try:
            if self._username:
                username = self._username
            else:
                username = input(f"Username for '{self._base_url}': ")
            if self._password:
                password = self._password
            else:
                password = getpass(f"Password for '{self._base_url}': ")
            return make_headers(basic_auth=f"{username}:{password}")
        except KeyboardInterrupt:
            return {}
