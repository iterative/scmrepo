from typing import Optional

from dulwich.client import Urllib3HttpGitClient

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
        if not username:
            import base64

            try:
                creds = Credential(url=base_url).fill()
            except CredentialNotFoundError:
                return
            encoded = base64.b64encode(
                f"{creds.username}:{creds.password}".encode()
            ).decode("ascii")
            basic_auth = {"authorization": f"Basic {encoded}"}
            self.pool_manager.headers.update(basic_auth)
            self._store_credentials = creds

    def _http_request(self, *args, **kwargs):
        result = super()._http_request(*args, **kwargs)
        if self._store_credentials is not None:
            self._store_credentials.approve()
        return result
