# Temporarily added while waiting for upstream PR to be merged.
# See https://github.com/jelmer/dulwich/pull/976

from dulwich.client import Urllib3HttpGitClient
from dulwich.config import StackedConfig

from .credentials import CredentialNotFoundError, get_credentials_from_helper


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

        if not username:
            try:
                helper_username, helper_password = get_credentials_from_helper(
                    base_url, config or StackedConfig.default()
                )
            except CredentialNotFoundError:
                pass
            else:
                credentials = helper_username + b":" + helper_password
                import base64

                encoded = base64.b64encode(credentials).decode("ascii")
                basic_auth = {"authorization": f"Basic {encoded}"}
                self.pool_manager.headers.update(basic_auth)
