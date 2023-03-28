from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Callable, Optional, Union

from pygit2 import RemoteCallbacks as _RemoteCallbacks

from scmrepo.git.credentials import Credential, CredentialNotFoundError
from scmrepo.progress import GitProgressReporter

if TYPE_CHECKING:
    from pygit2.credentials import Keypair, Username, UserPass

    from scmrepo.progress import GitProgressEvent


_Pygit2Credential = Union["Keypair", "Username", "UserPass"]


class RemoteCallbacks(_RemoteCallbacks, AbstractContextManager):
    def __init__(
        self,
        *args,
        progress: Optional[Callable[["GitProgressEvent"], None]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.progress = GitProgressReporter(progress) if progress else None
        self._store_credentials: Optional["Credential"] = None

    def __exit__(self, *args, **kwargs):
        self._approve_credentials()

    def sideband_progress(self, string: str):
        if self.progress is not None:
            self.progress(string)

    def credentials(
        self, url: str, username_from_url: Optional[str], allowed_types: int
    ) -> "_Pygit2Credential":
        from pygit2 import Passthrough
        from pygit2.credentials import GIT_CREDENTIAL_USERPASS_PLAINTEXT, UserPass

        if allowed_types & GIT_CREDENTIAL_USERPASS_PLAINTEXT:
            try:
                creds = Credential(username=username_from_url, url=url).fill()
                self._store_credentials = creds
                return UserPass(creds.username, creds.password)
            except CredentialNotFoundError:
                pass
        raise Passthrough

    def _approve_credentials(self):
        if self._store_credentials:
            self._store_credentials.approve()
