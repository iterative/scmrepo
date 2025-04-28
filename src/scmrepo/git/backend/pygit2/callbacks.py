from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Optional, Union

from pygit2 import RemoteCallbacks as _RemoteCallbacks

from scmrepo.git.backend.base import SyncStatus
from scmrepo.git.credentials import Credential, CredentialNotFoundError
from scmrepo.progress import GitProgressReporter

if TYPE_CHECKING:
    from pygit2 import Oid
    from pygit2.credentials import Keypair, Username, UserPass
    from pygit2.enums import CredentialType

    from scmrepo.progress import GitProgressEvent


_Pygit2Credential = Union["Keypair", "Username", "UserPass"]


class RemoteCallbacks(_RemoteCallbacks, AbstractContextManager):
    def __init__(
        self,
        *args,
        progress: Optional[Callable[["GitProgressEvent"], None]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress = GitProgressReporter(progress) if progress else None
        self._store_credentials: Optional[Credential] = None
        self._tried_credentials = False
        self.result: dict[str, SyncStatus] = {}

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        if exc_type is None:
            self._approve_credentials()

    def sideband_progress(self, string: str):
        if self.progress is not None:
            self.progress(string)

    def credentials(
        self,
        url: str,
        username_from_url: Optional[str],
        allowed_types: "CredentialType",
    ) -> "_Pygit2Credential":
        from pygit2 import GitError, Passthrough
        from pygit2.credentials import UserPass
        from pygit2.enums import CredentialType

        if self._tried_credentials:
            raise GitError(f"authentication failed for '{url}'")
        self._tried_credentials = True

        if allowed_types & CredentialType.USERPASS_PLAINTEXT:
            try:
                if self._store_credentials:
                    creds = self._store_credentials
                else:
                    creds = Credential(username=username_from_url, url=url).fill()
                    self._store_credentials = creds
                assert creds.username is not None
                assert creds.password is not None
                return UserPass(creds.username, creds.password)
            except CredentialNotFoundError:
                pass
        raise Passthrough

    def _approve_credentials(self):
        if self._store_credentials:
            self._store_credentials.approve()

    def update_tips(self, refname: str, old: "Oid", new: "Oid"):
        if old == new:
            self.result[refname] = SyncStatus.UP_TO_DATE
        else:
            self.result[refname] = SyncStatus.SUCCESS
