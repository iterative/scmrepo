"""asyncssh SSH vendor for Dulwich."""

import asyncio
import os
from collections.abc import Coroutine, Iterator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    cast,
)

from asyncssh import SSHClient
from dulwich.client import SSHVendor

from scmrepo.asyn import BaseAsyncObject, sync_wrapper
from scmrepo.exceptions import AuthError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from asyncssh.auth import KbdIntPrompts, KbdIntResponse
    from asyncssh.config import ConfigPaths, FilePath
    from asyncssh.connection import SSHClientConnection
    from asyncssh.misc import MaybeAwait
    from asyncssh.process import SSHClientProcess
    from asyncssh.public_key import KeyPairListArg, SSHKey
    from asyncssh.stream import SSHReader


async def _read_all(read: Callable[[int], Coroutine], n: Optional[int] = None) -> bytes:
    if n is None:
        return await read(-1)
    result = []
    while n > 0:
        data = await read(n)
        result.append(data)
        n -= len(data)
    return b"".join(result)


async def _getpass(*args, **kwargs) -> str:
    from getpass import getpass

    return await asyncio.to_thread(getpass, *args, **kwargs)


class _StderrWrapper:
    def __init__(self, stderr: "SSHReader", loop: asyncio.AbstractEventLoop) -> None:
        self.stderr = stderr
        self.loop = loop

    async def _readlines(self) -> list[bytes]:
        lines = []
        while True:
            line = await self.stderr.readline()
            if not line:
                break
            lines.append(line)
        return lines

    async def _read(self, n: Optional[int] = None) -> bytes:
        if self.stderr.at_eof():
            return b""
        return await _read_all(self.stderr.read, n)

    read = sync_wrapper(_read)
    readlines = sync_wrapper(_readlines)


class AsyncSSHWrapper(BaseAsyncObject):
    def __init__(
        self, conn: "SSHClientConnection", proc: "SSHClientProcess", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.conn: SSHClientConnection = conn
        self.proc: SSHClientProcess = proc
        self.stderr = _StderrWrapper(proc.stderr, self.loop)

    def can_read(self) -> bool:
        # pylint:disable=protected-access
        return self.proc.stdout._session._recv_buf_len > 0

    async def _read(self, n: Optional[int] = None) -> bytes:
        if self.proc.stdout.at_eof():
            return b""
        return await _read_all(self.proc.stdout.read, n)

    read = sync_wrapper(_read)

    def read_stderr(self, n: Optional[int] = None) -> bytes:
        return self.stderr.read(n)

    async def _write(self, data: bytes) -> None:
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    write = sync_wrapper(_write)

    async def _close(self) -> None:
        self.conn.close()
        await self.conn.wait_closed()

    close = sync_wrapper(_close)


class InteractiveSSHClient(SSHClient):
    _conn: Optional["SSHClientConnection"] = None
    _keys_to_try: Optional[list["FilePath"]] = None
    _passphrases: dict[str, str]

    def __init__(self, *args, **kwargs) -> None:
        super(*args, **kwargs)
        self._passphrases: dict[str, str] = {}

    def connection_made(self, conn: "SSHClientConnection") -> None:
        self._conn = conn
        self._keys_to_try = None

    def connection_lost(self, exc: Optional[Exception]) -> None:
        self._conn = None

    async def public_key_auth_requested(  # noqa: C901
        self,
    ) -> Optional["KeyPairListArg"]:
        from asyncssh.public_key import (
            _DEFAULT_KEY_FILES,
            KeyImportError,
            SSHLocalKeyPair,
            read_private_key,
            read_public_key,
        )

        if os.environ.get("GIT_TERMINAL_PROMPT") == "0":
            return None

        assert self._conn is not None
        if self._keys_to_try is None:
            self._keys_to_try = []
            options = self._conn._options  # pylint: disable=protected-access
            config = options.config
            client_keys = cast("Sequence[FilePath]", config.get("IdentityFile", ()))
            if not client_keys:
                client_keys = [
                    os.path.expanduser(os.path.join("~", ".ssh", path))
                    for path, cond in _DEFAULT_KEY_FILES
                    if cond
                ]
            for key_to_load in client_keys:
                try:
                    read_private_key(key_to_load, passphrase=options.passphrase)
                except KeyImportError as exc:
                    if str(exc).startswith("Passphrase"):
                        self._keys_to_try.append(key_to_load)
                except OSError:
                    pass

        while self._keys_to_try:
            key_to_load = self._keys_to_try.pop()
            pubkey_to_load = str(key_to_load) + ".pub"
            try:
                key = await self._read_private_key_interactive(key_to_load)
            except KeyImportError:
                continue
            try:
                pubkey = read_public_key(pubkey_to_load)
            except (OSError, KeyImportError):
                pubkey = None
            return SSHLocalKeyPair(key, pubkey, cert=None, enc_key=None)
        return None

    async def _read_private_key_interactive(self, path: "FilePath") -> "SSHKey":
        from asyncssh.public_key import (
            KeyEncryptionError,
            KeyImportError,
            read_private_key,
        )

        path = str(path)
        passphrase = self._passphrases.get(path)
        if passphrase:
            return read_private_key(path, passphrase=passphrase)

        for _ in range(3):
            passphrase = await _getpass(f"Enter passphrase for key {path!r}: ")
            if passphrase:
                try:
                    key = read_private_key(path, passphrase=passphrase)
                    self._passphrases[path] = passphrase
                    return key
                except (KeyImportError, KeyEncryptionError):
                    pass
        raise KeyImportError("Incorrect passphrase")

    def kbdint_auth_requested(self) -> "MaybeAwait[Optional[str]]":
        return ""

    async def kbdint_challenge_received(  # pylint: disable=invalid-overridden-method
        self,
        name: str,
        instructions: str,
        lang: str,
        prompts: "KbdIntPrompts",
    ) -> Optional["KbdIntResponse"]:
        if os.environ.get("GIT_TERMINAL_PROMPT") == "0":
            return None

        if instructions:
            pass

        response: list[str] = []
        for prompt, _echo in prompts:
            p = await _getpass(f"({name}) {prompt}" if name else prompt)
            response.append(p.rstrip())
        return response

    async def password_auth_requested(self) -> str:
        return await _getpass()


class AsyncSSHVendor(BaseAsyncObject, SSHVendor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    async def _run_command(
        self,
        host: str,
        command: list[str],
        username: Optional[str] = None,
        port: Optional[int] = None,
        password: Optional[str] = None,
        key_filename: Optional[str] = None,
        **kwargs: object,
    ) -> AsyncSSHWrapper:
        """Connect to an SSH server.

        Run a command remotely and return a file-like object for interaction
        with the remote command.

        Args:
          host: Host name
          command: Command to run (as argv array)
          username: Optional ame of user to log in as
          port: Optional SSH port to use
          password: Optional ssh password for login or private key
          key_filename: Optional path to private keyfile
        """
        import asyncssh

        try:
            conn = await asyncssh.connect(
                host,
                port=port if port is not None else (),
                username=username if username is not None else (),
                password=password,
                client_keys=[key_filename] if key_filename else (),
                ignore_encrypted=not key_filename,
                known_hosts=None,
                encoding=None,
                client_factory=InteractiveSSHClient,
            )
            proc: SSHClientProcess[Any] = await conn.create_process(
                command, encoding=None
            )
        except asyncssh.misc.PermissionDenied as exc:
            raise AuthError(f"{username}@{host}:{port or 22}") from exc
        return AsyncSSHWrapper(conn, proc)

    run_command = sync_wrapper(_run_command)


def get_unsupported_opts(config_paths: "ConfigPaths") -> Iterator[str]:
    from pathlib import Path, PurePath

    if config_paths:
        if isinstance(config_paths, (str, PurePath)):
            paths: Sequence[FilePath] = [config_paths]
        else:
            paths = config_paths

        for path in paths:
            try:
                yield from _parse_unsupported(Path(path))
            except FileNotFoundError:
                continue


def _parse_unsupported(path: "Path") -> Iterator[str]:
    import locale
    import shlex

    from asyncssh.config import SSHClientConfig

    handlers = SSHClientConfig._handlers  # pylint: disable=protected-access
    with open(path, encoding=locale.getpreferredencoding()) as fobj:
        for line in fobj:
            line = line.strip()
            if not line or line[0] == "#":
                continue

            try:
                args = shlex.split(line)
            except ValueError:
                continue

            option = args.pop(0)
            if option.endswith("="):
                option = option[:-1]
            elif "=" in option:
                option, _ = option.split("=", 1)
            loption = option.lower()
            if loption not in handlers:
                yield loption
