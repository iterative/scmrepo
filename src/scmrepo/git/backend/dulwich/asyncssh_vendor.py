"""asyncssh SSH vendor for Dulwich."""
import asyncio
import os
from typing import (
    TYPE_CHECKING,
    Callable,
    Coroutine,
    Iterator,
    List,
    Optional,
    Sequence,
)

from asyncssh import SSHClient
from dulwich.client import SSHVendor

from scmrepo.asyn import BaseAsyncObject, sync_wrapper
from scmrepo.exceptions import AuthError

if TYPE_CHECKING:
    from pathlib import Path

    from asyncssh.auth import KbdIntPrompts, KbdIntResponse
    from asyncssh.config import ConfigPaths, FilePath
    from asyncssh.connection import SSHClientConnection
    from asyncssh.misc import MaybeAwait
    from asyncssh.process import SSHClientProcess
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


class _StderrWrapper:
    def __init__(self, stderr: "SSHReader", loop: asyncio.AbstractEventLoop) -> None:
        self.stderr = stderr
        self.loop = loop

    async def _readlines(self) -> List[bytes]:
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
    def __init__(self, conn: "SSHClientConnection", proc: "SSHClientProcess", **kwargs):
        super().__init__(**kwargs)
        self.conn: "SSHClientConnection" = conn
        self.proc: "SSHClientProcess" = proc
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


# NOTE: Github's SSH server does not strictly comply with the SSH protocol.
# When validating a public key using the rsa-sha2-256 or rsa-sha2-512
# signature algorithms, RFC4252 + RFC8332 state that the server should respond
# with the same algorithm in SSH_MSG_USERAUTH_PK_OK. Github's server always
# returns "ssh-rsa" rather than the correct sha2 algorithm name (likely for
# backwards compatibility with old SSH client reasons). This behavior causes
# asyncssh to fail with a key-mismatch error (since asyncssh expects the server
# to behave properly).
#
# See also:
#   https://www.ietf.org/rfc/rfc4252.txt
#   https://www.ietf.org/rfc/rfc8332.txt
def _process_public_key_ok_gh(self, _pkttype, _pktid, packet):
    from asyncssh.misc import ProtocolError

    algorithm = packet.get_string()
    key_data = packet.get_string()
    packet.check_end()

    # pylint: disable=protected-access
    if (
        (
            algorithm == b"ssh-rsa"
            and self._keypair.algorithm
            not in (
                b"ssh-rsa",
                b"rsa-sha2-256",
                b"rsa-sha2-512",
            )
        )
        or (algorithm != b"ssh-rsa" and algorithm != self._keypair.algorithm)
        or key_data != self._keypair.public_data
    ):
        raise ProtocolError("Key mismatch")

    self.create_task(self._send_signed_request())
    return True


class InteractiveSSHClient(SSHClient):
    def kbdint_auth_requested(self) -> "MaybeAwait[Optional[str]]":
        return ""

    async def kbdint_challenge_received(  # pylint: disable=invalid-overridden-method
        self,
        name: str,
        instructions: str,
        lang: str,
        prompts: "KbdIntPrompts",
    ) -> Optional["KbdIntResponse"]:
        from getpass import getpass

        if os.environ.get("GIT_TERMINAL_PROMPT") == "0":
            return None

        def _getpass(prompt: str) -> str:
            return getpass(prompt=prompt).rstrip()

        if instructions:
            print(instructions)
        loop = asyncio.get_running_loop()
        return [
            await loop.run_in_executor(
                None, _getpass, f"({name}) {prompt}" if name else prompt
            )
            for prompt, _ in prompts
        ]


class AsyncSSHVendor(BaseAsyncObject, SSHVendor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    async def _run_command(
        self,
        host: str,
        command: List[str],
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
        from asyncssh.auth import MSG_USERAUTH_PK_OK, _ClientPublicKeyAuth

        # pylint: disable=protected-access
        _ClientPublicKeyAuth._packet_handlers[
            MSG_USERAUTH_PK_OK
        ] = _process_public_key_ok_gh

        try:
            conn = await asyncssh.connect(
                host,
                port=port if port is not None else (),
                username=username if username is not None else (),
                password=password if password is not None else (),
                client_keys=[key_filename] if key_filename else (),
                ignore_encrypted=not key_filename,
                known_hosts=None,
                encoding=None,
                client_factory=InteractiveSSHClient,
            )
            proc = await conn.create_process(command, encoding=None)
        except asyncssh.misc.PermissionDenied as exc:
            raise AuthError(f"{username}@{host}:{port or 22}") from exc
        return AsyncSSHWrapper(conn, proc)

    run_command = sync_wrapper(_run_command)


def get_unsupported_opts(config_paths: "ConfigPaths") -> Iterator[str]:
    from pathlib import Path, PurePath

    if config_paths:
        if isinstance(config_paths, (str, PurePath)):
            paths: Sequence["FilePath"] = [config_paths]
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
