# credentials.py -- support for git credential helpers

# Copyright (C) 2022 Daniele Trifirò <daniele@iterative.ai>
#
# Dulwich is dual-licensed under the Apache License, Version 2.0 and the GNU
# General Public License as public by the Free Software Foundation; version 2.0
# or (at your option) any later version. You can redistribute it and/or
# modify it under the terms of either of these two licenses.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# You should have received a copy of the licenses; if not, see
# <http://www.gnu.org/licenses/> for a copy of the GNU General Public License
# and <http://www.apache.org/licenses/LICENSE-2.0> for a copy of the Apache
# License, Version 2.0.
#

"""Support for git credential helpers

https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage

Currently Dulwich supports only the `get` operation

"""

import locale
import logging
import os
import shlex
import shutil
import subprocess  # nosec B404
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    Optional,
    Union,
)
from urllib.parse import urlparse, urlunparse

from dulwich.config import StackedConfig
from dulwich.credentials import urlmatch_credential_sections
from funcy import cached_property

from scmrepo.exceptions import SCMError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dulwich.config import ConfigDict

logger = logging.getLogger(__name__)

SectionLike = Union[bytes, str, tuple[Union[bytes, str], ...]]


class CredentialNotFoundError(SCMError):
    """Error occurred while retrieving credentials/no credentials available."""


class CredentialQuitError(SCMError):
    """Credential helper returned quit=1."""


class CredentialHelper(ABC):
    """Base git-credential helper."""

    @abstractmethod
    def get(self, credential: "Credential", **kwargs) -> "Credential":
        """Get a matching credential from this helper.

        Raises:
            CredentialNotFoundError: No matching credential was found.
        """

    @abstractmethod
    def store(self, credential: "Credential", **kwargs):
        """Store the credential, if applicable to the helper"""

    @abstractmethod
    def erase(self, credential: "Credential", **kwargs):
        """Remove a matching credential, if any, from the helper's storage"""


class GitCredentialHelper(CredentialHelper):
    """Helper for retrieving credentials through git-credential-<helper> commands.

    Users should generally not need to use credential helpers directly.
    The corresponding Credential methods should be used instead.
    (i.e. Credential.fill() instead of GitCredentialHelper.get())

    Usage:
    >>> helper = GitCredentialHelper("store") # Use `git credential-store`
    >>> generated = Credential(url="https://github.com/dtrifiro/aprivaterepo")
    >>> credentials = helper.get(generated)
    >>> username = credentials.username
    >>> password = credentials.password
    """

    def __init__(self, command: str, use_http_path: bool = False):
        super().__init__()
        self._command = command
        self._run_kwargs: dict[str, Any] = {}
        if self._command[0] == "!":
            # On Windows this will only work in git-bash and/or WSL2
            self._run_kwargs["shell"] = True
        self._encoding = locale.getpreferredencoding()
        self.use_http_path = use_http_path

    def _prepare_command(self, action: Optional[str] = None) -> Union[str, list[str]]:
        if self._command[0] == "!":
            return self._command[1:] + (f" {action}" if action else "")

        if sys.platform != "win32":
            argv = shlex.split(self._command)
        else:
            # On windows, subprocess.run uses subprocess.list2cmdline() to
            # join arguments when providing a list, so we can just split
            # using whitespace.
            argv = self._command.split()
        if action:
            argv.append(action)

        if os.path.isabs(argv[0]):
            return argv

        executable = f"git-credential-{argv[0]}"
        if not shutil.which(executable) and shutil.which("git"):
            # If the helper cannot be found in PATH, it might be
            # a C git helper in GIT_EXEC_PATH
            git_exec_path = subprocess.check_output(  # noqa: S603
                ("git", "--exec-path"),
                text=True,
            ).strip()
            if shutil.which(executable, path=git_exec_path):
                executable = os.path.join(git_exec_path, executable)

        return [executable, *argv[1:]]

    def get(self, credential: "Credential", **kwargs) -> "Credential":
        if not (credential.protocol or credential.host):
            raise ValueError("One of protocol, hostname must be provided")
        cmd = self._prepare_command("get")
        use_path = credential.protocol in ("http", "https") and self.use_http_path
        helper_input = [
            f"{key}={value}"
            for key, value in credential.items()
            if key != "path" or use_path
        ]
        helper_input.append("")

        try:
            res = subprocess.run(  # noqa: S603
                cmd,
                check=True,
                capture_output=True,
                input="\n".join(helper_input),
                encoding=self._encoding,
                **self._run_kwargs,
            )
        except subprocess.CalledProcessError as exc:
            raise CredentialNotFoundError(exc.stderr) from exc
        except FileNotFoundError as exc:
            raise CredentialNotFoundError("Helper not found") from exc
        if res.stderr:
            logger.debug(res.stderr)

        credentials: dict[str, Any] = {}
        for line in res.stdout.splitlines():
            try:
                key, value = line.split("=", maxsplit=1)
                # Only include credential values that are used in the Credential
                # constructor.
                # Other values may be returned by the subprocess, but they must be
                # ignored.
                # e.g. osxkeychain credential helper >= 2.46.0 can return
                # `capability[]` and `state`)
                if key in [
                    "protocol",
                    "host",
                    "path",
                    "username",
                    "password",
                    "password_expiry_utc",
                    "url",
                ]:
                    # Garbage bytes were output from git-credential-osxkeychain from
                    # 2.45.0 to 2.47.0:
                    # https://github.com/git/git/commit/6c3c451fb6e1c3ca83f74e63079d4d0af01b2d69
                    credentials[key] = _strip_garbage_bytes(value)
            except ValueError:
                continue
        if not credentials:
            raise CredentialNotFoundError("No credentials found")
        quit_ = credentials.get("quit")
        if quit_ is not None and quit_.lower() in ("true", "1"):
            raise CredentialQuitError("Helper returned quit=1")
        return Credential(**credentials)

    def store(self, credential: "Credential", **kwargs):
        """Store the credential, if applicable to the helper"""
        cmd = self._prepare_command("store")
        use_path = credential.protocol in ("http", "https") and self.use_http_path
        helper_input = [
            f"{key}={value}"
            for key, value in credential.items()
            if key != "path" or use_path
        ]
        helper_input.append("")

        try:
            res = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                input="\n".join(helper_input),
                encoding=self._encoding,
                **self._run_kwargs,
                check=False,
            )
            if res.stderr:
                logger.debug(res.stderr)
        except FileNotFoundError:
            logger.debug("Helper not found", exc_info=True)

    def erase(self, credential: "Credential", **kwargs):
        """Remove a matching credential, if any, from the helper's storage"""
        cmd = self._prepare_command("erase")
        use_path = credential.protocol in ("http", "https") and self.use_http_path
        helper_input = [
            f"{key}={value}"
            for key, value in credential.items()
            if key != "path" or use_path
        ]
        helper_input.append("")

        try:
            res = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                input="\n".join(helper_input),
                encoding=self._encoding,
                **self._run_kwargs,
                check=False,
            )
            if res.stderr:
                logger.debug(res.stderr)
        except FileNotFoundError:
            logger.debug("Helper not found", exc_info=True)

    @staticmethod
    def get_matching_commands(
        base_url: str, config: Optional[Union["ConfigDict", "StackedConfig"]] = None
    ) -> Iterator[tuple[str, bool]]:
        config = config or StackedConfig.default()
        if isinstance(config, StackedConfig):
            backends: Iterable[ConfigDict] = config.backends
        else:
            backends = [config]

        for conf in backends:
            # We will try to match credential sections' url with the given url,
            # falling back to the generic section if there's no match
            for section in urlmatch_credential_sections(conf, base_url):
                try:
                    command = conf.get(section, "helper")
                except KeyError:
                    # no helper configured
                    continue
                if not command:
                    continue
                use_http_path = conf.get_boolean(section, "usehttppath", False)
                yield (
                    command.decode(conf.encoding or sys.getdefaultencoding()),
                    use_http_path,
                )


def _strip_garbage_bytes(s: str) -> str:
    """
    Garbage (random) bytes were output from git-credential-osxkeychain from
    2.45.0 to 2.47.0 so must be removed.
    https://github.com/git/git/commit/6c3c451fb6e1c3ca83f74e63079d4d0af01b2d69
    :param s: string that might contain garbage/random bytes
    :return str: The string with the garbage bytes removed
    """
    # Assume that any garbage bytes begin with a 0-byte
    zero = s.find(chr(0))
    return s[0:zero] if zero >= 0 else s


class _CredentialKey(NamedTuple):
    protocol: str
    host: Optional[str]
    path: Optional[str]

    @classmethod
    def from_credential(cls, credential: "Credential"):
        return cls(credential.protocol or "", credential.host, credential.path)


def _input_tty(prompt: str = "Username: ") -> str:
    """Prompt for username on /dev/tty when available.

    Defaults to builtin input() when /dev/tty cannot be opened, does not disable
    echo (use getpass.getpass() instead).
    """
    import io
    from contextlib import ExitStack

    if os.name == "nt":
        if not sys.stdin.isatty():
            raise EOFError
        return input(prompt)

    with ExitStack() as stack:
        try:
            fd = os.open(
                "/dev/tty",
                os.O_RDWR | os.O_NOCTTY,  # pylint: disable=no-member
            )
            tty = io.FileIO(fd, "w+")
            stack.enter_context(tty)
            stream = io.TextIOWrapper(tty)
            stack.enter_context(stream)
        except OSError:
            stack.close()
            # fallback to default input()
            if not sys.stdin.isatty():
                raise
            return input(prompt)
        try:
            stream.write(prompt)
            stream.flush()
            line = stream.readline()
            if not line:
                raise EOFError
            if line[-1] == "\n":
                line = line[:-1]
            return line
        finally:
            stream.flush()
    raise EOFError


class MemoryCredentialHelper(CredentialHelper):
    """Memory credential helper that supports optional interactive input."""

    def __init__(self):
        super().__init__()
        self._credentials: dict[_CredentialKey, Credential] = {}

    def __getitem__(self, key: object) -> "Credential":
        if isinstance(key, _CredentialKey):
            return self._credentials[key]
        if isinstance(key, Credential):
            return self._credentials[_CredentialKey.from_credential(key)]
        raise KeyError

    def __setitem__(self, key: object, value: "Credential"):
        if isinstance(key, _CredentialKey):
            self._credentials[key] = value
        elif isinstance(key, Credential):
            self._credentials[_CredentialKey.from_credential(key)] = value
        else:
            raise ValueError

    def __delitem__(self, key: object):
        if isinstance(key, _CredentialKey):
            del self._credentials[key]
        elif isinstance(key, Credential):
            del self._credentials[_CredentialKey.from_credential(key)]
        else:
            raise ValueError

    def get(
        self, credential: "Credential", *, interactive: bool = False, **kwargs
    ) -> "Credential":
        """Get a matching credential from this helper.

        Raises:
            CredentialNotFoundError: No matching credential was found.
        """
        from getpass import getpass

        if credential.path:
            try_creds = [
                credential,
                Credential(protocol=credential.protocol, host=credential.host),
            ]
        else:
            try_creds = [credential]
        for cred in try_creds:
            try:
                return self[cred]
            except KeyError:
                pass
        if interactive:
            try:
                if self.askpass:
                    return self._get_interactive(credential, self.askpass.input)
                if os.environ.get("GIT_TERMINAL_PROMPT") != "0":
                    return self._get_interactive(credential, _input_tty, getpass)
            except (EOFError, OSError):
                pass
        raise CredentialNotFoundError("No matching credentials")

    def _get_interactive(
        self,
        credential: "Credential",
        input_echo: Callable[[str], str],
        input_noecho: Optional[Callable[[str], str]] = None,
    ) -> "Credential":
        if not input_noecho:
            input_noecho = input_echo
        new = Credential(
            protocol=credential.protocol,
            host=credential.host,
            path=credential.path,
            username=credential.username,
            password=credential.password,
        )
        try:
            if not new.username:
                prompt = f"Username for '{new.describe()}': "
                new.username = input_echo(prompt)
            if not new.password:
                prompt = f"Password for '{new.describe()}': "
                new.password = input_noecho(prompt)
        except KeyboardInterrupt as exc:
            raise CredentialNotFoundError("User cancelled prompt") from exc
        return new

    def store(self, credential: "Credential", **kwargs):
        """Store the credential, if applicable to the helper"""
        if credential.protocol or credential.host or credential.path:
            self[credential] = credential

    def erase(self, credential: "Credential", **kwargs):
        """Remove a matching credential, if any, from the helper's storage"""
        try:
            del self[credential]
        except KeyError:
            pass

    @cached_property
    def askpass(self) -> Optional["_AskpassCommand"]:
        return self.get_askpass()

    @staticmethod
    def get_askpass(
        config: Optional[Union["ConfigDict", "StackedConfig"]] = None,
    ) -> Optional["_AskpassCommand"]:
        askpass = os.environ.get("GIT_ASKPASS")
        if not askpass:
            config = config or StackedConfig.default()
            try:
                askpass = config.get("core", "askpass").decode(sys.getdefaultencoding())
            except KeyError:
                pass
        if not askpass:
            askpass = os.environ.get("SSH_ASKPASS")
        if askpass:
            return _AskpassCommand(askpass)
        return None


class _AskpassCommand:
    def __init__(self, command: str):
        self.command = command

    def input(self, prompt: str) -> str:
        argv = [self.command, prompt]
        try:
            res = subprocess.run(  # noqa: S603
                argv,
                check=True,
                capture_output=True,
                encoding=locale.getpreferredencoding(),
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return ""
        return res.stdout.splitlines()[0]


memory_helper = MemoryCredentialHelper()


class Credential(Mapping[str, str]):
    """Git credentials, equivalent to CGit git-credential API.

    Usage:

    1. Generate a credential based on context

        >>> generated = Credential(url="https://github.com/dtrifiro/aprivaterepo")

    2. Ask git-credential to give username/password for this context

        >>> credential = generated.fill()

    3. Use the credential from (2) in Git operation
    4. If the operation in (3) was successful, approve it for reuse in subsequent
       operations

       >>> credential.approve()

    See also:
        https://git-scm.com/docs/git-credential#_typical_use_of_git_credential
        https://github.com/git/git/blob/master/credential.h

    """

    _SUPPORTED_KEYS = (
        "protocol",
        "host",
        "path",
        "username",
        "password",
        "password_expiry_utc",
    )

    def __init__(
        self,
        *,
        protocol: Optional[str] = None,
        host: Optional[str] = None,  # host with optional ":<port>" included
        path: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        password_expiry_utc: Optional[int] = None,
        url: Optional[str] = None,
    ):
        self.protocol = protocol
        self.host = host
        self.path = path
        self.username = username
        self.password = password
        self.password_expiry_utc = password_expiry_utc
        self._approved = False
        if url:
            parsed = urlparse(url)
            self.protocol = self.protocol or parsed.scheme
            if not self.protocol:
                raise ValueError("protocol must be specified when using URL")
            port = f":{parsed.port}" if parsed.port is not None else ""
            hostname = parsed.hostname or ""
            self.host = self.host or f"{hostname}{port}"
            self.username = self.username or parsed.username
            self.password = self.password or parsed.password
            if parsed.path:
                self.path = self.path or parsed.path.lstrip("/")

    def __getitem__(self, key: object) -> str:
        if isinstance(key, str):
            try:
                return getattr(self, key)
            except AttributeError:
                pass
        raise KeyError

    def __iter__(self) -> Iterator[str]:
        for key in self._SUPPORTED_KEYS:
            try:
                value = self[key]
                if value is not None:
                    yield key
            except KeyError:
                pass

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __str__(self) -> str:
        return self.describe(use_path=True, sanitize=True)

    @property
    def url(self) -> str:
        return self.describe(use_path=True, sanitize=False)

    def describe(self, use_path: bool = False, sanitize: bool = True) -> str:
        username = self.username or ""
        if sanitize:
            password = ":***" if self.password else ""
        else:
            password = f":{self.password}" if self.password else ""
        at = "@" if (username or password) else ""
        host = f"{self.host}" if self.host else ""
        netloc = f"{username}{password}{at}{host}"
        if use_path:
            path = self.path or ""
        else:
            path = ""
        return urlunparse((self.protocol or "", netloc, path, "", "", ""))

    @cached_property
    def helpers(self) -> list["CredentialHelper"]:
        url = self.url
        return [
            GitCredentialHelper(command, use_http_path=use_http_path)
            for command, use_http_path in GitCredentialHelper.get_matching_commands(url)
        ]

    def fill(self, interactive: bool = True) -> "Credential":
        """Return a new credential with filled username and password."""
        if self.username and self.password:
            return Credential(**self)

        try:
            return memory_helper.get(self, interactive=False)
        except CredentialNotFoundError:
            pass

        for helper in self.helpers:
            try:
                return helper.get(self)
            except CredentialNotFoundError:
                continue
            except CredentialQuitError as exc:
                raise CredentialNotFoundError(
                    f"No available credentials for '{self}'"
                ) from exc

        try:
            return memory_helper.get(self, interactive=interactive)
        except CredentialNotFoundError:
            pass

        raise CredentialNotFoundError(f"No available credentials for '{self}'")

    def approve(self):
        """Store this credential in available helpers."""
        if self._approved or not (self.username and self.password):
            return
        for helper in self.helpers:
            helper.store(self)
        memory_helper.store(self)
        self._approved = True

    def reject(self):
        """Remove this credential from available helpers."""
        for helper in self.helpers:
            helper.erase(self)
        memory_helper.erase(self)
        self._approved = False
