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
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse, urlunparse

from dulwich.config import StackedConfig
from dulwich.credentials import urlmatch_credential_sections
from funcy import cached_property

from scmrepo.exceptions import SCMError

if TYPE_CHECKING:
    from dulwich.config import ConfigDict

logger = logging.getLogger(__name__)

SectionLike = Union[bytes, str, Tuple[Union[bytes, str], ...]]


class CredentialNotFoundError(SCMError):
    """Error occurred while retrieving credentials/no credentials available."""


class CredentialHelper:
    """Helper for retrieving credentials for http/https git remotes

    Usage:
    >>> helper = CredentialHelper("store") # Use `git credential-store`
    >>> credentials = helper.get("https://github.com/dtrifiro/aprivaterepo")
    >>> username = credentials["username"]
    >>> password = credentials["password"]
    """

    def __init__(self, command: str):
        self._command = command
        self._run_kwargs: Dict[str, Any] = {}
        if self._command[0] == "!":
            # On Windows this will only work in git-bash and/or WSL2
            self._run_kwargs["shell"] = True
        self._encoding = locale.getpreferredencoding()

    def _prepare_command(self, action: Optional[str] = None) -> Union[str, List[str]]:
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
            git_exec_path = subprocess.check_output(  # nosec B603
                ("git", "--exec-path"),
                text=True,
            ).strip()
            if shutil.which(executable, path=git_exec_path):
                executable = os.path.join(git_exec_path, executable)

        return [executable, *argv[1:]]

    def get(
        self,
        **kwargs,
    ) -> "Credential":
        if kwargs.get("protocol", kwargs.get("hostname")) is None:
            raise ValueError("One of protocol, hostname must be provided")
        cmd = self._prepare_command("get")
        helper_input = [f"{key}={value}" for key, value in kwargs.items()]
        helper_input.append("")

        try:
            res = subprocess.run(  # type: ignore # nosec B603 # breaks on 3.6
                cmd,
                check=True,
                capture_output=True,
                input=os.linesep.join(helper_input),
                encoding=self._encoding,
                **self._run_kwargs,
            )
        except subprocess.CalledProcessError as exc:
            raise CredentialNotFoundError(exc.stderr) from exc
        except FileNotFoundError as exc:
            raise CredentialNotFoundError("Helper not found") from exc
        if res.stderr:
            logger.debug(res.stderr)

        credentials = {}
        for line in res.stdout.strip().splitlines():
            try:
                key, value = line.split("=")
                credentials[key] = value
            except ValueError:
                continue
        return Credential(**credentials)

    def store(self, **kwargs):
        """Store the credential, if applicable to the helper"""
        cmd = self._prepare_command("store")
        helper_input = [f"{key}={value}" for key, value in kwargs.items()]
        helper_input.append("")

        try:
            res = subprocess.run(  # type: ignore # nosec B603 # pylint: disable=W1510
                cmd,
                capture_output=True,
                input=os.linesep.join(helper_input),
                encoding=self._encoding,
                **self._run_kwargs,
            )
            if res.stderr:
                logger.debug(res.stderr)
        except FileNotFoundError:
            logger.debug("Helper not found", exc_info=True)

    def erase(self, **kwargs):
        """Remove a matching credential, if any, from the helper’s storage"""
        cmd = self._prepare_command("erase")
        helper_input = [f"{key}={value}" for key, value in kwargs.items()]
        helper_input.append("")

        try:
            res = subprocess.run(  # type: ignore # nosec B603 # pylint: disable=W1510
                cmd,
                capture_output=True,
                input=os.linesep.join(helper_input),
                encoding=self._encoding,
                **self._run_kwargs,
            )
            if res.stderr:
                logger.debug(res.stderr)
        except FileNotFoundError:
            logger.debug("Helper not found", exc_info=True)


def get_matching_helper_commands(
    base_url: str, config: Optional[Union["ConfigDict", "StackedConfig"]] = None
):
    config = config or StackedConfig.default()
    if isinstance(config, StackedConfig):
        backends: Iterable["ConfigDict"] = config.backends
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
            yield command.decode(conf.encoding or sys.getdefaultencoding())


class Credential:
    """Git credentials, equivalent to CGit git-credential API.

    Usage:

    1. Generate a credential based on context

        >>> generated = Credential(url="https://github.com/dtrifiro/aprivaterepo")

    2. Ask git-credential to give username/password for this context

        >>> credential = generated.fill()

    3. Use the credential from (2) in Git operation
    4. If the operation in (3) was successful, approve it for re-use in subsequent
       operations

       >>> credential.approve()

    See also:
        https://git-scm.com/docs/git-credential#_typical_use_of_git_credential
        https://github.com/git/git/blob/master/credential.h

    """

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

    @property
    def url(self) -> str:
        if self.username or self.password:
            username = self.username or ""
            password = self.password or ""
            netloc = f"{username}:{password}@{self.host}"
        else:
            netloc = self.host or ""
        return urlunparse((self.protocol or "", netloc, self.path or "", "", "", ""))

    @property
    def _helper_kwargs(self) -> Dict[str, str]:
        kwargs = {}
        for attr in (
            "protocol",
            "host",
            "path",
            "username",
            "password",
            "password_expiry_utc",
        ):
            value = getattr(self, attr)
            if value is not None:
                kwargs[attr] = str(value)
        return kwargs

    @cached_property
    def helpers(self) -> List["CredentialHelper"]:
        url = self.url
        return [
            CredentialHelper(command) for command in get_matching_helper_commands(url)
        ]

    def fill(self) -> "Credential":
        """Return a new credential with filled username and password."""
        for helper in self.helpers:
            try:
                return helper.get(**self._helper_kwargs)
            except CredentialNotFoundError:
                continue
        raise CredentialNotFoundError(f"No available credentials for '{self.url}'")

    def approve(self):
        """Store this credential in available helpers."""
        if self._approved or not (self.username and self.password):
            return
        for helper in self.helpers:
            helper.store(**self._helper_kwargs)
        self._approved = True

    def reject(self):
        """Remove this credential from available helpers."""
        for helper in self.helpers:
            helper.erase(**self._helper_kwargs)
        self._approved = False
