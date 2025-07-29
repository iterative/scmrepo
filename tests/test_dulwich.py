import os
import socket
import threading
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock

import asyncssh
import paramiko
import pytest
from paramiko.server import InteractiveQuery
from pytest_mock import MockerFixture
from pytest_test_utils.waiters import wait_until

from scmrepo.exceptions import AuthError
from scmrepo.git.backend.dulwich.asyncssh_vendor import AsyncSSHVendor

# pylint: disable=redefined-outer-name


USER = "testuser"
PASSWORD = "test"
CLIENT_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEAxvREKSElPOm/0z/nPO+j5rk2tjdgGcGc7We1QZ6TRXYLu7nN
GeEFIL4p8N1i6dmB+Eydt7xqCU79MWD6Yy4prFe1+/K1wCDUxIbFMxqQcX5zjJzd
i8j8PbcaUlVhP/OkjtkSxrXaGDO1BzfdV4iEBtTV/2l3zmLKJlt3jnOHLczP24CB
DTQKp3rKshbRefzot9Y+wnaK692RsYgsyo9YEP0GyWKG9topCHk13r46J6vGLeuj
ryUKqmbLJkzbJbIcEqwTDo5iHaCVqaMr5Hrb8BdMucSseqZQJsXSd+9tdRcIblUQ
38kZjmFMm4SFbruJcpZCNM2wNSZPIRX+3eiwNwIDAQABAoIBAHSacOBSJsr+jIi5
KUOTh9IPtzswVUiDKwARCjB9Sf8p4lKR4N1L/n9kNJyQhApeikgGT2GCMftmqgoo
tlculQoHFgemBlOmak0MV8NNzF5YKEy/GzF0CDH7gJfEpoyetVFrdA+2QS5yD6U9
XqKQxiBi2VEqdScmyyeT8AwzNYTnPeH/DOEcnbdRjqiy/CD79F49CQ1lX1Fuqm0K
I7BivBH1xo/rVnUP4F+IzocDqoga+Pjdj0LTXIgJlHQDSbhsQqWujWQDDuKb+MAw
sNK4Zf8ErV3j1PyA7f/M5LLq6zgstkW4qikDHo4SpZX8kFOO8tjqb7kujj7XqeaB
CxqrOTECgYEA73uWkrohcmDJ4KqbuL3tbExSCOUiaIV+sT1eGPNi7GCmXD4eW5Z4
75v2IHymW83lORSu/DrQ6sKr1nkuRpqr2iBzRmQpl/H+wahIhBXlnJ25uUjDsuPO
1Pq2LcmyD+jTxVnmbSe/q7O09gZQw3I6H4+BMHmpbf8tC97lqimzpJ0CgYEA1K0W
ZL70Xtn9quyHvbtae/BW07NZnxvUg4UaVIAL9Zu34JyplJzyzbIjrmlDbv6aRogH
/KtuG9tfbf55K/jjqNORiuRtzt1hUN1ye4dyW7tHx2/7lXdlqtyK40rQl8P0kqf8
zaS6BqjnobgSdSpg32rWoL/pcBHPdJCJEgQ8zeMCgYEA0/PK8TOhNIzrP1dgGSKn
hkkJ9etuB5nW5mEM7gJDFDf6JPupfJ/xiwe6z0fjKK9S57EhqgUYMB55XYnE5iIw
ZQ6BV9SAZ4V7VsRs4dJLdNC3tn/rDGHJBgCaym2PlbsX6rvFT+h1IC8dwv0V79Ui
Ehq9WTzkMoE8yhvNokvkPZUCgYEAgBAFxv5xGdh79ftdtXLmhnDvZ6S8l6Fjcxqo
Ay/jg66Tp43OU226iv/0mmZKM8Dd1xC8dnon4GBVc19jSYYiWBulrRPlx0Xo/o+K
CzZBN1lrXH1i6dqufpc0jq8TMf/N+q1q/c1uMupsKCY1/xVYpc+ok71b7J7c49zQ
nOeuUW8CgYA9Infooy65FTgbzca0c9kbCUBmcAPQ2ItH3JcPKWPQTDuV62HcT00o
fZdIV47Nez1W5Clk191RMy8TXuqI54kocciUWpThc6j44hz49oUueb8U4bLcEHzA
WxtWBWHwxfSmqgTXilEA3ALJp0kNolLnEttnhENwJpZHlqtes0ZA4w==
-----END RSA PRIVATE KEY-----"""


class Server(paramiko.ServerInterface):
    """http://docs.paramiko.org/en/2.4/api/server.html."""

    def __init__(self, commands, *args, **kwargs) -> None:
        super().__init__()
        self.commands = commands
        self.allowed_auths = kwargs.get("allowed_auths", "publickey,password")

    def check_channel_exec_request(self, channel, command):
        self.commands.append(command)
        return True

    def check_auth_interactive(self, username: str, submethods: str):
        return InteractiveQuery(
            "Password", "Enter the password", f"Password for user {USER}:"
        )

    def check_auth_interactive_response(self, responses):
        if responses[0] == PASSWORD:
            return paramiko.AUTH_SUCCESSFUL  # type: ignore[attr-defined]
        return paramiko.AUTH_FAILED  # type: ignore[attr-defined]

    def check_auth_password(self, username, password):
        if username == USER and password == PASSWORD:
            return paramiko.AUTH_SUCCESSFUL  # type: ignore[attr-defined]
        return paramiko.AUTH_FAILED  # type: ignore[attr-defined]

    def check_auth_publickey(self, username, key):
        pubkey = paramiko.RSAKey.from_private_key(StringIO(CLIENT_KEY))
        if username == USER and key == pubkey:
            return paramiko.AUTH_SUCCESSFUL  # type: ignore[attr-defined]
        return paramiko.AUTH_FAILED  # type: ignore[attr-defined]

    def check_channel_request(self, kind, chanid):
        if kind == "session":
            return paramiko.OPEN_SUCCEEDED  # type: ignore[attr-defined]
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED  # type: ignore[attr-defined]

    def get_allowed_auths(self, username):
        return self.allowed_auths


@pytest.fixture
def ssh_conn(request: pytest.FixtureRequest) -> dict[str, Any]:
    server = Server([], **getattr(request, "param", {}))

    socket.setdefaulttimeout(10)
    request.addfinalizer(lambda: socket.setdefaulttimeout(None))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(5)
    request.addfinalizer(sock.close)  # noqa: PT021
    port = sock.getsockname()[1]

    conn_info: dict[str, Any] = {"port": port, "server": server}

    def _run_server():
        try:
            conn, _ = sock.accept()
        except OSError:
            return False
        server.transport = transport = paramiko.Transport(conn)  # type: ignore[attr-defined]
        request.addfinalizer(transport.close)
        host_key = paramiko.RSAKey.from_private_key(StringIO(CLIENT_KEY))
        transport.add_server_key(host_key)
        transport.start_server(server=server)

    thread = threading.Thread(target=_run_server)
    thread.start()
    return conn_info


@pytest.fixture
def ssh_port(ssh_conn: dict[str, Any]) -> int:
    return ssh_conn["port"]


@pytest.fixture
def server(ssh_conn: dict[str, Any]) -> Server:
    return ssh_conn["server"]


def test_run_command_password(server: Server, ssh_port: int):
    vendor = AsyncSSHVendor()
    vendor.run_command(
        "127.0.0.1",
        "test_run_command_password",
        username=USER,
        port=ssh_port,
        password=PASSWORD,
    )

    assert b"test_run_command_password" in server.commands


@pytest.mark.parametrize("ssh_conn", [{"allowed_auths": "publickey"}], indirect=True)
def test_run_command_no_password(ssh_port: int):
    vendor = AsyncSSHVendor()
    with pytest.raises(AuthError):
        vendor.run_command(
            "127.0.0.1",
            "test_run_command_password",
            username=USER,
            port=ssh_port,
            password=None,
        )


@pytest.mark.parametrize(
    "ssh_conn",
    [{"allowed_auths": "password"}, {"allowed_auths": "keyboard-interactive"}],
    indirect=True,
    ids=["password", "interactive"],
)
def test_should_prompt_for_password_when_no_password_passed(
    mocker: MockerFixture, server: Server, ssh_port: int
):
    mocked_getpass = mocker.patch("getpass.getpass", return_value=PASSWORD)
    vendor = AsyncSSHVendor()
    vendor.run_command(
        "127.0.0.1",
        "test_run_command_password",
        username=USER,
        port=ssh_port,
        password=None,
    )
    assert server.commands == [b"test_run_command_password"]
    mocked_getpass.asssert_called_once()


def test_run_command_with_privkey(server: Server, ssh_port: int):
    key = asyncssh.import_private_key(CLIENT_KEY)

    vendor = AsyncSSHVendor()
    vendor.run_command(
        "127.0.0.1",
        "test_run_command_with_privkey",
        username=USER,
        port=ssh_port,
        key_filename=key,
    )

    assert b"test_run_command_with_privkey" in server.commands


@pytest.mark.parametrize("use_len", [True, False])
def test_run_command_data_transfer(server: Server, ssh_port: int, use_len: bool):
    vendor = AsyncSSHVendor()
    con = vendor.run_command(
        "127.0.0.1",
        "test_run_command_data_transfer",
        username=USER,
        port=ssh_port,
        password=PASSWORD,
    )

    assert b"test_run_command_data_transfer" in server.commands

    channel = server.transport.accept(5)  # type: ignore[attr-defined]
    channel.send(b"stdout\n")
    channel.send_stderr(b"stderr\n")
    channel.close()

    assert wait_until(con.can_read, timeout=1, pause=0.1)
    assert con.read(n=7 if use_len else None) == b"stdout\n"
    assert con.read_stderr(n=7 if use_len else None) == b"stderr\n"


def test_run_command_partial_transfer(ssh_port: int, mocker: MockerFixture):
    vendor = AsyncSSHVendor()
    con = vendor.run_command(
        "127.0.0.1",
        "test_run_command_data_transfer",
        username=USER,
        port=ssh_port,
        password=PASSWORD,
    )

    mock_stdout = mocker.patch.object(
        con.proc.stdout,
        "read",
        side_effect=[b"s", b"tdout", b"\n"],
        new_callable=AsyncMock,
    )
    assert con.read(n=7) == b"stdout\n"
    assert mock_stdout.call_count == 3

    mock_stderr = mocker.patch.object(
        con.stderr.stderr,
        "read",
        side_effect=[b"s", b"tderr", b"\n"],
        new_callable=AsyncMock,
    )
    assert con.read_stderr(n=7) == b"stderr\n"
    assert mock_stderr.call_count == 3


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_git_bash_ssh_vendor(mocker):
    from dulwich.client import SubprocessSSHVendor

    from scmrepo.git.backend.dulwich import _get_ssh_vendor

    mocker.patch.dict(os.environ, {"MSYSTEM": "MINGW64"})
    assert isinstance(_get_ssh_vendor(), SubprocessSSHVendor)

    del os.environ["MSYSTEM"]
    assert isinstance(_get_ssh_vendor(), AsyncSSHVendor)


def test_unsupported_config_ssh_vendor():
    from dulwich.client import SubprocessSSHVendor

    from scmrepo.git.backend.dulwich import _get_ssh_vendor

    config = os.path.expanduser(os.path.join("~", ".ssh", "config"))
    os.makedirs(os.path.dirname(config), exist_ok=True)

    with open(config, "wb") as fobj:
        fobj.write(
            b"""
Host *
    IdentityFile ~/.ssh/id_rsa
"""
        )
    assert isinstance(_get_ssh_vendor(), AsyncSSHVendor)

    with open(config, "wb") as fobj:
        fobj.write(
            b"""
Host *
    UseKeychain yes
"""
        )
    assert isinstance(_get_ssh_vendor(), SubprocessSSHVendor)
