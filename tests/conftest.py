import asyncio
import os
import subprocess
import sys
from typing import Any, AsyncIterator, Dict, Iterator

import asyncssh
import pygit2
import pytest
from pytest_test_utils import TempDirFactory, TmpDir

from scmrepo.git import Git

TEST_SSH_USER = "user"
TEST_SSH_KEY_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), f"{TEST_SSH_USER}.key"
)

# pylint: disable=redefined-outer-name


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def isolate(tmp_dir_factory: TempDirFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_dir_factory.mktemp("mock")
    home_dir = path / "home"
    home_dir.mkdir()

    if sys.platform == "win32":
        home_drive, home_path = os.path.splitdrive(home_dir)
        monkeypatch.setenv("USERPROFILE", str(home_dir))
        monkeypatch.setenv("HOMEDRIVE", home_drive)
        monkeypatch.setenv("HOMEPATH", home_path)
    else:
        monkeypatch.setenv("HOME", str(home_dir))

    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    contents = b"""
[user]
name=DVC Tester
email=dvctester@example.com
[init]
defaultBranch=master
"""
    (home_dir / ".gitconfig").write_bytes(contents)
    pygit2.settings.search_path[pygit2.GIT_CONFIG_LEVEL_GLOBAL] = str(home_dir)


@pytest.fixture
def scm(tmp_dir: TmpDir) -> Iterator[Git]:
    git_ = Git.init(tmp_dir)
    sig = git_.pygit2.default_signature

    assert sig.email == "dvctester@example.com"
    assert sig.name == "DVC Tester"

    yield git_
    git_.close()


backends = ["gitpython", "dulwich", "pygit2"]


@pytest.fixture(params=backends)
def git_backend(request) -> str:
    marker = request.node.get_closest_marker("skip_git_backend")
    to_skip = marker.args if marker else []

    backend = request.param
    if backend in to_skip:
        pytest.skip()
    return backend


@pytest.fixture
def git(tmp_dir: TmpDir, git_backend: str) -> Iterator[Git]:
    git_ = Git(tmp_dir, backends=[git_backend])
    yield git_
    git_.close()


@pytest.fixture
def remote_git_dir(tmp_dir_factory: TempDirFactory):
    git_dir = tmp_dir_factory.mktemp("git-remote")
    remote_git = Git.init(git_dir)
    remote_git.close()
    return git_dir


@pytest.fixture(scope="session")
def docker(request: pytest.FixtureRequest):
    for cmd in [("docker", "ps"), ("docker-compose", "version")]:
        try:
            subprocess.check_call(
                cmd,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, OSError):
            pytest.skip(f"no {cmd[0]} installed")

    if "CI" in os.environ and os.name == "nt":
        pytest.skip("disabled for Windows on Github Actions")

    pytest.importorskip("pytest_docker")
    yield request.getfixturevalue("docker_services")


@pytest.fixture
def ssh_conn_info(
    docker,  # pylint: disable=unused-argument
) -> Dict[str, Any]:
    conn_info = {
        "host": "127.0.0.1",
        "port": docker.port_for("git-server", 2222),
        "client_keys": TEST_SSH_KEY_PATH,
        "known_hosts": None,
        "username": TEST_SSH_USER,
    }

    async def _check() -> bool:
        try:
            async with asyncssh.connect(**conn_info) as conn:
                result = await conn.run("git --version")
                assert result.returncode == 0
                async with conn.start_sftp_client() as sftp:
                    assert await sftp.exists("/")
        except Exception:  # pylint: disable=broad-except
            return False
        return True

    def check() -> bool:
        return asyncio.run(_check())

    docker.wait_until_responsive(timeout=30.0, pause=1, check=check)
    return conn_info


@pytest.fixture
async def ssh_connection(
    ssh_conn_info: Dict[str, Any],
) -> AsyncIterator[asyncssh.connection.SSHClientConnection]:
    async with asyncssh.connect(**ssh_conn_info) as conn:
        yield conn


@pytest.fixture
async def sftp(
    ssh_connection: asyncssh.connection.SSHClientConnection,
) -> AsyncIterator[asyncssh.SFTPClient]:
    async with ssh_connection.start_sftp_client() as sftp:
        yield sftp
