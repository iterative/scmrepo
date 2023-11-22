import io
import os

import pytest

from scmrepo.git.credentials import (
    Credential,
    CredentialNotFoundError,
    GitCredentialHelper,
    MemoryCredentialHelper,
)


@pytest.fixture(name="git_helper")
def git_helper_fixture(mocker) -> "GitCredentialHelper":
    mocker.patch("shutil.which", return_value="/usr/bin/git-credential-foo")
    return GitCredentialHelper("foo")


def test_subprocess_get(git_helper, mocker):
    run = mocker.patch(
        "subprocess.run",
        return_value=mocker.Mock(
            stdout="\n".join(
                ["protocol=https", "host=foo.com", "username=foo", "password=bar", ""]
            )
        ),
    )
    creds = git_helper.get(Credential(protocol="https", host="foo.com", path="foo.git"))
    assert run.call_args.args[0] == ["git-credential-foo", "get"]
    assert run.call_args.kwargs.get("input") == "\n".join(
        ["protocol=https", "host=foo.com", ""]
    )
    assert creds == Credential(url="https://foo:bar@foo.com")


def test_subprocess_get_use_http_path(git_helper, mocker):
    git_helper.use_http_path = True
    run = mocker.patch(
        "subprocess.run",
        return_value=mocker.Mock(
            stdout="\n".join(["username=foo", "password=bar", ""])
        ),
    )
    creds = git_helper.get(Credential(protocol="https", host="foo.com", path="foo.git"))
    assert run.call_args.args[0] == ["git-credential-foo", "get"]
    assert run.call_args.kwargs.get("input") == "\n".join(
        ["protocol=https", "host=foo.com", "path=foo.git", ""]
    )
    assert creds == Credential(username="foo", password="bar")


def test_subprocess_get_failed(git_helper, mocker):
    from subprocess import CalledProcessError

    mocker.patch("subprocess.run", side_effect=CalledProcessError(1, "/usr/bin/foo"))
    with pytest.raises(CredentialNotFoundError):
        git_helper.get(Credential(protocol="https", host="foo.com"))


def test_subprocess_get_no_output(git_helper, mocker):
    mocker.patch("subprocess.run", return_value=mocker.Mock(stdout="\n"))
    with pytest.raises(CredentialNotFoundError):
        git_helper.get(Credential(protocol="https", host="foo.com"))


def test_subprocess_store(git_helper, mocker):
    run = mocker.patch("subprocess.run")
    git_helper.store(
        Credential(protocol="https", host="foo.com", username="foo", password="bar")
    )
    assert run.call_args.args[0] == ["git-credential-foo", "store"]
    assert run.call_args.kwargs.get("input") == "\n".join(
        ["protocol=https", "host=foo.com", "username=foo", "password=bar", ""]
    )


def test_subprocess_erase(git_helper, mocker):
    run = mocker.patch("subprocess.run")
    git_helper.erase(Credential(protocol="https", host="foo.com"))
    assert run.call_args.args[0] == ["git-credential-foo", "erase"]
    assert run.call_args.kwargs.get("input") == "\n".join(
        ["protocol=https", "host=foo.com", ""]
    )


def test_memory_helper_get(mocker):
    from getpass import getpass

    from scmrepo.git.credentials import _input_tty

    helper = MemoryCredentialHelper()
    expected = Credential(
        protocol="https", host="foo.com", username="foo", password="bar"
    )
    get_interactive = mocker.patch.object(
        helper,
        "_get_interactive",
        return_value=expected,
    )
    with pytest.raises(CredentialNotFoundError):
        helper.get(Credential(protocol="https", host="foo.com"), interactive=False)
    get_interactive.assert_not_called()
    assert (
        helper.get(Credential(protocol="https", host="foo.com"), interactive=True)
        == expected
    )
    assert get_interactive.call_args.args[0] == Credential(
        protocol="https", host="foo.com"
    )
    assert get_interactive.call_args.args[1:] == (_input_tty, getpass)


def test_memory_helper_get_cached(mocker):
    helper = MemoryCredentialHelper()
    expected = Credential(
        protocol="https", host="foo.com", username="foo", password="bar"
    )
    helper[expected] = expected

    get_interactive = mocker.patch.object(
        helper,
        "_get_interactive",
        return_value=expected,
    )
    assert (
        helper.get(Credential(protocol="https", host="foo.com"), interactive=False)
        == expected
    )
    get_interactive.assert_not_called()


def test_memory_helper_prompt_disabled(mocker):
    helper = MemoryCredentialHelper()
    get_interactive = mocker.patch.object(
        helper,
        "_get_interactive",
    )
    mocker.patch.dict(os.environ, {"GIT_TERMINAL_PROMPT": "0"})
    with pytest.raises(CredentialNotFoundError):
        helper.get(Credential(protocol="https", host="foo.com"), interactive=True)
    get_interactive.assert_not_called()


def test_memory_helper_prompt_askpass(mocker):
    helper = MemoryCredentialHelper()
    mocker.patch.dict(os.environ, {"GIT_ASKPASS": "/usr/local/bin/my-askpass"})
    run = mocker.patch(
        "subprocess.run",
        side_effect=[
            mocker.Mock(stdout="foo"),
            mocker.Mock(stdout="\n".join(["bar", ""])),
        ],
    )
    expected = Credential(
        protocol="https", host="foo.com", username="foo", password="bar"
    )
    assert (
        helper.get(Credential(protocol="https", host="foo.com"), interactive=True)
        == expected
    )
    assert len(run.call_args_list) == 2
    assert run.call_args_list[0].args[0] == [
        "/usr/local/bin/my-askpass",
        "Username for 'https://foo.com': ",
    ]
    assert run.call_args_list[1].args[0] == [
        "/usr/local/bin/my-askpass",
        "Password for 'https://foo@foo.com': ",
    ]


def test_get_matching_commands():
    from dulwich.config import ConfigFile

    config_file = io.BytesIO(
        b"""
[credential]
    helper = /usr/local/bin/my-helper
    UseHttpPath = true
"""
    )
    config_file.seek(0)
    config = ConfigFile.from_file(config_file)
    assert list(
        GitCredentialHelper.get_matching_commands("https://foo.com/foo.git", config)
    ) == [("/usr/local/bin/my-helper", True)]

    config_file = io.BytesIO(
        b"""
[credential]
    helper = /usr/local/bin/my-helper
"""
    )
    config_file.seek(0)
    config = ConfigFile.from_file(config_file)
    assert list(
        GitCredentialHelper.get_matching_commands("https://foo.com/foo.git", config)
    ) == [("/usr/local/bin/my-helper", False)]

    config_file = io.BytesIO(
        b"""
[credential]
    helper =
"""
    )
    config_file.seek(0)
    config = ConfigFile.from_file(config_file)
    assert (
        list(
            GitCredentialHelper.get_matching_commands("https://foo.com/foo.git", config)
        )
        == []
    )
