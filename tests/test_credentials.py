import os

import pytest

from scmrepo.git.credentials import (
    Credential,
    CredentialNotFoundError,
    GitCredentialHelper,
)


@pytest.fixture(name="helper")
def helper_fixture(mocker) -> "GitCredentialHelper":
    mocker.patch("shutil.which", return_value="/usr/bin/git-credential-foo")
    return GitCredentialHelper("foo")


def test_subprocess_get(helper, mocker):
    run = mocker.patch(
        "subprocess.run",
        return_value=mocker.Mock(
            stdout=os.linesep.join(
                ["protocol=https", "host=foo.com", "username=foo", "password=bar", ""]
            )
        ),
    )
    creds = helper.get(protocol="https", host="foo.com")
    assert run.call_args.args[0] == ["git-credential-foo", "get"]
    assert run.call_args.kwargs.get("input") == os.linesep.join(
        ["protocol=https", "host=foo.com", ""]
    )
    assert creds == Credential(url="https://foo:bar@foo.com")


def test_subprocess_get_failed(helper, mocker):
    from subprocess import CalledProcessError

    mocker.patch("subprocess.run", side_effect=CalledProcessError(1, "/usr/bin/foo"))
    with pytest.raises(CredentialNotFoundError):
        helper.get(protocol="https", host="foo.com")


def test_subprocess_get_no_output(helper, mocker):
    mocker.patch("subprocess.run", return_value=mocker.Mock(stdout=os.linesep))
    with pytest.raises(CredentialNotFoundError):
        helper.get(protocol="https", host="foo.com")


def test_subprocess_store(helper, mocker):
    run = mocker.patch("subprocess.run")
    helper.store(protocol="https", host="foo.com", username="foo", password="bar")
    assert run.call_args.args[0] == ["git-credential-foo", "store"]
    assert run.call_args.kwargs.get("input") == os.linesep.join(
        ["protocol=https", "host=foo.com", "username=foo", "password=bar", ""]
    )


def test_subprocess_erase(helper, mocker):
    run = mocker.patch("subprocess.run")
    helper.erase(protocol="https", host="foo.com")
    assert run.call_args.args[0] == ["git-credential-foo", "erase"]
    assert run.call_args.kwargs.get("input") == os.linesep.join(
        ["protocol=https", "host=foo.com", ""]
    )
