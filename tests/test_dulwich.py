import pytest
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "algorithm", [b"ssh-rsa", b"rsa-sha2-256", b"rsa-sha2-512"]
)
def test_dulwich_github_compat(mocker: MockerFixture, algorithm: bytes):
    from asyncssh.misc import ProtocolError

    from scmrepo.git.backend.dulwich.asyncssh_vendor import (
        _process_public_key_ok_gh,
    )

    key_data = b"foo"
    auth = mocker.Mock(
        _keypair=mocker.Mock(algorithm=algorithm, public_data=key_data),
    )
    packet = mocker.Mock()

    with pytest.raises(ProtocolError):
        strings = iter((b"ed21556", key_data))
        packet.get_string = lambda: next(strings)
        _process_public_key_ok_gh(auth, None, None, packet)

    strings = iter((b"ssh-rsa", key_data))
    packet.get_string = lambda: next(strings)
    _process_public_key_ok_gh(auth, None, None, packet)
