import pytest
from pytest_mock import MockerFixture
from pytest_test_utils import TmpDir

from scmrepo.git import Git
from scmrepo.git.backend.dulwich import DulwichBackend


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


def test_dulwich_get_ref(tmp_dir: TmpDir, scm: Git):
    backend = DulwichBackend(tmp_dir)
    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="foo")

    scm.set_ref("refs/exp/base", "refs/heads/master")

    custom_ref = backend.get_ref("refs/exp/base", follow=False)
    assert str(custom_ref).rstrip("\n") == "refs/heads/master"

    head = scm.get_rev()
    tag = "my_tag"
    scm.tag(["-a", tag, "-m", "Annotated Tag"])

    tag_ref = backend.get_ref(f"refs/tags/{tag}", follow=False)
    assert head == tag_ref
