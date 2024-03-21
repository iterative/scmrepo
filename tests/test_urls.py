import pytest

from scmrepo.urls import is_scp_style_url


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:iterative/scmrepo.git",
        "github.com:iterative/scmrepo.git",
        "user@github.com:iterative/scmrepo.git",
    ],
)
def test_scp_url(url: str):
    assert is_scp_style_url(url)


@pytest.mark.parametrize(
    "url",
    [
        r"C:\foo\bar",
        "C:/foo/bar",
        "/home/user/iterative/scmrepo/git",
        "~/iterative/scmrepo/git",
        "ssh://login@server.com:12345/repository.git",
        "https://user:password@github.com/iterative/scmrepo.git",
        "https://github.com/iterative/scmrepo.git",
    ],
)
def test_scp_url_invalid(url: str):
    assert not is_scp_style_url(url)
