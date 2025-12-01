import pytest

from scmrepo.urls import is_scp_style_url


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:treeverse/scmrepo.git",
        "github.com:treeverse/scmrepo.git",
        "user@github.com:treeverse/scmrepo.git",
    ],
)
def test_scp_url(url: str):
    assert is_scp_style_url(url)


@pytest.mark.parametrize(
    "url",
    [
        r"C:\foo\bar",
        "C:/foo/bar",
        "/home/user/treeverse/scmrepo/git",
        "~/treeverse/scmrepo/git",
        "ssh://login@server.com:12345/repository.git",
        "https://user:password@github.com/treeverse/scmrepo.git",
        "https://github.com/treeverse/scmrepo.git",
    ],
)
def test_scp_url_invalid(url: str):
    assert not is_scp_style_url(url)
