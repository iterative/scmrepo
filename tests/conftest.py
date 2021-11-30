import pytest
from pytest_test_utils import TmpDir

from scmrepo.git import Git


@pytest.fixture
def scm(tmp_dir: TmpDir, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "DVC test user")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "dvctester@example.com")

    git_ = Git.init(tmp_dir)
    yield git_
    git_.close()
