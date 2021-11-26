import pytest
from pytest_test_utils import TmpDir

from scmrepo.git import Git


@pytest.mark.parametrize("backend", ["gitpython", "dulwich", "pygit2"])
def test_git_init(tmp_dir: TmpDir, backend: str):
    Git.init(".", _backend=backend)
    assert (tmp_dir / ".git").is_dir()
    Git(tmp_dir)


@pytest.mark.parametrize("backend", ["gitpython", "dulwich", "pygit2"])
def test_git_init_bare(tmp_dir: TmpDir, backend: str):
    Git.init(".", bare=True, _backend=backend)
    assert list(tmp_dir.iterdir())
    Git(tmp_dir)
