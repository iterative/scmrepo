import pytest
from pytest_test_utils import TmpDir

from scmrepo.git import Git


@pytest.fixture
def scm(tmp_dir: TmpDir):
    git_ = Git.init(tmp_dir)
    yield git_
    git_.close()
