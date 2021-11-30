from typing import Type
from unittest.mock import MagicMock

from pytest_test_utils import TmpDir
from pytest_test_utils.matchers import Matcher

from scmrepo.git import Git
from scmrepo.progress import GitProgressEvent


def test_clone(tmp_dir: TmpDir, matcher: Type[Matcher]):
    progress = MagicMock()
    url = "https://github.com/iterative/dvcyaml-schema"

    Git.clone(url, "dir", progress=progress)

    progress.assert_called_with(matcher.instance_of(GitProgressEvent))
    assert (tmp_dir / "dir").exists()
