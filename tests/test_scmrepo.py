from typing import Type
from unittest.mock import MagicMock

from pytest_test_utils import TmpDir
from pytest_test_utils.matchers import Matcher

from scmrepo.git import Git
from scmrepo.progress import GitProgressEvent


def test_clone(tmp_dir: TmpDir, matcher: Type[Matcher]):
    progress = MagicMock()
    url = "https://github.com/iterative/dvcyaml-schema"
    rev = "cf279597596b54c5b0ce089eb4bda41ebbbb5db4"

    repo = Git.clone(url, "dir", rev=rev, progress=progress)
    assert repo.get_rev() == rev

    progress.assert_called_with(matcher.instance_of(GitProgressEvent))
    assert (tmp_dir / "dir").exists()
