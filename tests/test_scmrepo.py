from unittest.mock import MagicMock

from scmrepo.git import Git
from scmrepo.progress import GitProgressEvent


def test_clone(tmp_dir: TmpDir):
    progress = MagicMock()
    url = "https://github.com/iterative/dvcyaml-schema"

    Git.clone(url, "dir", progress=progress)

    progress.assert_called_with(ANY(GitProgressEvent))
    assert (tmp_dir / "dir").exists()
