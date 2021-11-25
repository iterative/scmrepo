import os
from unittest.mock import MagicMock

from scmrepo.git import Git
from scmrepo.progress import GitProgressEvent


class ANY:
    def __init__(self, expected_type):
        self.expected_type = expected_type

    def __repr__(self):
        return "Any" + self.expected_type.__name__.capitalize()

    def __eq__(self, other):
        return isinstance(other, self.expected_type)


def test_clone(tmp_path: os.PathLike):
    os.chdir(tmp_path)
    progress = MagicMock()
    url = "https://github.com/iterative/dvcyaml-schema"

    Git.clone(url, "dir", progress=progress)

    progress.assert_called_with(ANY(GitProgressEvent))
    assert (tmp_path / "dir").exists()
