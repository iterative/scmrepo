import pygit2
import pytest
from pytest_test_utils import TmpDir

from scmrepo.git import Git
from scmrepo.git.backend.pygit2 import Pygit2Backend


@pytest.mark.parametrize("use_sha", [True, False])
def test_pygit_resolve_refish(tmp_dir: TmpDir, scm: Git, use_sha: str):
    backend = Pygit2Backend(tmp_dir)
    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="foo")
    head = scm.get_rev()
    tag = "my_tag"
    scm.gitpython.git.tag("-a", tag, "-m", "create annotated tag")

    if use_sha:
        # refish will be annotated tag SHA (not commit SHA)
        ref = backend.repo.references.get(f"refs/tags/{tag}")
        refish = str(ref.target)
    else:
        refish = tag

    assert refish != head
    commit, ref = backend._resolve_refish(  # pylint: disable=protected-access
        refish
    )
    assert isinstance(commit, pygit2.Commit)
    assert str(commit.id) == head
    if not use_sha:
        assert ref.name == f"refs/tags/{tag}"
