import pygit2
import pytest
from pytest_mock import MockerFixture
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
    commit, ref = backend._resolve_refish(refish)  # pylint: disable=protected-access
    assert isinstance(commit, pygit2.Commit)
    assert str(commit.id) == head
    if not use_sha:
        assert ref.name == f"refs/tags/{tag}"


@pytest.mark.parametrize("skip_conflicts", [True, False])
def test_pygit_stash_apply_conflicts(
    tmp_dir: TmpDir, scm: Git, skip_conflicts: bool, mocker: MockerFixture
):
    from pygit2 import GIT_CHECKOUT_ALLOW_CONFLICTS

    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="foo")
    tmp_dir.gen("foo", "bar")
    scm.stash.push()
    rev = scm.resolve_rev(r"stash@{0}")

    backend = Pygit2Backend(tmp_dir)
    mock = mocker.patch.object(backend.repo, "stash_apply")
    backend._stash_apply(  # pylint: disable=protected-access
        rev, skip_conflicts=skip_conflicts
    )
    expected_strategy = (
        backend._get_checkout_strategy()  # pylint: disable=protected-access
    )
    if skip_conflicts:
        expected_strategy |= GIT_CHECKOUT_ALLOW_CONFLICTS
    mock.assert_called_once_with(
        0,
        strategy=expected_strategy,
        reinstate_index=False,
    )
