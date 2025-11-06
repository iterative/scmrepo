# pylint: disable=unused-argument
import os

import pygit2
import pytest
from pytest_mock import MockerFixture
from pytest_test_utils import TmpDir

from scmrepo.exceptions import SCMError
from scmrepo.git import Git
from scmrepo.git.backend.pygit2 import Pygit2Backend


@pytest.mark.parametrize("use_sha", [True, False])
def test_pygit_resolve_refish(tmp_dir: TmpDir, scm: Git, use_sha: str):
    backend = Pygit2Backend(tmp_dir)
    tmp_dir.gen("foo", "foo")
    scm.add_commit("foo", message="foo")
    head = scm.get_rev()
    tag = "my_tag"
    scm.tag(tag, annotated=True, message="create annotated tag")

    if use_sha:
        # refish will be annotated tag SHA (not commit SHA)
        _ref = backend.repo.references.get(f"refs/tags/{tag}")
        assert _ref
        refish = str(_ref.target)
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
    from pygit2 import GIT_CHECKOUT_ALLOW_CONFLICTS  # type: ignore[attr-defined]

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


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:iterative/scmrepo.git",
        "github.com:iterative/scmrepo.git",
        "user@github.com:iterative/scmrepo.git",
        "ssh://login@server.com:12345/repository.git",
    ],
)
def test_pygit_ssh_error(tmp_dir: TmpDir, scm: Git, url):
    backend = Pygit2Backend(tmp_dir)
    with pytest.raises(NotImplementedError):
        with backend._get_remote(url):  # pylint: disable=protected-access
            pass


@pytest.mark.parametrize("name", ["committer", "author"])
def test_pygit_use_env_vars_for_signature(
    tmp_dir: TmpDir, mocker: MockerFixture, name: str
):
    from pygit2 import Signature

    mocker.patch(
        "scmrepo.git.Pygit2Backend.default_signature",
        new=mocker.PropertyMock(side_effect=SCMError),
    )
    git = Git.init(tmp_dir)
    with pytest.raises(SCMError):
        _ = git.pygit2.default_signature  # pylint: disable=W0104

    # Make sure that the environment variables are not set to not interfere with
    # with the check below
    for var in [f"GIT_{name.upper()}_NAME", f"GIT_{name.upper()}_EMAIL"]:
        assert os.environ.get(var, None) is None

    # Basic expected behavior if vars are not set. Another sanity check
    with pytest.raises(SCMError):
        getattr(git.pygit2, name)

    mocker.patch.dict(os.environ, {f"GIT_{name.upper()}_EMAIL": "olivaw@iterative.ai"})
    with pytest.raises(SCMError):
        getattr(git.pygit2, name)

    mocker.patch.dict(os.environ, {f"GIT_{name.upper()}_NAME": "R. Daneel Olivaw"})
    assert getattr(git.pygit2, name) == Signature(
        email="olivaw@iterative.ai", name="R. Daneel Olivaw"
    )
