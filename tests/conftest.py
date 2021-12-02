import os
import sys

import pygit2
import pytest
from pytest_test_utils import TempDirFactory, TmpDir

from scmrepo.git import Git


@pytest.fixture(autouse=True)
def isolate(tmp_dir_factory: TempDirFactory, monkeypatch: pytest.MonkeyPatch):
    path = tmp_dir_factory.mktemp("mock")
    home_dir = path / "home"
    home_dir.mkdir()

    if sys.platform == "win32":
        home_drive, home_path = os.path.splitdrive(home_dir)
        monkeypatch.setenv("USERPROFILE", str(home_dir))
        monkeypatch.setenv("HOMEDRIVE", home_drive)
        monkeypatch.setenv("HOMEPATH", home_path)
    else:
        monkeypatch.setenv("HOME", str(home_dir))

    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    contents = b"""
[user]
name=DVC Tester
email=dvctester@example.com
[init]
defaultBranch=master
"""
    (home_dir / ".gitconfig").write_bytes(contents)
    pygit2.settings.search_path[pygit2.GIT_CONFIG_LEVEL_GLOBAL] = str(home_dir)


@pytest.fixture
def scm(tmp_dir: TmpDir):
    git_ = Git.init(tmp_dir)
    sig = git_.pygit2.default_signature

    assert sig.email == "dvctester@example.com"
    assert sig.name == "DVC Tester"

    yield git_
    git_.close()
