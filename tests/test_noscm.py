from collections.abc import Callable

import pytest
from pytest_test_utils import TmpDir
from pytest_test_utils.matchers import Matcher

from scmrepo.git import Git
from scmrepo.noscm import NoSCM


def test_noscm(tmp_dir: TmpDir):
    scm = NoSCM(tmp_dir)
    scm.add("test")


def test_noscm_raises_exc_on_unimplemented_apis(tmp_dir: TmpDir, matcher: Matcher):
    class Unimplemented(Exception):
        pass

    scm = NoSCM(tmp_dir, _raise_not_implemented_as=Unimplemented)
    assert scm._exc is Unimplemented  # pylint: disable=protected-access

    assert Git.reset == matcher.instance_of(Callable)
    with pytest.raises(Unimplemented):
        assert scm.reset()
