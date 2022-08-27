scmrepo
=======

|PyPI| |Status| |Python Version| |License|

|Tests| |Codecov| |pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/scmrepo.svg
   :target: https://pypi.org/project/scmrepo/
   :alt: PyPI
.. |Status| image:: https://img.shields.io/pypi/status/scmrepo.svg
   :target: https://pypi.org/project/scmrepo/
   :alt: Status
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/scmrepo
   :target: https://pypi.org/project/scmrepo
   :alt: Python Version
.. |License| image:: https://img.shields.io/pypi/l/scmrepo
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License
.. |Tests| image:: https://github.com/iterative/scmrepo/workflows/Tests/badge.svg
   :target: https://github.com/iterative/scmrepo/actions?workflow=Tests
   :alt: Tests
.. |Codecov| image:: https://codecov.io/gh/iterative/scmrepo/branch/main/graph/badge.svg
   :target: https://app.codecov.io/gh/iterative/scmrepo
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black


SCM wrapper and fsspec filesystem for Git for use in DVC.


Features
--------

* Works with multiple backends: ``pygit2``, ``dulwich`` and ``gitpython``.
* Provides fsspec filesystem over Git: ``GitFileSystem``.


See `fsspec docs`_ for full list of available fs methods.


Requirements
------------


Installation
------------

You can install *scmrepo* via pip_ from PyPI_:

.. code:: console

   $ pip install scmrepo


Usage
-----

Git File System
^^^^^^^^^^^^^^^

scmrepo provides `fsspec`_ based gitfs that provides fs-like API for your git
repositories without having to ``git checkout`` them first. For example:

.. code-block:: python

    from scmrepo.fs import GitFileSystem

    fs = GitFileSystem("path/to/my/repo", rev="mybranch")

    for root, dnames, fnames in fs.walk("path/in/repo"):
        for dname in dnames:
            print(fs.path.join(root, dname))

        for fname in fnames:
            print(fs.path.join(root, fname))


Contributing
------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


License
-------

Distributed under the terms of the `Apache 2.0 license`_,
*scmrepo* is free and open source software.


Issues
------

If you encounter any problems,
please `file an issue`_ along with a detailed description.


.. _Apache 2.0 license: https://opensource.org/licenses/Apache-2.0
.. _fsspec: https://filesystem-spec.readthedocs.io/
.. _fsspec docs: https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=walk#fsspec.spec.AbstractFileSystem
.. _PyPI: https://pypi.org/
.. _file an issue: https://github.com/iterative/scmrepo/issues
.. _pip: https://pip.pypa.io/
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
