# scmrepo

[![Tests](https://github.com/iterative/scmrepo/workflows/Tests/badge.svg)](https://github.com/iterative/scmrepo/actions?workflow=Tests)

SCM wrapper and fsspec filesystem for Git for use in DVC

## Git File System

scmrepo provides [fsspec](https://filesystem-spec.readthedocs.io/)-based gitfs that provides fs-like API for your git repositories without having to `git checkout` them first. For example:

```python
from scmrepo.fs import GitFileSystem

fs = GitFileSystem("path/to/my/repo", rev="mybranch")

for root, dnames, fnames in fs.walk("path/in/repo"):
    for dname in dnames:
        print(fs.path.join(root, dname))

    for fname in fnames:
        print(fs.path.join(root, fname))

```

See [fsspec docs](https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=walk#fsspec.spec.AbstractFileSystem) for full list of available fs methods.
