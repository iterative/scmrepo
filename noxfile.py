"""Automation using nox.
"""
import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = "lint", "tests"
locations = "scmrepo", "tests"


@nox.session(python=["3.7", "3.8", "3.9", "3.10"])
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest")


@nox.session
def lint(session: nox.Session) -> None:
    session.install("pre-commit")
    session.install("-e", ".[dev]")

    if session.posargs:
        args = session.posargs + ["--all-files"]
    else:
        args = ["--all-files", "--show-diff-on-failure"]

    session.run("pre-commit", "run", *args)
    session.run("python", "-m", "mypy")
    session.run("python", "-m", "pylint", *locations)


@nox.session
def build(session: nox.Session) -> None:
    session.install("build", "setuptools")
    session.run("python", "-m", "build")
