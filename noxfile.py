import nox

nox.options.default_venv_backend = "uv"

_py_versions = range(11, 15)


def _sync(session, *groups, only=False):
    """Install the project and the named dependency groups into the session venv.

    Set only=True to install just the groups, omitting the project and its
    dependencies.
    """
    flag = "--only-group" if only else "--group"
    args = [] if only else ["--no-default-groups"]
    for group in groups:
        args.extend([flag, group])
    session.run_install(
        "uv",
        "sync",
        *args,
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test(session):
    _sync(session, "test")
    session.chdir("tests")
    session.run(
        "pytest",
        "-x",
        "-s",
        *session.posargs,  # propagates sys.argv to pytest
    )


@nox.session(python=["3.12"])
def test_dev(session):
    """runs against cogent3 develop branch"""
    _sync(session, "test")
    session.install(
        "cogent3 @ git+https://github.com/cogent3/cogent3.git@develop",
        "--no-cache-dir",
    )
    session.chdir("tests")
    session.run(
        "pytest",
        "-x",
        *session.posargs,  # propagates sys.argv to pytest
    )


@nox.session(python=[f"3.{v}" for v in _py_versions])
def htmlcov(session):
    _sync(session, "test")
    session.chdir("tests")
    session.run(
        "pytest",
        "-x",
        "--cov-report",
        "html",
        "--cov",
        "mdeq",
    )


@nox.session(python=None)
def fmt(session: nox.Session) -> None:
    _sync(session, "lint", only=True)
    session.run("ruff", "check", "--fix-only", ".")
    session.run("ruff", "format", ".")
