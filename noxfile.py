import nox

_py_versions = range(10, 13)


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test(session):
    session.install(".[test]")
    session.chdir("tests")
    session.run(
        "pytest",
        "-x",
    )


@nox.session(python=["3.12"])
def test_dev(session):
    """runs against cogent3 develop branch"""
    session.install(".[test]")
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
    session.install(".[test]")
    session.install(".")
    session.chdir("tests")
    session.run(
        "pytest",
        "-x",
        "--cov-report",
        "html",
        "--cov",
        "mdeq",
    )
