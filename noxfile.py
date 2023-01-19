import nox


_py_versions = range(10, 11)


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test(session):
    session.install(".[test]")
    session.install(".")
    session.chdir("tests")
    session.run(
        "pytest",
        "-x",
        "--junitxml",
        f"junit-{py_version}.xml",
        "--cov-report",
        "xml",
        "--cov",
        "mdeq",
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
