import nox


dependencies = (
    "blosc2",
    "cogent3",
    "click",
    "accupy",
    "numpy",
    "scipy",
    "rich",
    "scitrack",
    "pytest",
    "pytest-xdist",
    "pytest-cov",
)

_py_versions = range(10, 11)


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test(session):
    py_version = session.python.replace(".", "")
    session.install(*dependencies)
    session.install(".")
    session.chdir("tests")
    session.run(
        "pytest",
        # "-n",
        # "auto",
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
    session.install(*dependencies)
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
