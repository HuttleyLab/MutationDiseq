#!/usr/bin/env python
import pathlib
import sys

from setuptools import find_packages, setup

# REPLACE WITH YOUR NAME ETC...
__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]
__license__ = "n/a"
__version__ = "2021.06.28"
__maintainer__ = "Katherine Caley"
__email__ = "katherine.caley@anu.edu.au"
__status__ = "Development"

if sys.version_info < (3, 6):
    py_version = ".".join([str(n) for n in sys.version_info])
    raise RuntimeError(
        "Python-3.6 or greater is required, Python-%s used." % py_version
    )

# REPLACE WITH YOUR PROJECT DETAILS
PROJECT_URLS = {
    "Documentation": "https://github.com/GavinHuttley/KathLibrary",
    "Bug Tracker": "https://github.com/GavinHuttley/KathLibrary/issues",
    "Source Code": "https://github.com/GavinHuttley/KathLibrary",
}

# REPLACE WITH YOUR PROJECT NAME
short_description = "KathLibrary"

# REPLACE WITH YOUR PROJECT README NAME
readme_path = pathlib.Path(__file__).parent / "README.md"

long_description = readme_path.read_text()

PACKAGE_DIR = "src"

setup(
    name="KathLibrary",  # REPLACE WITH YOUR PROJECT NAME
    version=__version__,
    author="Katherine Caley",
    author_email="katherine.caley@anu.edu.au",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/x-rst",  # change if it's in markdown format
    platforms=["any"],
    license=__license__,
    keywords=["science"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(where="src"),
    package_dir={"": PACKAGE_DIR},
    url="https://github.com/GavinHuttley/KathLibrary",  # REPLACE WITH YOUR PROJECT DETAILS
    project_urls=PROJECT_URLS,
    # REPLACE WITH YOUR PROJECT DEPENDENCIES
    install_requires=["numpy", "cogent3", "click", "accupy", "scipy"],
    # THE FOLLOWING BECOMES A COMMAND LINE SCRIPT
    entry_points={
        "console_scripts": [
            "demo_cli=kath_library.demo:main"  # cli_name=python_package.module_name:function_name
        ]
    },
)
