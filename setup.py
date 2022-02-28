#!/usr/bin/env python

import os
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import find_packages, setup

spec = spec_from_file_location("pkginfo.version", "src/version.py")
pkginfo = module_from_spec(spec)
spec.loader.exec_module(pkginfo)


with open("README.md") as fd:
    README = fd.read()

with open("requirements-dev.txt") as fd:
    development_requirements = fd.readlines()

with open("requirements.txt") as fd:
    requirements = fd.readlines()


setup(
    name=pkginfo.__packagename__,
    version=pkginfo.__version__,
    description=pkginfo.__description__,
    # Meta information
    url="https://github.com/probinso/ACOio",
    author="Philip Robinson",
    author_email="probinso+acoio@protonmail.com",
    # package information
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=f">={pkginfo.__pythonbaseversion__}",
    classifiers=[
        # Language support
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish (should match 'license' above)
        "License :: OSI Approved :: MIT License",
        # Operating System
        "Operating System :: OS Independent",
    ],
    # Dependencies
    install_requires=requirements,
    extras_require={"dev": development_requirements},
    # Description
    long_description=README,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            f"{key} = {value}" for key, value in pkginfo.__scripts__.items()
        ]
    },  # noqa: E501
)
