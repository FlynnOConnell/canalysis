#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# setup.py

Module: Package information.
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = "0.0.1"

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [
    x.strip().replace("git+", "") for x in all_reqs if x.startswith("git+")
]

setup(
    name="canalysis",
    version=__version__,
    description="Calcium Imaging trace analysis with statistics and machine learning.",
    license="MIT",
    packages=find_packages(),
    package_dir={"canalysis": "canalysis"},
    include_package_data=True,
    author="Flynn OConnell",
    zip_safe=False,
    author_email="oconnell@binghatmon.edu",
)
