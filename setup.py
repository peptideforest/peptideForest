#!/usr/bin/env python3
from setuptools import setup
import os

version_path = os.path.join(os.path.dirname(__file__), "peptide_forest", "version.txt")

with open(version_path, "r") as version_file:
    peptide_forest_version = version_file.read().strip()

with open("requirements.txt") as req_file:
    reqs = req_file.readlines()

setup(
    name="peptide_forest",
    version=peptide_forest_version,
    packages=[
        "peptide_forest",
    ],
    python_requires=">=3.8.0",
    install_requires=reqs,
    description="Integrate search engines",
    long_description="Integrating multiple search engines for peptide identification",
    author="T. Ranff, M. Dennison, J. Bédorf, S. Schulze, N. Zinn, M. Bantscheff, J.J.R.M. van Heugten, C. Fufezan",
    url="http://github.com/fu",
    license="The MIT license",
    platforms="any that supports python 3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
