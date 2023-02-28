# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os.path as osp

from setuptools import find_packages, setup

with open("README.md") as fh:
    long_description = fh.read()

exec(open(osp.join(osp.dirname(__file__), "cliffordlayers", "version.py")).read())

base_requires = [
    "torch",
    "pytest",
]

setup(
    name="cliffordlayers",
    version=__version__,
    description="A PyTorch library for Clifford layers",
    install_requires=base_requires,
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    author="Jayesh K. Gupta, Johannes Brandstetter, David Ruhe, and contributors",
    python_requires=">=3.6",
    project_urls={
        "Documentation": "https://microsoft.github.io/cliffordlayers",
        "Source code": "https://github.com/microsoft/cliffordlayers",
        "Bug tracker": "https://github.com/microsoft/cliffordlayers/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
