# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:35:43 2018

@author: Payam Aminpour
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyFCM",
    version="0.0.1",
    author="Payam Aminpour",
    author_email="aminpour@msu.edu",
    description="A package for FCM scenario analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/payamaminpour/PyFCM",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ),
)