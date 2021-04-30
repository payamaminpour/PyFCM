# -*- coding: utf-8 -*-
"""
Created on Fri Apr  30 11:55:30 2021

@author: Corey White
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyFCM",
    version="1.0.0",
    author="Corey White",
    author_email="ctwhite@ncsu.edu",
    description="A package for FCM scenario analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomorrownow/PyFCM",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ),
)
