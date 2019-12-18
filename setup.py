'''
    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FDR_SGD",
    version="1.0.0",
    author="Sho Yaida",
    author_email="shoyaida@fb.com",
    description="Learning-rate scheduler in PyTorch, based on a fluctuation-dissipation relation for SGD.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/FDR_scheduler",
    packages=['FDR_SGD'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
