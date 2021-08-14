#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name='jmlm',
    version='0.1',
    description="Japanese Masked Language Model Scoring",
    author='Pham Quang Nhat Minh',
    packages=find_packages('src'),
    package_dir={'': 'src'},

    install_requires=[
        'transformers~=4.6.1',
        'unidic_lite',
        'fugashi',
        'ipadic'
    ],

    extras_require={
    },

    # Needed for static type checking
    # https://mypy.readthedocs.io/en/latest/installed_packages.html
    zip_safe=False
)
