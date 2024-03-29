import os

from setuptools import find_packages, setup

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fcpy",
    version="0.2.0",
    description="ECMWF Field Compression Laboratory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ecmwf/field-compression",
    license="Apache 2.0",
    author="ECMWF",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "fcpy = fcpy.cli:main",
        ]
    },
    python_requires=">=3.8",
)
