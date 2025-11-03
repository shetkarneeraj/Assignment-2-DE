"""Setup configuration for electricity_pipeline package."""
from setuptools import setup, find_packages

setup(
    name="electricity_pipeline",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.10",
)
