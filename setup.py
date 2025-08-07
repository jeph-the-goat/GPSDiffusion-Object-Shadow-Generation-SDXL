# setup.py
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="gpsdiffusion",
    version="0.1.0",
    description="Geometry-Prior-guided Shadow Diffusion model.",
    packages=find_packages(),
    install_requires=requirements,
)