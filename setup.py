# setup.py
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="gpsdiffusion",
    version="0.1.0",
    description="Geometry-Prior-guided Shadow Diffusion model.",
    packages=find_packages(),
    py_modules=[
        "train_GPSDiffusion_sdxl",
        "test_GPSDiffusion_sdxl", 
        "base_network",
        "attention_processor",
        "post_processing"
    ],
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'gpsdiffusion-train=train_GPSDiffusion_sdxl:main',
            'gpsdiffusion-test=test_GPSDiffusion_sdxl:main',
        ],
    },
)