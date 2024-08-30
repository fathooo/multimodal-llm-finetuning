# setup.py
from setuptools import setup, find_packages

setup(
    name="triton",
    version="0.0.1",
    packages=find_packages(include=['python', 'python.*']),
    include_package_data=True,
    description="Triton - Deep Learning Compiler",
    install_requires=[
        # Add any dependencies here if needed
    ],
    python_requires='>=3.6',
)