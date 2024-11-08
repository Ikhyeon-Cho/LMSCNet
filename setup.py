# setup.py

from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="LMSCNet",
    version="0.1",
    author="Ikhyeon Cho",
    author_email="tre0430@korea.ac.kr",
    url="https://github.com/Ikhyeon-Cho/LMSCNet",
    packages=find_packages(),
    install_requires=[],  # requirements.txt
    description="Learning Multi-modal Semantic Context for 3D Semantic Segmentation",
    python_requires='>=3.6',
    include_package_data=True,
)
