#!/usr/bin/env python3
"""
Setup script for robot_planning package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="robot_planning",
    version="1.0.0",
    author="Robot Planning Team",
    author_email="team@robotplanning.com",
    description="Advanced motion planning library for robot manipulators using AORRTC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robotplanning/robot_planning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "robot-planning-demo=robot_planning.examples.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "robot_planning": ["config/*.yaml"],
    },
)

