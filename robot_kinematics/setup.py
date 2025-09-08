#!/usr/bin/env python3
"""
Setup script for robot_kinematics package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Robot Kinematics Package for 6-DOF Manipulator"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="robot_kinematics",
    version="1.0.0",
    author="Robot Control Team",
    author_email="robot@company.com",
    description="Production-ready kinematics library for 6-DOF robot manipulators",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/company/robot_core_control",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Robotics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "visualization": [
            "matplotlib>=3.3",
            "plotly>=5.0",
        ],
    },
    include_package_data=True,
    package_data={
        "robot_kinematics": [
            "config/*.yaml",
            "data/*.json",
        ],
    },
    entry_points={
        "console_scripts": [
            "robot-kinematics-demo=robot_kinematics.examples.main:main",
            "robot-kinematics-validate=robot_kinematics.src.kinematics_validation:main",
        ],
    },
)

