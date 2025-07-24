#!/usr/bin/env python3
"""
Setup script for AI From Scratch to Scale project.

This setup script enables proper installation of the shared packages
(utils, data_utils, engine, plotting) across the project, making them
importable from any model implementation.
"""

import re
from pathlib import Path

from setuptools import find_packages, setup


# Read version from __init__.py
def get_version():
    init_file = Path(__file__).parent / "utils" / "__init__.py"
    with open(init_file, "r") as f:
        content = f.read()
        version_match = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE
        )
        if version_match:
            return version_match.group(1)
    return "0.1.0"


# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Base dependencies for all packages
BASE_DEPENDENCIES = [
    "torch>=1.12.0",
    "numpy>=1.21.0",
    "pandas>=1.4.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "scikit-learn>=1.1.0",
    "tqdm>=4.64.0",
]

# Optional dependencies for enhanced features
OPTIONAL_DEPENDENCIES = {
    "wandb": ["wandb>=0.13.0"],
    "plotting": ["plotly>=5.0.0"],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "pylint>=3.0.0",
        "mypy>=0.991",
        "pre-commit>=2.20.0",
    ],
    "docs": ["sphinx>=5.0.0", "sphinx-rtd-theme>=1.0.0", "myst-parser>=0.18.0"],
}

# All optional dependencies combined
OPTIONAL_DEPENDENCIES["all"] = [
    dep
    for deps in OPTIONAL_DEPENDENCIES.values()
    for dep in deps
    if isinstance(deps, list)
]

setup(
    # Package metadata
    name="ai-from-scratch-to-scale",
    version=get_version(),
    description="Educational neural network implementations from basic perceptrons to modern architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Author information
    author="AI From Scratch to Scale Team",
    author_email="contact@ai-from-scratch.dev",
    url="https://github.com/ai-from-scratch/ai-from-scratch-to-scale",
    # Package configuration
    packages=find_packages(
        include=[
            "utils",
            "utils.*",
            "data_utils",
            "data_utils.*",
            "engine",
            "engine.*",
            "plotting",
            "plotting.*",
        ]
    ),
    python_requires=">=3.8",
    # Dependencies
    install_requires=BASE_DEPENDENCIES,
    extras_require=OPTIONAL_DEPENDENCIES,
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Keywords for discovery
    keywords=[
        "machine learning",
        "deep learning",
        "neural networks",
        "education",
        "tutorial",
        "pytorch",
        "perceptron",
        "mlp",
        "cnn",
        "rnn",
        "transformer",
        "from scratch",
    ],
    # Entry points for command line tools
    entry_points={
        "console_scripts": [
            "ai-validate=utils.validation:main",
        ]
    },
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
        "docs": ["*.md", "*.rst"],
        "models": ["*/requirements.txt", "*/README.md"],
    },
    # Project URLs
    project_urls={
        "Documentation": "https://ai-from-scratch.readthedocs.io/",
        "Source": "https://github.com/ai-from-scratch/ai-from-scratch-to-scale",
        "Tracker": "https://github.com/ai-from-scratch/ai-from-scratch-to-scale/issues",
        "Tutorial": "https://ai-from-scratch.dev/tutorial/",
    },
    # License
    license="MIT",
    # Development status
    zip_safe=False,
)
