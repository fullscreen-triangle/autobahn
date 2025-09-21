from setuptools import setup, find_packages

setup(
    name="scqp",
    version="1.0.0",
    description="S-Entropy Counterfactual Quantum Processor - Experimental Validation Package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kundai Farai Sachikonye",
    author_email="kundai.sachikonye@wzw.tum.de",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "sympy>=1.9.0",
        "numba>=0.54.0",
        "pytest>=6.0.0",
        "tqdm>=4.62.0",
        "plotly>=5.0.0",
        "opencv-python>=4.5.0",
        "librosa>=0.8.0",
        "nltk>=3.6.0",
        "transformers>=4.12.0",
        "torch>=1.10.0",
        "memory_profiler>=0.60.0",
        "psutil>=5.8.0"
    ],
    extras_require={
        "dev": [
            "black>=21.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "jupyter>=1.0.0"
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    keywords="consciousness quantum-computing s-entropy counterfactual-reasoning gas-molecular-processing"
)
