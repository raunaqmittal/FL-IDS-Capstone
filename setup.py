from setuptools import setup, find_packages

setup(
    # Package metadata
    name="fl-ids",
    version="0.1.0",
    description=(
        "Robust Federated Learning Intrusion Detection System for IoT Edge Gateways. "
        "Implements a 3-part server-side Byzantine defense pipeline: "
        "Layer-Wise Cosine Similarity + MAD, Capped Simplex Projection, "
        "and EMA-based Momentum Trust Scoring."
    ),
    author="Raunaq Mittal",
    python_requires=">=3.10",

    # Automatically discover all packages under src/
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    # Runtime dependencies — mirrors requirements.txt (without dev/test extras)
    install_requires=[
        "flwr[simulation]>=1.8.0",
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "colorlog>=6.8.0",
    ],

    # Optional extras for development
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.29.0",
        ]
    },

    # CLI entry point
    entry_points={
        "console_scripts": [
            "fl-ids=app:main",  # Allows running `fl-ids` from the terminal
        ]
    },
)
