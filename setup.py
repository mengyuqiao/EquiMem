from setuptools import setup, find_packages

setup(
    name="equimem",
    version="0.1.0",
    description="Zero-Trust Memory Calibration for Multi-Agent LLM Debate",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "torch>=2.1",
        "transformers>=4.40",
        "sentence-transformers>=2.7",
        "networkx>=3.2",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "ollama>=0.2",
    ],
)
