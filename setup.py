from setuptools import setup, find_packages

setup(
    name="synth",
    version="0.1.0",
    description="Production-ready synthetic data simulation system",
    author="Synth Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "faker>=9.0.0",
        "pyyaml>=6.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "prometheus-client>=0.11.0",
        "confluent-kafka>=1.8.0",
        "pytest>=6.2.5",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
    ],
    python_requires=">=3.8",
)