from setuptools import setup, find_packages

setup(
    name="lm-local-evaluation",
    version="0.1.0",
    description="Angepasstes Evaluations-Framework für Sprachmodelle",
    author="Niklas Schwanitz",
    packages=find_packages(where="lm-evaluation-harness"),
    package_dir={"": "lm-evaluation-harness"},
    python_requires="==3.11.*",
)