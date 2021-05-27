from pathlib import Path

from setuptools import find_packages, setup

with (Path(__file__).parent / "neural_decision_forest" / "_version.py").open(
    "r"
) as fid:
    scope = {}
    exec(fid.read(), scope)
    _version = scope["__version__"]

setup(
    name="neural-decision-forest",
    packages=find_packages(),
    install_requires=["torch", "torchvision", "scikit-learn"],
    version=_version,
)
