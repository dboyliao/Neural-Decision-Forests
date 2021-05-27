from setuptools import find_packages, setup

setup(
    name="neural-decision-forest",
    packages=find_packages(),
    install_requires=["torch", "torchvision", "scikit-learn"],
)
