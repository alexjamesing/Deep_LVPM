import setuptools

""" This file makes it possible to install the DLVPM package using the pip package manager """


setuptools.setup(
    name="deep-lvpm",
    version="0.1",
    author="Alex James Ing",
    description="A package for carrying out deep latent variable path modeĺling",
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow',
        'pydot',
        'scikit-learn',
        'matplotlib'
    ]
)