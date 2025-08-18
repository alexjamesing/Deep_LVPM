# import setuptools

# """ This file makes it possible to install the DLVPM package using the pip package manager """


# setuptools.setup(
#     name="deep-lvpm",
#     version="0.1.1",
#     author="Alex James Ing",
#     description="A package for carrying out deep latent variable path modeĺling",
#     packages=setuptools.find_packages(),
#     install_requires=[
#         'tensorflow==2.16.2',
#         'pydot',
#         'scikit-learn',
#         'matplotlib'
#     ]
# )

# pyproject.toml
import setuptools

# Keras 3 (multi-backend) compatible packaging for deep-lvpm
# Choose ONE backend at install time via extras:
#   pip install .[tf]     or .[torch]     or .[jax]

setuptools.setup(
    name="deep-lvpm",
    version="0.2.0",
    author="Alex James Ing",
    description="Deep Latent Variable Path Modelling (Keras 3, multi-backend)",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "keras>=3.0.0",      # Keras Core (multi-backend)
        "pydot",
        "scikit-learn",
        "matplotlib",
    ],
    extras_require={
        "tf":    ["tensorflow>=2.15"],
        "torch": ["torch>=2.2"],
        "jax":   ["jax>=0.4.20", "jaxlib>=0.4.20"],
    },
)
