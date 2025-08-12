# setup.py
import setuptools

setuptools.setup(
    name="deep-lvpm",
    version="0.1",
    author="Alex James Ing",
    description="A package for carrying out deep latent variable path modelling",
    packages=setuptools.find_packages(),
    install_requires=[
        "pydot",
        "scikit-learn",
        "matplotlib",
        # NOTE: TensorFlow chosen via extras to avoid forcing GPU/CPU variant
    ],
    extras_require={
        "cpu": [
            "tensorflow==2.16.2",
        ],
        "gpu": [
            'tensorflow[and-cuda]==2.16.2',
        ],
        # Optional: Apple Silicon route (users often prefer this)
        "apple": [
            "tensorflow-macos==2.16.1",
            "tensorflow-metal==1.1.0",
        ],
    },
    package_data={
        "deep_lvpm.data": ["*.npz"],
    },
)
