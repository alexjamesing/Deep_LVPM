import setuptools

setuptools.setup(
    name="deep-lvpm",
    version="0.2.3",
    author="Alex James Ing",
    description="Deep Latent Variable Path Modelling (Keras 3, multi-backend)",
    packages=setuptools.find_packages(),
    python_requires=">=3.11,<3.13",
    install_requires=[
        "keras==3.14.0",
        "pydot==4.0.1",
        "scikit-learn==1.6.1",
        "matplotlib==3.9.4",
    ],
    extras_require={
        # ---------------- TensorFlow ----------------
        "tf-cpu": [
            "tensorflow==2.16.2",
        ],
        # Linux-only; bundles CUDA/cuDNN runtime wheels
        "tf-gpu": [
            'tensorflow[and-cuda]==2.16.2; platform_system=="Linux"',
        ],
        # Apple Silicon (macOS arm64) via Metal
        "tf-apple": [
            'tensorflow-macos==2.16.2; platform_system=="Darwin" and platform_machine=="arm64"',
            'tensorflow-metal==1.2.0; platform_system=="Darwin" and platform_machine=="arm64"',
        ],

        # ---------------- PyTorch -------------------
        # CPU on Linux/Windows/macOS Intel; MPS is auto for macOS arm64 too,
        # but we keep a separate "torch-apple" extra for clarity/consistency.
        "torch-cpu": [
            "torch==2.8.0",
            "torchvision==0.23.0",
            "torchaudio==2.8.0",
        ],
        # Apple Silicon (macOS arm64) – same wheels as torch-cpu but labeled separately
        "torch-apple": [
            'torch==2.8.0; platform_system=="Darwin" and platform_machine=="arm64"',
            'torchvision==0.23.0; platform_system=="Darwin" and platform_machine=="arm64"',
            'torchaudio==2.8.0; platform_system=="Darwin" and platform_machine=="arm64"',
        ],
        # CUDA wheels must come from the PyTorch index; keep this extra minimal
        # so it won't try to pull CPU wheels from PyPI after you preinstall CUDA builds.
        "torch-gpu": [
            # intentionally left empty
        ],
    },
    package_data={"deep_lvpm.data": ["*.npz"]},
)
