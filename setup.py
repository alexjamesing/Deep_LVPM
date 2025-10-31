import setuptools

setuptools.setup(
    name="deep-lvpm",
    version="0.2.3",
    author="Alex James Ing",
    description="Deep Latent Variable Path Modelling (Keras 3, multi-backend)",
    packages=setuptools.find_packages(),
    python_requires=">=3.10,<3.12",
    install_requires=[
        "keras==3.10.0",
        "pydot==4.0.1",
        "scikit-learn==1.6.1",
        "matplotlib==3.9.4",
    ],
    extras_require={
        # ---------------- TensorFlow ----------------
        "tf-cpu": [
            "tensorflow==2.20.0",
        ],
        # Linux-only; bundles CUDA/cuDNN runtime wheels
        "tf-gpu": [
            'tensorflow[and-cuda]==2.20.0; platform_system=="Linux"',
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

### Updated install instructions:
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[tf-cpu]"
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[tf-gpu]"
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[tf-apple]"
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[torch-cpu]"
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[torch-apple]"
# pip install --index-url https://download.pytorch.org/whl/cu124 \
#    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[torch-gpu]"
# (Install PyTorch CUDA wheels from the official index before installing the torch-gpu extra.)
