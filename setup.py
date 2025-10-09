from pathlib import Path
import setuptools

setuptools.setup(
    name="deep-lvpm",
    version="0.2.3",
    author="Alex James Ing",
    description="Deep Latent Variable Path Modelling (Keras 3, multi-backend)",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "keras>=3.0.0,<4",   # Keras Core (multi-backend)
        "pydot",
        "scikit-learn",
        "matplotlib",
    ],
    extras_require={
        # ---------------- TensorFlow ----------------
        "tf-cpu": [
            "tensorflow>=2.16,<3",
        ],
        # Linux-only; bundles CUDA/cuDNN runtime wheels
        "tf-gpu": [
            'tensorflow[and-cuda]>=2.16,<3; platform_system=="Linux"',
        ],
        # Apple Silicon (macOS arm64) via Metal
        "tf-apple": [
            'tensorflow-macos>=2.16,<3; platform_system=="Darwin" and platform_machine=="arm64"',
            'tensorflow-metal>=1.1; platform_system=="Darwin" and platform_machine=="arm64"',
        ],

        # ---------------- PyTorch -------------------
        # CPU on Linux/Windows/macOS Intel; MPS is auto for macOS arm64 too,
        # but we keep a separate "torch-apple" extra for clarity/consistency.
        "torch-cpu": [
            "torch>=2.2",
            "torchvision>=0.17",
            "torchaudio>=2.2",
        ],
        # Apple Silicon (macOS arm64) – same wheels as torch-cpu but labeled separately
        "torch-apple": [
            'torch>=2.2; platform_system=="Darwin" and platform_machine=="arm64"',
            'torchvision>=0.17; platform_system=="Darwin" and platform_machine=="arm64"',
            'torchaudio>=2.2; platform_system=="Darwin" and platform_machine=="arm64"',
        ],
        # CUDA wheels must come from the PyTorch index; keep this extra minimal
        # so it won't try to pull CPU wheels from PyPI after you preinstall CUDA builds.
        "torch-gpu": [
            # intentionally left empty 
        ],
    },
    package_data={"deep_lvpm.data": ["*.npz"]},
)

### Here are instructions for different installs:
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[tf-cpu]"
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[tf-gpu]"
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[tf-apple]"
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[torch-cpu]"
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[torch-apple]"
# pip install --index-url https://download.pytorch.org/whl/cu121 \
#    torch torchvision torchaudio
# pip install "git+https://github.com/alexjamesing/Deep_LVPM.git#egg=deep-lvpm[torch-gpu]"
# (Note that the torch GPU install requires pre-installation of several packages)




