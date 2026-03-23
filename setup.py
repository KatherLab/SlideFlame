from pathlib import Path
from setuptools import find_packages, setup

if __name__ == "__main__":
    readme_path = Path(__file__).parent / "README.md"
    long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    REQUIREMENTS = [
        "einops",
        "einops-exts",
        "numpy>=1.26",
        "h5py",
        "pandas",
        "tqdm",
        "transformers>=4.40.0",
        "accelerate>=0.26.0",
        "peft>=0.10.0",
        "sentencepiece",
        "protobuf",
        "safetensors",
    ]

    TRAINING = [
        "wandb",
        "scipy",
        "scikit-learn",
    ]

    DEV = [
        "black",
        "mypy",
        "pylint",
        "pytest",
    ]

    setup(
        name="slideflame",
        packages=find_packages(),
        include_package_data=True,
        version="0.1.0",
        license="MIT",
        description="SlideFlame: WSI-to-text generation with patch/slide features and gated cross-attention",
        long_description=long_description,
        long_description_content_type="text/markdown",
        data_files=[(".", ["README.md"])] if readme_path.exists() else None,
        keywords=["computational pathology", "WSI", "vision-language", "report generation"],
        install_requires=REQUIREMENTS,
        extras_require={
            "training": TRAINING,
            "dev": DEV,
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.9",
        ],
    )