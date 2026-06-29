from setuptools import setup, find_packages

setup(
    name="ebgs",
    version="0.1.0",
    description="Euclid BGS Image Synthesis via Schrödinger Bridge Diffusion",
    author="EBGS Authors",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2",
        "torchvision",
        "pytorch-lightning>=2.2",
        "omegaconf",
        "wandb",
        "astropy",
        "safetensors",
        "xformers",
        "reproject",
        "natsort",
        "packaging",
        "tqdm",
        "numpy",
        "scipy",
    ],
    extras_require={
        "dev": ["pytest", "matplotlib"],
    },
)
