import setuptools

from neural_cherche.__version__ import __version__

with open(file="README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

base_packages = [
    "torch >= 1.13",
    "tqdm >= 4.66",
    "transformers >= 4.34.0",
    "lenlp >= 1.1.1",
    "scikit-learn >= 1.5.0",
]

eval = ["ranx >= 0.3.16", "beir >= 2.0.0"]

dev = [
    "mkdocs-material == 9.2.8",
    "mkdocs-awesome-pages-plugin == 2.9.2",
    " mkdocs-jupyter == 0.24.7",
]

setuptools.setup(
    name="neural_cherche",
    version=f"{__version__}",
    license="",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="Sparse Embeddings for Neural Search.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/neural-cherche",
    download_url="https://github.com/user/neural-cherche/archive/v_01.tar.gz",
    keywords=[
        "neural search",
        "information retrieval",
        "semantic search",
        "SparseEmbed",
        "Google Research",
        "Splade",
        "Stanford",
        "ColBERT",
    ],
    packages=setuptools.find_packages(),
    install_requires=base_packages,
    extras_require={"eval": base_packages + eval, "dev": base_packages + eval + dev},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
