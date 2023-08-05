import setuptools

from sparsembed.__version__ import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

base_packages = [
    "torch >= 1.13",
    "tqdm >= 4.65",
    "transformers >= 4.30.2",
]

setuptools.setup(
    name="sparsembed",
    version=f"{__version__}",
    license="MIT",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="Sparse Embeddings for Neural Search.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/sparseembed",
    download_url="https://github.com/user/sparseembed/archive/v_01.tar.gz",
    keywords=[
        "neural search",
        "information retrieval",
        "semantic search",
        "SparseEmbed",
        "Google Research",
    ],
    packages=setuptools.find_packages(),
    install_requires=base_packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
