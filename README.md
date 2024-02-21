<div align="center">
  <h1>Neural-Cherche</h1>
  <p>Neural Search</p>
</div>

<p align="center"><img width=500 src="docs/img/logo.png"/></p>

<div align="center">
  <!-- Documentation -->
  <a href="https://raphaelsty.github.io/neural-cherche/"><img src="https://img.shields.io/website?label=Documentation&style=flat-square&url=https%3A%2F%2Fraphaelsty.github.io/neural-cherche/%2F" alt="documentation"></a>
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="license"></a>
</div>

Neural-Cherche is a library designed to fine-tune neural search models such as Splade, ColBERT, and SparseEmbed on a specific dataset. Neural-Cherche also provide classes to run efficient inference on a fine-tuned retriever or ranker. Neural-Cherche aims to offer a straightforward and effective method for fine-tuning and utilizing neural search models in both offline and online settings. It also enables users to save all computed embeddings to prevent redundant computations.

Neural-Cherche is compatible with CPU, GPU and MPS devices.

## Installation

We can install neural-cherche using:

```
pip install neural-cherche
```

If we plan to evaluate our model while training install:

```
pip install "neural-cherche[eval]"
```

## Documentation

The complete documentation is available [here](https://raphaelsty.github.io/neural-cherche/).

## Quick Start

Your training dataset must be made out of triples `(anchor, positive, negative)` where anchor is a query, positive is a document that is directly linked to the anchor and negative is a document that is not relevant for the anchor.

```python
X = [
    ("anchor 1", "positive 1", "negative 1"),
    ("anchor 2", "positive 2", "negative 2"),
    ("anchor 3", "positive 3", "negative 3"),
]
```

And here is how to fine-tune ColBERT from a Sentence Transformer pre-trained checkpoint using neural-cherche:

```python
import torch

from neural_cherche import models, utils, train

model = models.ColBERT(
    model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    device="cuda" if torch.cuda.is_available() else "cpu" # mps is also supported
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

X = [
    ("query", "positive document", "negative document"),
    ("query", "positive document", "negative document"),
    ("query", "positive document", "negative document"),
]

for anchor, positive, negative in utils.iter(
        X,
        epochs=1,
        batch_size=32,
        shuffle=True
    ):

    loss = train.train_colbert(
        model=model,
        optimizer=optimizer,
        anchor=anchor,
        positive=positive,
        negative=negative,
    )

model.save_pretrained("checkpoint")
```

## Retrieval

Here is how to use the fine-tuned ColBERT model to retrieve documents:

```python
from neural_cherche import models, retrieve
import torch

device = "cuda" if torch.cuda.is_available() else "cpu" # mps is also supported
batch_size = 32

documents = [
    {"id": 0, "document": "Food"},
    {"id": 1, "document": "Sports"},
    {"id": 2, "document": "Cinema"},
]

model = models.ColBERT(
    model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    device=device,
)

retriever = retrieve.ColBERT(
    key="id",
    on=["document"],
    model=model,
)

documents_embeddings = retriever.encode_documents(
    documents=documents,
    batch_size=batch_size,
)

retriever = retriever.add(
    documents_embeddings=documents_embeddings,
)
```

Now we can retrieve documents using the fine-tuned model:

```python
queries_embeddings = retriever.encode_queries(
    queries=["Food", "Sports", "Cinema"],
    batch_size=batch_size,
)

scores = retriever(
    queries_embeddings=queries_embeddings,
    batch_size=batch_size,
    k=10, # number of documents to retrieve
)

scores
```

```python
[[{'id': 0, 'similarity': 22.825355529785156},
  {'id': 1, 'similarity': 11.201947212219238},
  {'id': 2, 'similarity': 10.748161315917969}],
 [{'id': 1, 'similarity': 23.21628189086914},
  {'id': 0, 'similarity': 9.9658203125},
  {'id': 2, 'similarity': 7.308732509613037}],
 [{'id': 1, 'similarity': 6.4031805992126465},
  {'id': 0, 'similarity': 5.601611137390137},
  {'id': 2, 'similarity': 5.599479675292969}]]
```

Neural-Cherche also provides a `SparseEmbed`, a `SPLADE`, a `TFIDF` retriever and a `ColBERT` ranker which can be used to re-order output of a retriever. For more information, please refer to the [documentation](https://raphaelsty.github.io/neural-cherche/).

### Neural-Cherche Contributors

- [Benjamin Clavié](https://github.com/bclavie)

## References

- *[SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)* authored by Thibault Formal, Benjamin Piwowarski, Stéphane Clinchant, SIGIR 2021.

- *[SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval](https://arxiv.org/abs/2109.10086)* authored by Thibault Formal, Carlos Lassance, Benjamin Piwowarski, Stéphane Clinchant, SIGIR 2022.

- *[SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval](https://research.google/pubs/pub52289/)* authored by Weize Kong, Jeffrey M. Dudek, Cheng Li, Mingyang Zhang, and Mike Bendersky, SIGIR 2023.

- *[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)* authored by Omar Khattab, Matei Zaharia, SIGIR 2020.

## License

This Python library is licensed under the MIT open-source license, and the splade model is licensed as non-commercial only by the authors. SparseEmbed and ColBERT are fully open-source including commercial usage.
