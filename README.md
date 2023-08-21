<div align="center">
  <h1>SparsEmbed - Splade</h1>
  <p>Neural search</p>
</div>

This repository presents an unofficial replication of the research papers:

- *[SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)* authored by Thibault Formal, Benjamin Piwowarski, Stéphane Clinchant, SIGIR 2021.

- *[SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval](https://research.google/pubs/pub52289/)* authored by Weize Kong, Jeffrey M. Dudek, Cheng Li, Mingyang Zhang, and Mike Bendersky, SIGIR 2023.

**Note:** This project is currently a work in progress and models are not ready to use. 🔨🧹

## Installation

```
pip install sparsembed
```

If you plan to evaluate your model, install:

```
pip install "sparsembed[eval]"
```

## Training

### Dataset

Your training dataset must be made out of triples `(anchor, positive, negative)` where anchor is a query, positive is a document that is directly linked to the anchor and negative is a document that is not relevant for the query.

```python
X = [
    ("anchor 1", "positive 1", "negative 1"),
    ("anchor 2", "positive 2", "negative 2"),
    ("anchor 3", "positive 3", "negative 3"),
]
```

### Models

Both Splade and SparseEmbed models can be initialized from the `AutoModelForMaskedLM` pretrained models.

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = model.Splade(
    model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    device=device,
)
```

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = model.SparsEmbed(
    model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    embedding_size=64,
    k_tokens=96,
    device=device,
)
```

### Splade

The following PyTorch code snippet illustrates the training loop to fine-tune Splade:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sparsembed import model, utils, train, retrieve
import torch

device = "cuda" # cpu
batch_size = 8

model = model.Splade(
    model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    device=device
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

flops_scheduler = losses.FlopsScheduler(weight=3e-5, steps=10_000)

X = [
    ("anchor 1", "positive 1", "negative 1"),
    ("anchor 2", "positive 2", "negative 2"),
    ("anchor 3", "positive 3", "negative 3"),
]

for anchor, positive, negative in utils.iter(
        X,
        epochs=1,
        batch_size=batch_size,
        shuffle=True
    ):
        loss = train.train_splade(
            model=model,
            optimizer=optimizer,
            anchor=anchor,
            positive=positive,
            negative=negative,
            flops_loss_weight=flops_scheduler(),
            in_batch_negatives=True,
        )

documents, queries, qrels = utils.load_beir("scifact", split="test")

retriever = retrieve.SpladeRetriever(
    key="id",
    on=["title", "text"],
    model=model
)

retriever = retriever.add(
    documents=documents,
    batch_size=batch_size,
    k_tokens=96,
)

utils.evaluate(
    retriever=retriever,
    batch_size=batch_size,
    qrels=qrels,
    queries=queries,
    k=100,
    k_tokens=96,
    metrics=["map", "ndcg@10", "ndcg@10", "recall@10", "hits@10"]
)
```

## SparsEmbed

The following PyTorch code snippet illustrates the training loop to fine-tune SparseEmbed:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sparsembed import model, utils, train, retrieve
import torch

device = "cuda" # cpu

batch_size = 8

model = model.SparsEmbed(
    model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    device=device
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

flops_scheduler = losses.FlopsScheduler(weight=3e-5, steps=10_000)

X = [
    ("anchor 1", "positive 1", "negative 1"),
    ("anchor 2", "positive 2", "negative 2"),
    ("anchor 3", "positive 3", "negative 3"),
]

for anchor, positive, negative in utils.iter(
        X,
        epochs=1,
        batch_size=batch_size,
        shuffle=True
    ):
        loss = train.train_sparsembed(
            model=model,
            optimizer=optimizer,
            k_tokens=96,
            anchor=anchor,
            positive=positive,
            negative=negative,
            flops_loss_weight=flops_scheduler(),
            sparse_loss_weight=0.1,
            in_batch_negatives=True,
        )

documents, queries, qrels = utils.load_beir("scifact", split="test")

retriever = retrieve.SparsEmbedRetriever(
    key="id",
    on=["title", "text"],
    model=model
)

retriever = retriever.add(
    documents=documents,
    k_tokens=96,
    batch_size=batch_size
)

utils.evaluate(
    retriever=retriever,
    batch_size=batch_size,
    qrels=qrels,
    queries=queries,
    k=100,
    k_tokens=96,
    metrics=["map", "ndcg@10", "ndcg@10", "recall@10", "hits@10"]
)
```