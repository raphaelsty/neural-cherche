<div align="center">
  <h1>SparsEmbed</h1>
  <p>Neural search</p>
</div>

This repository presents an unofficial replication of the research paper *[SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval](https://research.google/pubs/pub52289/)* authored by Weize Kong, Jeffrey M. Dudek, Cheng Li, Mingyang Zhang, and Mike Bendersky, SIGIR 2023.

**Note:** This project is currently a work in progress. 🔨🧹

## Overview

This repository aims to replicate the SparseEmbed model, focusing on learning both sparse lexical representations and contextual token level embeddings for retrieval tasks. 

The `SparsEmbed` model available here is compatible with any model compatible with the class `AutoModelForMaskedLM` from HuggingFace.

### Differences with the original paper

1. **Loss Function:** We did not yet implement the distillation loss used in the paper. We have initially opted for a cosine loss like the one used in SentenceTransformer library. This decision was made to fine-tune the model from scratch, avoiding the use of a cross-encoder as a teacher. The distillation loss should be available soon.

2. **Multi-Head Implementation:** At this stage, the distinct MLM (Masked Language Model) head for document encoding and query encoding has not been incorporated. Our current implementation employs a shared MLM head (calculating sparse activations) for both documents and queries.

## Installation

```
pip install sparsembed
```

## Training

The following PyTorch code snippet illustrates the training loop to fine-tune the model:

```python
from sparsembed import model, utils, losses
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

device = "cuda"  # cpu / cuda
batch_size = 32

model = model.SparsEmbed(
    model=AutoModelForMaskedLM.from_pretrained("Luyu/co-condenser-marco").to(device),
    tokenizer=AutoTokenizer.from_pretrained("Luyu/co-condenser-marco"),
    device=device,
)

model = model.to(device)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5,
)

flops_loss = losses.Flops()

cosine_loss = losses.Cosine()

dataset = [
    # Query, Document, Label (1: Relevant, 0: Not Relevant)
    ("Apple", "Apple is a popular fruit.", 1),
    ("Apple", "Banana is a popular fruit.", 0),
    ("Banana", "Apple is a popular fruit.", 0),
    ("Banana", "Banana is a yellow fruit.", 1),
]

for queries, documents, labels in utils.iter(
    dataset,
    device=device,
    epochs=1,
    batch_size=batch_size,
    shuffle=True,
):
    queries_embeddings = model(queries, k=32)

    documents_embeddings = model(documents, k=32)

    scores = utils.scores(
        queries_activations=queries_embeddings["activations"],
        queries_embeddings=queries_embeddings["embeddings"],
        documents_activations=documents_embeddings["activations"],
        documents_embeddings=documents_embeddings["embeddings"],
        device=device,
    )

    loss = cosine_loss.dense(
        scores=scores,
        labels=labels,
    )

    loss += 0.1 * cosine_loss.sparse(
        queries_sparse_activations=queries_embeddings["sparse_activations"],
        documents_sparse_activations=documents_embeddings["sparse_activations"],
        labels=labels,
    )

    loss += 4e-3 * flops_loss(
        sparse_activations=queries_embeddings["sparse_activations"]
    )
    loss += 4e-3 * flops_loss(
        sparse_activations=documents_embeddings["sparse_activations"]
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Inference

Once we trained the model, we can initialize a `Retriever` to retrieve relevant documents given a query.

- It build a sparse matrix from sparse activations of documents.
- It build a sparse matrix from sparse activations of queries.
- It match relevant documents using dot product of both sparse matrix.
- It re-rank documents based on contextual embbedings similarity score.

```python
from sparsembed import retrieve

documents = [{
    "id": 0,
    "document": "Apple is a popular fruit.",
  },
  {
    "id": 1,
    "document": "Banana is a popular fruit.",
  },
  {
    "id": 2,
    "document": "Banana is a yellow fruit.",
  }
]

retriever = retrieve.Retriever(
    key="id", 
    on="document", 
    model=model # Trained SparseEmbed model.
)

retriever = retriever.add(
    documents=documents,
    k_token=32, # Number of tokens to activate.
    batch_size=3,
)

retriever(
    q = [
        "Apple", 
        "Banana",
    ], 
    k_sparse=20, # Number of documents to retrieve.
    k_token=32, # Number of tokens to activate.
    batch_size=3
)
```

```python
[[{'id': 0, 'similarity': 195.057861328125},
  {'id': 1, 'similarity': 183.51429748535156},
  {'id': 2, 'similarity': 158.66012573242188}],
 [{'id': 1, 'similarity': 214.34048461914062},
  {'id': 2, 'similarity': 194.5692901611328},
  {'id': 0, 'similarity': 192.5744171142578}]]
```

## Evaluations

Work in progress.
