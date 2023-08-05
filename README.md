# SparseEmbed 

**Note:** This project is currently a work in progress. 🔨🧹

This repository presents an unofficial replication of the research paper titled "[SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval](https://research.google/pubs/pub52289/)" authored by Weize Kong, Jeffrey M. Dudek, Cheng Li, Mingyang Zhang, and Mike Bendersky, SIGIR 2023.

## Overview

This repository aims to replicate the findings of the SparseEmbed paper, focusing on learning both sparse lexical representations and contextual token level embeddings for retrieval tasks. We propose to fine-tune the model and then to retrieve documents from a set of queries.

The `SparsEmbed` model is compatible with any MLM based model using the class `AutoModelForMaskedLM` from HuggingFace.

### Differences with the original paper

1. **Loss Function:** We did not yet implement the distillation loss used in the paper. We have initially opted for a cosine loss like the one used in SentenceTransformer library. This decision was made to fine-tune the model from scratch, avoiding the use of a cross-encoder as a teacher. The distillation loss should be available soon.

2. **Multi-Head Implementation:** At this stage, the distinct MLM (Masked Language Model) head for document encoding and query encoding has not been incorporated. Our current implementation employs a shared MLM head (calculating sparse activations) for both documents and queries.

## Installation

```
pip install sparsembed
```

## Training

The following PyTorch code snippet illustrates the training loop designed to fine-tune the model:

```python
from sparsembed import model, utils, losses
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

device = "cuda"  # cpu / cuda / mps
batch_size = 32

model = model.SparsEmbed(
    model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    device=device,
)

model = model.to(device)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-6,
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
    queries_embeddings = model(queries, k=96)

    documents_embeddings = model(documents, k=256)

    scores = utils.scores(
        queries_activations=queries_embeddings["activations"],
        queries_embeddings=queries_embeddings["embeddings"],
        documents_activations=documents_embeddings["activations"],
        documents_embeddings=documents_embeddings["embeddings"],
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

This code segment encapsulates the iterative process of fine-tuning the model's parameters using a combination of cosine and Flops loss functions. Queries and documents are both a list of strings. Labels is a Torch tensor containing binary values of 0 or 1. A label of 0 indicates that the query is not relevant to the document, while a label of 1 signifies that the query is indeed relevant to the document. 

## Inference

Upon successfully training our model, the next step involves initializing a retriever to facilitate the retrieval of the most accurate documents based on a given set of queries. The retrieval process is conducted through two main steps: adding documents and querying.

1. **Adding Documents**: By invoking the `add` method, the retriever undertakes document encoding. It performs this task by generating a sparse matrix that encapsulates the contributions of sparsely activated tokens. This matrix is constructed using the weighted values of these tokens.

2. **Querying**: When the `__call__` method is invoked, the retriever proceeds to encode the query. Similar to the document encoding phase, the retriever constructs a sparse matrix that represents the query. This matrix is then used in a dot product operation against the sparse matrices of the stored documents. After this initial retrieval, a re-ranking process takes place. This re-ranking is based on the contextual representations of activated tokens derived from both the queries and the documents. This is in accordance with the established SparseEmbed scoring formula.

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
    k_token=64,
    batch_size=3,
)

retriever(
    q = [
        "Apple", 
        "Banana",
    ], 
    k_sparse=64, 
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

## Acknowledgments

I would like to express my gratitude to the authors of the SparseEmbed paper for their valuable contributions, which serve as the foundation for this replication attempt.
