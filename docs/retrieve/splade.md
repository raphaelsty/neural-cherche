# Splade

Retrieving documents using Splade. Splade activations are stored in a sparse matrix.

```python
from neural_cherche import models, retrieve
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32

model = models.Splade(
    model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
    device=device,
)

documents = [
    {"id": "doc1", "title": "Paris", "text": "Paris is the capital of France."},
    {"id": "doc2", "title": "Montreal", "text": "Montreal is the largest city in Quebec."},
    {"id": "doc3", "title": "Bordeaux", "text": "Bordeaux in Southwestern France."},
]

retriever = retrieve.Splade(
    key="id",
    on=["title", "text"],
    model=model
)

documents_embeddings = retriever.encode_documents(
    documents=documents,
    batch_size=batch_size,
)

retriever.add(
    documents_embeddings=documents_embeddings,
)

queries = [
    "What is the capital of France?",
    "What is the largest city in Quebec?",
    "Where is Bordeaux?",
]

queries_embeddings = retriever.encode_queries(
    queries=queries,
    batch_size=batch_size,
)

scores = retriever(
    queries_embeddings=queries_embeddings,
    k=100,
)

scores
```

```python
[[{'id': 'doc1', 'similarity': 509.41132},
  {'id': 'doc2', 'similarity': 416.7586},
  {'id': 'doc3', 'similarity': 391.12656}],
 [{'id': 'doc2', 'similarity': 539.30164},
  {'id': 'doc1', 'similarity': 438.1915},
  {'id': 'doc3', 'similarity': 366.35565}],
 [{'id': 'doc3', 'similarity': 402.1179},
  {'id': 'doc1', 'similarity': 382.23434},
  {'id': 'doc2', 'similarity': 357.77188}]]
```