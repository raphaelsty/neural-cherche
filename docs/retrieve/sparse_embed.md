# SparseEmbed

Retrieving documents using SparseEmbed. SparseEmbed first retrieve documents following Splade
procedure and then computes dot products of embeddings between common activated tokens.

```python
from neural_cherche import models, retrieve
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1

model = models.SparseEmbed(
    model_name_or_path="distilbert-base-uncased",
    device=device,
)

documents = [
    {"id": "doc1", "title": "Paris", "text": "Paris is the capital of France."},
    {"id": "doc2", "title": "Montreal", "text": "Montreal is the largest city in Quebec."},
    {"id": "doc3", "title": "Bordeaux", "text": "Bordeaux in Southwestern France."},
]

retriever = retrieve.SparseEmbed(
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
[[{'id': 'doc1', 'similarity': 144.48985290527344},
  {'id': 'doc2', 'similarity': 111.0398941040039},
  {'id': 'doc3', 'similarity': 80.72007751464844}],
 [{'id': 'doc2', 'similarity': 169.8221435546875},
  {'id': 'doc1', 'similarity': 125.84573364257812},
  {'id': 'doc3', 'similarity': 77.57147216796875}],
 [{'id': 'doc1', 'similarity': 103.0795669555664},
  {'id': 'doc2', 'similarity': 81.4903564453125},
  {'id': 'doc3', 'similarity': 77.25212097167969}]]
```