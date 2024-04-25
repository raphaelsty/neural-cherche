# SparseEmbed

## SparseEmbed retriever

Retrieving documents using SparseEmbed. SparseEmbed first retrieve documents following Splade
procedure and then computes dot products of embeddings between common activated tokens.

```python
from neural_cherche import models, retrieve
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32

model = models.SparseEmbed(
    model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
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

## SparseEmbed ranker

Ranking documents using SparseEmbed. The following code use BM25 to retrieve documents and then
rank them using SparseEmbed.

```python
from neural_cherche import models, rank, retrieve
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32

documents = [
    {"id": "doc1", "title": "Paris", "text": "Paris is the capital of France."},
    {"id": "doc2", "title": "Montreal", "text": "Montreal is the largest city in Quebec."},
    {"id": "doc3", "title": "Bordeaux", "text": "Bordeaux in Southwestern France."},
]

retriever = retrieve.BM25(
    key="id",
    on=["title", "text"],
)

model = models.SparseEmbed(
    model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
    device=device,
)

ranker = rank.SparseEmbed(
    key="id",
    on=["title", "text"],
    model=model
)

retriever_documents_embeddings = retriever.encode_documents(
    documents=documents,
)

retriever.add(
    documents_embeddings=retriever_documents_embeddings,
)

ranker_documents_embeddings = ranker.encode_documents(
    documents=documents,
    batch_size=batch_size,
)
```

Once we have created our indexes, we can use the ranker to re-rank the candidates retrieved by the retriever.

```python
queries = [
    "What is the capital of France?",
    "What is the largest city in Quebec?",
    "Where is Bordeaux?",
]

retriever_queries_embeddings = retriever.encode_queries(
    queries=queries,
)

ranker_queries_embeddings = ranker.encode_queries(
    queries=queries,
    batch_size=batch_size,
)

candidates = retriever(
    queries_embeddings=retriever_queries_embeddings,
    k=1000,
)

scores = ranker(
    documents=candidates,
    queries_embeddings=ranker_queries_embeddings,
    documents_embeddings=ranker_documents_embeddings,
    k=100,
    batch_size=32,
)

scores
```

```python
[[{'id': 'doc1', 'similarity': 450.2735},
  {'id': 'doc3', 'similarity': 184.59885},
  {'id': 'doc2', 'similarity': 98.53701}],
 [{'id': 'doc2', 'similarity': 391.74216},
  {'id': 'doc1', 'similarity': 111.45184},
  {'id': 'doc3', 'similarity': 51.19094}],
 [{'id': 'doc3', 'similarity': 349.82397},
  {'id': 'doc1', 'similarity': 74.993576},
  {'id': 'doc2', 'similarity': 33.37598}]]
```

If we don't want to pre-compute the whole set of documents embeddings in order to re-rank documents,
we can call the `encode_candidates_documents` methods to only compute the embeddings of the candidates
documents.

```python
queries = [
    "What is the capital of France?",
    "What is the largest city in Quebec?",
    "Where is Bordeaux?",
]

retriever_queries_embeddings = retriever.encode_queries(
    queries=queries,
)

ranker_queries_embeddings = ranker.encode_queries(
    queries=queries,
    batch_size=batch_size,
)

candidates = retriever(
    queries_embeddings=retriever_queries_embeddings,
    k=1000,
)

# Only compute the embeddings of the candidates documents
ranker_documents_embeddings = ranker.encode_candidates_documents(
    documents=documents,
    candidates=candidates,
    batch_size=batch_size,
)

scores = ranker(
    documents=candidates,
    queries_embeddings=ranker_queries_embeddings,
    documents_embeddings=ranker_documents_embeddings,
    k=100,
    batch_size=32,
)
```
