# Colbert

It's higly recommended to use a GPU with ColBERT and to use a fine-tuned model as the base 
model. 

ColBERT can either act as a retriever or as a ranker. While it has the capability to be employed as a retriever, its primary design and strength lie in ranking documents rather than retrieving them. The retriever's key objectives is to eliminate irrelevant documents among a large collection. On the other hand, the ranker is designed to re-order documents (more relevant first) among a small subset of document pre-filtered by a retriever.

In practical terms, if your document collection is relatively small, ColBERT can effectively function as a retriever. However, if you're dealing with a big document collection, ColBERT is better suited to operate as a ranker. In this case, you should use a retriever such as TF IDF, BM25, Sentence Transformers, Splade to pre-filter the documents and then use ColBERT to re-rank the documents.

## Colbert retriever

```python
from neural_cherche import models, retrieve
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32

documents = [
    {"id": 0, "document": "Food"},
    {"id": 1, "document": "Sports"},
    {"id": 2, "document": "Cinema"},
]

queries = ["Food", "Sports", "Cinema"]

model = models.ColBERT(
    model_name_or_path="raphaelsty/neural-cherche-colbert",
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

queries_embeddings = retriever.encode_queries(
    queries=queries,
    batch_size=batch_size,
)

scores = retriever(
    queries_embeddings=queries_embeddings,
    batch_size=batch_size,
    k=3,
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

## Colbert ranker with BM25 retriever

ColBERT ranker can be used to re-rank candidates in output of a retriever following the
code below. We can use a TfIdf retriever, a Splade retriever or a SparseEmbed retriever.

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

model = models.ColBERT(
    model_name_or_path="raphaelsty/neural-cherche-colbert",
    device=device,
)

ranker = rank.ColBERT(
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
[[{'id': 'doc1', 'similarity': 20.224000930786133},
  {'id': 'doc2', 'similarity': 15.980965614318848},
  {'id': 'doc3', 'similarity': 9.628191947937012}],
 [{'id': 'doc2', 'similarity': 25.223176956176758},
  {'id': 'doc1', 'similarity': 17.255863189697266},
  {'id': 'doc3', 'similarity': 10.55866813659668}],
 [{'id': 'doc3', 'similarity': 20.619739532470703},
  {'id': 'doc1', 'similarity': 13.072492599487305},
  {'id': 'doc2', 'similarity': 12.057984352111816}]]
```
