# Colbert

Ranking documents using the ColBERT model. The ColBERT models is powerful to rank
documents. It's higly recommended to use a GPU and to use a fine-tuned model as the base 
model. 

ColBERT ranker can be used to re-rank candidates in output of a retriever following the
code below. We can use a TfIdf retriever, a Splade retriever or a SparseEmbed retriever.
We can also use ColBERT as a standalone ranker.

## Colbert ranker with TfIdf retriever

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

queries = [
    "What is the capital of France?",
    "What is the largest city in Quebec?",
    "Where is Bordeaux?",
]

retriever = retrieve.TfIdf(
    key="id",
    on=["title", "text"],
)

model = models.ColBERT(
    model_name_or_path="sentence-transformers/all-mpnet-base-v2",
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

## ColBERT standalone

We can also use ColBERT as a standalone ranker. In this case, we need to provide the
set of documents we want to rank for each query.

```python
from neural_cherche import models, rank
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32

model = models.ColBERT(
    model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    device=device,
)

documents = [
    {"id": "doc1", "title": "Paris", "text": "Paris is the capital of France."},
    {"id": "doc2", "title": "Montreal", "text": "Montreal is the largest city in Quebec."},
    {"id": "doc3", "title": "Bordeaux", "text": "Bordeaux in Southwestern France."},
]

ranker = rank.ColBERT(
    key="id",
    on=["title", "text"],
    model=model
)

documents_embeddings = ranker.encode_documents(
    documents=documents,
    batch_size=batch_size,
)

queries = [
    "What is the capital of France?",
    "What is the largest city in Quebec?",
    "Where is Bordeaux?",
]

queries_embeddings = ranker.encode_queries(
    queries=queries,
    batch_size=batch_size,
)

scores = ranker(
    documents=[documents for _ in queries], # We provide the same documents for each query
    documents_embeddings=documents_embeddings,
    queries_embeddings=queries_embeddings,
    batch_size=batch_size,
    k=100,
)

scores
```

```python
[[{'id': 'doc1',
   'title': 'Paris',
   'text': 'Paris is the capital of France.',
   'similarity': 22.771282196044922},
  {'id': 'doc3',
   'title': 'Bordeaux',
   'text': 'Bordeaux in Southwestern France.',
   'similarity': 14.437524795532227},
  {'id': 'doc2',
   'title': 'Montreal',
   'text': 'Montreal is the largest city in Quebec.',
   'similarity': 14.067940711975098}],
 [{'id': 'doc2',
   'title': 'Montreal',
   'text': 'Montreal is the largest city in Quebec.',
   'similarity': 25.385271072387695},
  {'id': 'doc1',
   'title': 'Paris',
   'text': 'Paris is the capital of France.',
   'similarity': 15.790643692016602},
  {'id': 'doc3',
   'title': 'Bordeaux',
   'text': 'Bordeaux in Southwestern France.',
   'similarity': 11.97323226928711}],
 [{'id': 'doc3',
   'title': 'Bordeaux',
   'text': 'Bordeaux in Southwestern France.',
   'similarity': 22.203411102294922},
  {'id': 'doc1',
   'title': 'Paris',
   'text': 'Paris is the capital of France.',
   'similarity': 13.355796813964844},
  {'id': 'doc2',
   'title': 'Montreal',
   'text': 'Montreal is the largest city in Quebec.',
   'similarity': 11.263651847839355}]]
```
