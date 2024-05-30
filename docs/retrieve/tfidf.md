# TfIdf

Retrieving documents using TfIdf. We must always encode documents before queries when 
using TfIdf retriever to fit the vectorizer, otherwise the system will raise an error.

```python
from neural_cherche import retrieve
from lenlp import sparse

documents = [
    {"id": "doc1", "title": "Paris", "text": "Paris is the capital of France."},
    {"id": "doc2", "title": "Montreal", "text": "Montreal is the largest city in Quebec."},
    {"id": "doc3", "title": "Bordeaux", "text": "Bordeaux in Southwestern France."},
]

retriever = retrieve.TfIdf(
    key="id",
    on=["title", "text"],
    tfidf=sparse.TfidfVectorizer(normalize=True, ngram_range=(3, 5), analyzer="char_wb"),
)

documents_embeddings = retriever.encode_documents(
    documents=documents,
)

retriever.add(
    documents_embeddings=documents_embeddings,
)
```

Once we have created our index, we can use the retriever to retrieve the candidates.

```python
queries = [
    "What is the capital of France?",
    "What is the largest city in Quebec?",
    "Where is Bordeaux?",
]

queries_embeddings = retriever.encode_queries(
    queries=queries,
)

scores = retriever(
    queries_embeddings=queries_embeddings,
    k=100,
)

scores
```

```python
[[{'id': 'doc1', 'similarity': 0.7398676589821398},
  {'id': 'doc2', 'similarity': 0.0835572631633293},
  {'id': 'doc3', 'similarity': 0.0610449729335254}],
 [{'id': 'doc2', 'similarity': 0.681400067648393},
  {'id': 'doc1', 'similarity': 0.08957331010686152},
  {'id': 'doc3', 'similarity': 0.014090768382163084}],
 [{'id': 'doc3', 'similarity': 0.5803551728233277},
  {'id': 'doc1', 'similarity': 0.043108258977414556},
  {'id': 'doc2', 'similarity': 0.02088494592973491}]]
```