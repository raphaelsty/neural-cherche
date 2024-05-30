# Colbert

It's higly recommended to use a GPU with ColBERT and to use a fine-tuned model as the base 
model. 

ColBERT can either act as a retriever or as a ranker. While it has the capability to be employed as a retriever, its primary design and strength lie in ranking documents rather than retrieving them. The retriever's key objectives is to eliminate irrelevant documents among a large collection. On the other hand, the ranker is designed to re-order documents (more relevant first) among a small subset of document pre-filtered by a retriever.

In practical terms, if your document collection is relatively small, ColBERT can effectively function as a retriever. However, if you're dealing with a big document collection, ColBERT is better suited to operate as a ranker. In this case, you should use a retriever such as TF IDF, BM25, Sentence Transformers, Splade to pre-filter the documents and then use ColBERT to re-rank the documents.

## Colbert ranker with BM25 retriever

ColBERT ranker can be used to re-rank candidates in output of a retriever following the
code below. We can use a TfIdf retriever, a Splade retriever or a SparseEmbed retriever.

```python
from neural_cherche import models, rank, retrieve
import torch

device = "cuda" if torch.cuda.is_available() else "cpu" # or mps
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

# Compute the embeddings of the candidates with the ranker model:
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

scores
```

```python
[[{'id': 'doc1', 'similarity': 9.37690258026123},
  {'id': 'doc3', 'similarity': 6.458564758300781},
  {'id': 'doc2', 'similarity': 6.071964263916016}],
 [{'id': 'doc2', 'similarity': 10.65597915649414},
  {'id': 'doc3', 'similarity': 6.5705132484436035},
  {'id': 'doc1', 'similarity': 5.962393283843994}],
 [{'id': 'doc3', 'similarity': 6.877983570098877},
  {'id': 'doc2', 'similarity': 4.163510799407959},
  {'id': 'doc1', 'similarity': 3.5986523628234863}]]
```


Note, we could also use the `encode_documents` method which allows to pre-compute all the embeddings of the documents. This is useful when we have a large number of documents and we want to pre-compute the embeddings of the documents to save time later.

```python
ranker_documents_embeddings = ranker.encode_documents(
    documents=documents,
    batch_size=batch_size,
)
```
