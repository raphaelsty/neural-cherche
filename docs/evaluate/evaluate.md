# Evaluate

Neural-Cherche evaluation is based on [RANX](https://github.com/AmenRa/ranx). We can also download datasets of [BEIR Benchmark](https://github.com/beir-cellar/beir) with the `utils.load_beir` function.


## Installation

```bash
pip install "neural-cherche[eval]"
```

## Usage

Let"s first create a pipeline which output candidates and scores:

```python
from neural_cherche import retrieve, utils

# Input dataset for evaluation
documents, queries, qrels = utils.load_beir(
    "scifact",
    split="test",
)

retriever = retrieve.BM25(key="id", on=["title", "text"])

documents_embeddings = retriever.encode_documents(
    documents=documents,
)

documents_embeddings = retriever.add(
    documents_embeddings=documents_embeddings,
)

queries_embeddings = retriever.encode_queries(
    queries=queries,
)

scores = retriever(
    queries_embeddings=queries_embeddings,
    k=30,
)

utils.evaluate(
    scores=scores,
    qrels=qrels,
    queries=queries,
    metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
)
```

```python
{
    "map": 0.6433690206955331,
    "ndcg@10": 0.6848343124746807,
    "ndcg@100": 0.7046426757236496,
    "recall@10": 0.8167222222222221,
    "recall@100": 0.8933333333333333,
}
```

## Evaluation dataset

Here are what documents should looks like (an id with multiples fields, no matter the name):

```python
[
    {
        "id": "document_0",
        "title": "title 0",
        "text": "text 0",
    },
    {
        "id": "document_1",
        "title": "title 1",
        "text": "text 1",
    },
    ...
    {
        "id": "document_n",
        "title": "title n",
        "text": "text n",
    },
]
```

Queries is a list of strings:

```python
[
    "first query",
    "second query",
    "third query",
    "fourth query",
    "fifth query",
]
```

Qrels is the mapping between queries ids as key and dict of relevant documents with 1 as value:

```python
{
    "first query": {"document_0": 1},
    "second query": {"document_10": 1},
    "third query": {"document_5": 1},
    "fourth query": {"document_22": 1},
    "fifth query": {"document_23": 1, "document_0": 1},
}
```

## Metrics

We can evaluate our model with various metrics detailed [here](https://amenra.github.io/ranx/metrics/).