# evaluate

Evaluate candidates matchs.



## Parameters

- **scores** (*list[list[dict]]*)

- **qrels** (*dict*)

    Qrels.

- **queries_ids** (*list[str]*)

- **metrics** (*list*) – defaults to `[]`

    Metrics to compute.



## Examples

```python
>>> from neural_cherche import models, retrieve, utils
>>> import torch

>>> _ = torch.manual_seed(42)

>>> model = models.Splade(
...     model_name_or_path="distilbert-base-uncased",
...     device="cpu",
... )

>>> documents, queries_ids, queries, qrels = utils.load_beir(
...     "scifact",
...     split="test",
... )

>>> documents = documents[:10]

>>> retriever = retrieve.Splade(
...     key="id",
...     on=["title", "text"],
...     model=model
... )

>>> documents_embeddings = retriever.encode_documents(
...     documents=documents,
...     batch_size=1,
... )

>>> documents_embeddings = retriever.add(
...     documents_embeddings=documents_embeddings,
... )

>>> queries_embeddings = retriever.encode_queries(
...     queries=queries,
...     batch_size=1,
... )

>>> scores = retriever(
...     queries_embeddings=queries_embeddings,
...     k=30,
...     batch_size=1,
... )

>>> utils.evaluate(
...     scores=scores,
...     qrels=qrels,
...     queries_ids=queries_ids,
...     metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]
... )
{'map': 0.0033333333333333335, 'ndcg@10': 0.0033333333333333335, 'ndcg@100': 0.0033333333333333335, 'recall@10': 0.0033333333333333335, 'recall@100': 0.0033333333333333335}
```

