# ColBERT

ColBERT ranker.



## Parameters

- **key** (*str*)

- **on** (*list[str]*)

- **model** (*[models.ColBERT](../../models/ColBERT)*)

    ColBERT model.



## Examples

```python
>>> from neural_cherche import models, rank
>>> from pprint import pprint
>>> import torch

>>> _ = torch.manual_seed(42)

>>> encoder = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-mpnet-base-v2",
...     device="mps",
... )

>>> documents = [
...     {"id": 0, "document": "Food"},
...     {"id": 1, "document": "Sports"},
...     {"id": 2, "document": "Cinema"},
... ]

>>> queries = ["Food", "Sports", "Cinema"]

>>> ranker = rank.ColBERT(
...    key="id",
...    on=["document"],
...    model=encoder,
... )

>>> queries_embeddings = ranker.encode_queries(
...     queries=queries,
...     batch_size=3,
... )

>>> documents_embeddings = ranker.encode_documents(
...     documents=documents,
...     batch_size=3,
... )

>>> scores = ranker(
...     documents=[documents for _ in queries],
...     queries_embeddings=queries_embeddings,
...     documents_embeddings=documents_embeddings,
...     batch_size=3,
...     tqdm_bar=True,
...     k=3,
... )

>>> pprint(scores)
[[{'document': 'Food', 'id': 0, 'similarity': 20.23601531982422},
  {'document': 'Cinema', 'id': 2, 'similarity': 7.255690574645996},
  {'document': 'Sports', 'id': 1, 'similarity': 6.666046142578125}],
 [{'document': 'Sports', 'id': 1, 'similarity': 21.373430252075195},
  {'document': 'Cinema', 'id': 2, 'similarity': 5.494492053985596},
  {'document': 'Food', 'id': 0, 'similarity': 4.814355850219727}],
 [{'document': 'Sports', 'id': 1, 'similarity': 9.25660228729248},
  {'document': 'Food', 'id': 0, 'similarity': 8.206350326538086},
  {'document': 'Cinema', 'id': 2, 'similarity': 5.496612548828125}]]
```

## Methods

???- note "__call__"

    Rank documents  givent queries.

    **Parameters**

    - **documents**     (*list[list[dict]]*)    
    - **queries_embeddings**     (*dict[str, torch.Tensor]*)    
    - **documents_embeddings**     (*dict[str, torch.Tensor]*)    
    - **batch_size**     (*int*)     – defaults to `32`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    - **k**     (*int*)     – defaults to `None`    
    
???- note "encode_documents"

    Encode documents.

    **Parameters**

    - **documents**     (*list[str]*)    
    - **batch_size**     (*int*)     – defaults to `32`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    - **query_mode**     (*bool*)     – defaults to `False`    
    - **kwargs**    
    
???- note "encode_queries"

    Encode queries.

    **Parameters**

    - **queries**     (*list[str]*)    
    - **batch_size**     (*int*)     – defaults to `32`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    - **query_mode**     (*bool*)     – defaults to `True`    
    - **kwargs**    
    
