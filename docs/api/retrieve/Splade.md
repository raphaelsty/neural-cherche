# Splade

Retriever class.



## Parameters

- **key** (*str*)

    Document unique identifier.

- **on** (*list[str]*)

    Document texts.

- **model** (*[models.Splade](../../models/Splade)*)

    SparsEmbed model.

- **tokenizer_parallelism** (*str*) – defaults to `false`



## Examples

```python
>>> from neural_cherche import models, retrieve
>>> from pprint import pprint
>>> import torch

>>> _ = torch.manual_seed(42)

>>> documents = [
...     {"id": 0, "document": "Food"},
...     {"id": 1, "document": "Sports"},
...     {"id": 2, "document": "Cinema"},
... ]

>>> queries = ["Food", "Sports", "Cinema"]

>>> model = models.Splade(
...     model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
...     device="mps",
... )

>>> retriever = retrieve.Splade(
...     key="id",
...     on="document",
...     model=model
... )

>>> documents_embeddings = retriever.encode_documents(
...     documents=documents,
...     batch_size=32,
... )

>>> queries_embeddings = retriever.encode_queries(
...     queries=queries,
...     batch_size=32,
... )

>>> retriever = retriever.add(
...     documents_embeddings=documents_embeddings,
... )

>>> scores = retriever(
...     queries_embeddings=queries_embeddings,
...     k=3,
... )

>>> pprint(scores)
[[{'id': 0, 'similarity': 489.65244},
  {'id': 2, 'similarity': 338.9705},
  {'id': 1, 'similarity': 332.3472}],
 [{'id': 1, 'similarity': 470.40497},
  {'id': 2, 'similarity': 301.56982},
  {'id': 0, 'similarity': 278.8062}],
 [{'id': 2, 'similarity': 472.487},
  {'id': 1, 'similarity': 341.8396},
  {'id': 0, 'similarity': 319.97287}]]
```

## Methods

???- note "__call__"

    Retrieve documents from batch of queries.

    **Parameters**

    - **queries_embeddings**     (*dict[str, scipy.sparse._csr.csr_matrix]*)    
    - **k**     (*int*)     – defaults to `None`    
    - **batch_size**     (*int*)     – defaults to `2000`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    
???- note "add"

    Add new documents to the TFIDF retriever. The tfidf won't be refitted.

    **Parameters**

    - **documents_embeddings**     (*dict[str, scipy.sparse._csr.csr_matrix]*)    
    
???- note "encode_documents"

    Encode queries into sparse matrix.

    **Parameters**

    - **documents**     (*list[dict]*)    
    - **batch_size**     (*int*)     – defaults to `32`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    - **query_mode**     (*bool*)     – defaults to `False`    
    - **kwargs**    
    
???- note "encode_queries"

    Encode queries into sparse matrix.

    **Parameters**

    - **queries**     (*list[str]*)    
    - **batch_size**     (*int*)     – defaults to `32`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    - **query_mode**     (*bool*)     – defaults to `True`    
    - **kwargs**    
    
???- note "top_k"

    Return the top k documents for each query.

    **Parameters**

    - **similarities**     (*scipy.sparse._csc.csc_matrix*)    
    - **k**     (*int*)    
    
