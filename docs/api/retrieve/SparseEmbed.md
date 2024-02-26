# SparseEmbed

Retriever class.



## Parameters

- **key** (*str*)

    Document unique identifier.

- **on** (*list[str]*)

    Document texts.

- **model** (*[models.SparseEmbed](../../models/SparseEmbed)*)

    SparsEmbed model.

- **tokenizer_parallelism** (*str*) – defaults to `false`



## Examples

```python
>>> from neural_cherche import models, retrieve
>>> from pprint import pprint
>>> import torch

>>> _ = torch.manual_seed(42)

>>> device = "mps"

>>> model = models.SparseEmbed(
...     model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
...     device=device,
...     embedding_size=64,
... )

>>> retriever = retrieve.SparseEmbed(
...     key="id",
...     on="document",
...     model=model,
... )

>>> documents = [
...     {"id": 0, "document": "Food"},
...     {"id": 1, "document": "Sports"},
...     {"id": 2, "document": "Cinema"},
... ]

>>> queries = ["Food", "Sports", "Cinema"]

>>> documents_embeddings = retriever.encode_documents(
...     documents=documents,
...     batch_size=1,
... )

>>> queries_embeddings = retriever.encode_queries(
...     queries=queries,
...     batch_size=1,
... )

>>> retriever = retriever.add(
...     documents_embeddings=documents_embeddings,
... )

>>> scores = retriever(
...     queries_embeddings=queries_embeddings,
...     batch_size=32
... )

>>> pprint(scores)
[[{'id': 0, 'similarity': 62.01531219482422},
  {'id': 1, 'similarity': 59.01810836791992},
  {'id': 2, 'similarity': 40.613182067871094}],
 [{'id': 1, 'similarity': 97.81436920166016},
  {'id': 2, 'similarity': 32.50034713745117},
  {'id': 0, 'similarity': 25.678363800048828}],
 [{'id': 2, 'similarity': 56.019283294677734},
  {'id': 1, 'similarity': 37.612735748291016},
  {'id': 0, 'similarity': 26.307708740234375}]]
```

## Methods

???- note "__call__"

    Retrieve documents.

    **Parameters**

    - **queries_embeddings**     (*dict[str, scipy.sparse._csr.csr_matrix]*)    
    - **k**     (*int*)     – defaults to `None`    
    - **batch_size**     (*int*)     – defaults to `64`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    
???- note "add"

    Add documents embeddings and activations to the retriever.

    **Parameters**

    - **documents_embeddings**     (*dict[dict[str, torch.Tensor]]*)    
    
???- note "encode_documents"

    Encode documents.

    **Parameters**

    - **documents**     (*list[dict]*)    
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
    
???- note "top_k"

    Return the top k documents for each query.

    **Parameters**

    - **similarities**     (*scipy.sparse._csc.csc_matrix*)    
    - **k**     (*int*)    
    
