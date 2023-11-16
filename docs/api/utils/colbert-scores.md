# colbert_scores

ColBERT scoring function for training.



## Parameters

- **anchor_embeddings** (*torch.Tensor*)

    Anchor embeddings.

- **positive_embeddings** (*torch.Tensor*)

    Positive embeddings.

- **negative_embeddings** (*torch.Tensor*)

    Negative embeddings.

- **in_batch_negatives** (*bool*) – defaults to `False`

    Whether to use in batch negatives or not. Defaults to False.



## Examples

```python
>>> from neural_cherche import models, utils
>>> import torch

>>> _ = torch.manual_seed(42)

>>> model = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-mpnet-base-v2",
...     device="mps",
... )

>>> anchor_embeddings = model(
...     ["Paris", "Toulouse"],
...     query_mode=True,
... )

>>> positive_embeddings = model(
...    ["Paris", "Toulouse"],
...    query_mode=False,
... )

>>> negative_embeddings = model(
...    ["Toulouse", "Paris"],
...    query_mode=False,
... )

>>> scores = utils.colbert_scores(
...     anchor_embeddings=anchor_embeddings["embeddings"],
...     positive_embeddings=positive_embeddings["embeddings"],
...     negative_embeddings=negative_embeddings["embeddings"],
... )

>>> scores
{'positive_scores': tensor([24.7555, 26.4455], device='mps:0', grad_fn=<SumBackward1>), 'negative_scores': tensor([18.3089, 17.1017], device='mps:0', grad_fn=<SumBackward1>)}
```

