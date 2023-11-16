# dense_scores

Computes score between queries and documents intersected activated tokens.



## Parameters

- **anchor_activations** (*torch.Tensor*)

- **positive_activations** (*torch.Tensor*)

- **negative_activations** (*torch.Tensor*)

- **anchor_embeddings** (*torch.Tensor*)

- **positive_embeddings** (*torch.Tensor*)

- **negative_embeddings** (*torch.Tensor*)

- **func** – defaults to `<built-in method sum of type object at 0x107e07280>`

    Either torch.sum or torch.mean. torch.mean is dedicated to training and torch.sum is dedicated to inference.



## Examples

```python
>>> from neural_cherche import models, utils
>>> import torch

>>> _ = torch.manual_seed(42)

>>> model = models.SparseEmbed(
...     model_name_or_path="distilbert-base-uncased",
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

>>> scores = utils.dense_scores(
...     anchor_activations=anchor_embeddings["activations"],
...     positive_activations=positive_embeddings["activations"],
...     negative_activations=negative_embeddings["activations"],
...     anchor_embeddings=anchor_embeddings["embeddings"],
...     positive_embeddings=positive_embeddings["embeddings"],
...     negative_embeddings=negative_embeddings["embeddings"],
...     func=torch.sum,
... )

>>> scores
{'positive_scores': tensor([144.4106, 155.5398], device='mps:0', grad_fn=<StackBackward0>), 'negative_scores': tensor([173.4966,  99.9521], device='mps:0', grad_fn=<StackBackward0>)}
```

