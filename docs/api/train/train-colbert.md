# train_colbert

Compute the ranking loss and the flops loss for a single step.



## Parameters

- **model**

    Colbert model.

- **optimizer**

    Optimizer.

- **anchor** (*list[str]*)

    Anchor.

- **positive** (*list[str]*)

    Positive.

- **negative** (*list[str]*)

    Negative.

- **in_batch_negatives** (*bool*) – defaults to `False`

    Whether to use in batch negatives or not. Defaults to True.

- **kwargs**



## Examples

```python
>>> from neural_cherche import models, utils, train
>>> import torch

>>> _ = torch.manual_seed(42)

>>> device = "mps"

>>> model = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-mpnet-base-v2",
...     device=device
... )

>>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

>>> X = [
...     ("Sports", "Football", "Cinema"),
...     ("Sports", "Rugby", "Cinema"),
...     ("Sports", "Tennis", "Cinema"),
... ]

>>> for anchor, positive, negative in utils.iter(
...         X,
...         epochs=3,
...         batch_size=3,
...         shuffle=False
...     ):
...     loss = train.train_colbert(
...         model=model,
...         optimizer=optimizer,
...         anchor=anchor,
...         positive=positive,
...         negative=negative,
...         in_batch_negatives=False,
...     )

>>> loss
{'loss': tensor(0.0054, device='mps:0', grad_fn=<ClampBackward1>)}
```

