# train_sparse_embed

Compute the ranking loss and the flops loss for a single step.



## Parameters

- **model**

    Splade model.

- **optimizer**

    Optimizer.

- **anchor** (*list[str]*)

    Anchor.

- **positive** (*list[str]*)

    Positive.

- **negative** (*list[str]*)

    Negative.

- **flops_loss_weight** (*float*) – defaults to `0.0001`

    Flops loss weight. Defaults to 1e-5.

- **sparse_loss_weight** (*float*) – defaults to `0.1`

    Sparse loss weight. Defaults to 1.0.

- **dense_loss_weight** (*float*) – defaults to `1.0`

    Dense loss weight. Defaults to 1.0.

- **in_batch_negatives** (*bool*) – defaults to `False`

    Whether to use in batch negatives or not. Defaults to True.

- **threshold_flops** (*float*) – defaults to `30`

    Threshold margin for the flops loss. Defaults to 10.

- **kwargs**



## Examples

```python
>>> from neural_cherche import models, utils, train
>>> import torch

>>> _ = torch.manual_seed(42)

>>> model = models.SparseEmbed(
...     model_name_or_path="distilbert-base-uncased",
...     device="mps",
... )

>>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

>>> X = [
...     ("Sports", "Music", "Cinema"),
...     ("Sports", "Music", "Cinema"),
...     ("Sports", "Music", "Cinema"),
... ]

>>> flops_scheduler = losses.FlopsScheduler()

>>> for anchor, positive, negative in utils.iter(
...         X,
...         epochs=3,
...         batch_size=3,
...         shuffle=False
...     ):
...     loss = train.train_sparse_embed(
...         model=model,
...         optimizer=optimizer,
...         anchor=anchor,
...         positive=positive,
...         negative=negative,
...         flops_loss_weight=flops_scheduler.get(),
...         in_batch_negatives=False,
...     )
...     flops_scheduler.step()

>>> loss
{'dense': tensor(0.0015, device='mps:0', grad_fn=<ClampBackward1>), 'sparse': tensor(1.1921e-07, device='mps:0', grad_fn=<ClampBackward1>), 'flops': tensor(10., device='mps:0', grad_fn=<ClampBackward1>)}
```

