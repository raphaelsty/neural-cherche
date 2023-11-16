# in_batch_sparse_scores

Computes dot product between anchor, positive and negative activations.



## Parameters

- **activations**



## Examples

```python
>>> from neural_cherche import utils

>>> activations = torch.tensor([
...     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
...     [0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
...     [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
... ], device="mps")

>>> utils.in_batch_sparse_scores(
...     activations=activations,
... )
tensor([5, 8, 9], device='mps:0')
```

