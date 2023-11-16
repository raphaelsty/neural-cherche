# iter

Iterate over a list of tuples (query, document, score) by batch of size batch_size.



## Parameters

- **X** (*list[tuple[str, str, float]]*)

    List of tuples (query, document, score).

- **epochs** (*int*)

    Number of epochs.

- **batch_size** (*int*)

    Batch size.

- **shuffle** (*bool*) – defaults to `True`

    Shuffle the data.



## Examples

```python
>>> from neural_cherche import utils

>>> X = [
...    ("Apple", "🍏", "cherry"),
...    ("Banana", "🍌", "cherry"),
... ]

>>> for anchor, positive, negative in utils.iter(
...         X,
...         epochs=1,
...         batch_size=3,
...         shuffle=False
...     ):
...     break

>>> print(anchor)
['Apple', 'Banana']

>>> print(positive)
['🍏', '🍌']

>>> print(negative)
['cherry', 'cherry']
```

