import os
import random

import torch
import tqdm

__all__ = ["iter"]


def iter(
    X: list[tuple[str, str, float]],
    device: str,
    epochs: int,
    batch_size: int,
    shuffle: bool = True,
) -> tuple[list[str], list[str], list[float]]:
    """Iterate over a list of tuples (query, document, score) by batch of size batch_size.

    Parameters
    ----------
    X
        List of tuples (query, document, score).
    batch_size
        Batch size.

    Example
    -------
    >>> from sparsembed import utils
    >>> from pprint import pprint as print

    >>> X = [
    ...    ("Apple", "Apple is a popular fruit.", 1),
    ...    ("Apple", "Banana is a popular fruit.", 0),
    ...    ("Banana", "Apple is a popular fruit.", 0),
    ...    ("Banana", "Banana is a yellow fruit.", 1),
    ... ]

    >>> for queries, documents, labels in utils.iter(
    ...         X,
    ...         device="cpu",
    ...         epochs=1,
    ...         batch_size=3,
    ...         shuffle=False
    ...     ):
    ...     break

    >>> print(queries)
    ['Apple', 'Apple', 'Banana']

    >>> print(documents)
    ['Apple is a popular fruit.',
     'Banana is a popular fruit.',
     'Apple is a popular fruit.']

    >>> print(labels)
    tensor([1., 0., 0.])

    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for epoch in range(epochs):
        if shuffle:
            random.shuffle(X)

        for batch in tqdm.tqdm(
            [X[pos : pos + batch_size] for pos in range(0, len(X), batch_size)],
            position=0,
            total=1 + len(X) // batch_size,
            desc=f"Epoch {epoch + 1}/{epochs}",
        ):
            yield [sample[0] for sample in batch], [
                sample[1] for sample in batch
            ], torch.tensor(
                [sample[2] for sample in batch], dtype=torch.float, device=device
            )
