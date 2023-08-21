import os
import random

import tqdm

__all__ = ["iter"]


def iter(
    X: list[tuple[str, str, float]],
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
    ...    ("Apple", "🍏", "🍌"),
    ...    ("Banana", "🍌", "🍏"),
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
    ['🍌', '🍏']

    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for epoch in range(epochs):
        if shuffle:
            random.shuffle(X)

        for batch in tqdm.tqdm(
            [X[pos : pos + batch_size] for pos in range(0, len(X), batch_size)],
            position=0,
            total=1 + len(X) // batch_size,
            desc=f"Epoch {epoch}",
        ):
            yield [sample[0] for sample in batch], [sample[1] for sample in batch], [
                sample[2] for sample in batch
            ]
