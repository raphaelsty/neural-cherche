import os
import random

import tqdm

__all__ = ["batchify", "iter", "iter_triples"]


def batchify(
    X: list[str], batch_size: int, desc: str = "", tqdm_bar: bool = True
) -> list:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    batchs = [X[pos : pos + batch_size] for pos in range(0, len(X), batch_size)]

    if tqdm_bar:
        for batch in tqdm.tqdm(
            batchs,
            position=0,
            total=1 + len(X) // batch_size,
            desc=desc,
        ):
            yield batch
    else:
        yield from batchs


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
    epochs
        Number of epochs.
    batch_size
        Batch size.
    shuffle
        Shuffle the data.

    Example
    -------
    >>> from sparsembed import utils

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

    """
    for epoch in range(epochs):
        if shuffle:
            random.shuffle(X)
        for batch in batchify(X=X, batch_size=batch_size, desc=f"Epoch {epoch}"):
            yield [sample[0] for sample in batch], [sample[1] for sample in batch], [
                sample[2] for sample in batch
            ]


def iter_triples(
    X: list[tuple[str, str, float]],
    epochs: int,
    batch_size: int,
    shuffle: bool = True,
) -> tuple[list[str], list[str], list[str], list[float]]:
    """Iterate over a list of triples (head, relation, tail, score) by batch of size batch_size.

    Parameters
    ----------
    X
        List of triples (head, relation, tail, score).
    epochs
        Number of epochs.
    batch_size
        Batch size.
    shuffle
        Shuffle the data.

    Example
    -------
    >>> from sparsembed import utils

    >>> X = [
    ...    ("Apple", "emoji", "🍏", "🍒"),
    ...    ("Apple", "emoji", "🍌", "🍒"),
    ...    ("Banana", "emoji", "🍏", "🍒"),
    ... ]

    >>> for anchor, relation, positive, negative in utils.iter_triples(
    ...         X,
    ...         epochs=1,
    ...         batch_size=3,
    ...         shuffle=False
    ...     ):
    ...     break

    >>> print(anchor)
    ['Apple', 'Apple', 'Banana']

    >>> print(relation)
    ['emoji', 'emoji', 'emoji']

    >>> print(positive)
    ['🍏', '🍌', '🍏']

    >>> print(negative)
    ['🍒', '🍒', '🍒']

    """
    for epoch in range(epochs):
        if shuffle:
            random.shuffle(X)
        for batch in batchify(X=X, batch_size=batch_size, desc=f"Epoch {epoch}"):
            yield [sample[0] for sample in batch], [sample[1] for sample in batch], [
                sample[2] for sample in batch
            ], [sample[3] for sample in batch]
