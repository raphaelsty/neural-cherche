import collections
import itertools

import torch

__all__ = ["in_batch_sparse_scores"]


def in_batch_sparse_scores(
    activations,
):
    """Computes dot product between anchor, positive and negative activations.

    Parameters
    ----------
    anchors_activations
        Activations of the anchors.
    other_activations
        Activations of the other documents.
    device
        Device to use.

    Examples
    --------

    >>> from sparsembed import utils

    >>> activations = torch.tensor([
    ...     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ...     [0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    ...     [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    ... ], device="mps")

    >>> utils.in_batch_sparse_scores(
    ...     activations=activations,
    ... )
    tensor([5, 8, 9], device='mps:0')

    """
    list_idx_a, list_idx_b = [], []
    size = activations.shape[0]

    index = collections.defaultdict(list)

    for row, (idx_a, idx_b) in enumerate(
        list(itertools.combinations(list(range(size)), 2))
    ):
        list_idx_a.append(idx_a)
        list_idx_b.append(idx_b)

        index[idx_a].append(row)
        index[idx_b].append(row)

    list_idx_a = torch.tensor(list_idx_a, device=activations.device)
    list_idx_b = torch.tensor(list_idx_b, device=activations.device)

    index = torch.tensor(list(index.values()), device=activations.device)

    sparse_activations_a = torch.index_select(
        input=activations, dim=0, index=list_idx_a
    )

    sparse_activations_b = torch.index_select(
        input=activations, dim=0, index=list_idx_b
    )

    sparse_scores = torch.sum(sparse_activations_a * sparse_activations_b, dim=1)

    return torch.gather(
        input=sparse_scores.repeat(size, sparse_activations_a.shape[0]),
        dim=1,
        index=index,
    ).sum(axis=1)
