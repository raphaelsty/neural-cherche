import torch

from .in_batch import in_batch_sparse_scores

__all__ = ["sparse_scores"]


def sparse_scores(
    anchor_activations: torch.Tensor,
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    in_batch_negatives: bool = False,
):
    """Computes dot product between anchor, positive and negative activations.

    Parameters
    ----------
    anchor_activations
        Activations of the anchors.
    positive_activations
        Activations of the positive documents.
    negative_activations
        Activations of the negative documents.
    in_batch_negatives
        Whether to use in batch negatives or not. Defaults to True.
        Sum up with negative scores the dot product.

    Examples
    --------
    >>> from sparsembed import models

    >>> model = models.Splade(
    ...     model_name_or_path="distilbert-base-uncased",
    ...     device="mps"
    ... )

    >>> anchor_activations = model(
    ...     ["Sports", "Music"],
    ...     query_mode=True,
    ... )

    >>> positive_activations = model(
    ...    ["Sports", "Music"],
    ...    query_mode=False,
    ... )

    >>> negative_activations = model(
    ...    ["Cinema", "Movie"],
    ...    query_mode=False,
    ... )

    >>> sparse_scores(
    ...     anchor_activations=anchor_activations["sparse_activations"],
    ...     positive_activations=positive_activations["sparse_activations"],
    ...     negative_activations=negative_activations["sparse_activations"],
    ... )
    {'positive_scores': tensor([356.5258, 348.1818], device='mps:0', grad_fn=<SumBackward1>), 'negative_scores': tensor([312.3294, 286.0956], device='mps:0', grad_fn=<SumBackward1>)}

    """
    positive_scores = torch.sum(anchor_activations * positive_activations, axis=1)
    negative_scores = torch.sum(anchor_activations * negative_activations, axis=1)

    if in_batch_negatives:
        negative_scores += in_batch_sparse_scores(
            activations=positive_activations,
        )

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
    }
