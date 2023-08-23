import torch

from .in_batch import in_batch_sparse_scores

__all__ = ["sparse_scores"]


def sparse_scores(
    anchor_activations: torch.Tensor,
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    in_batch_negatives: bool = True,
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

    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import model
    >>> from pprint import pprint as print

    >>> device = "mps"

    >>> model = model.Splade(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ...     device=device
    ... )

    >>> queries_activations = model(
    ...     ["Sports", "Music"],
    ... )

    >>> positive_activations = model(
    ...    ["Sports", "Music"],
    ... )

    >>> negative_activations = model(
    ...    ["Cinema", "Movie"],
    ... )

    >>> sparse_scores(
    ...     anchor_activations=queries_activations["sparse_activations"],
    ...     positive_activations=positive_activations["sparse_activations"],
    ...     negative_activations=negative_activations["sparse_activations"],
    ...     in_batch_negatives=True,
    ... )
    {'positive_scores': tensor([1147.8369,  668.3896], device='mps:0', grad_fn=<SumBackward1>), 'negative_scores': tensor([1244.4702, 1133.3767], device='mps:0', grad_fn=<AddBackward0>)}

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
