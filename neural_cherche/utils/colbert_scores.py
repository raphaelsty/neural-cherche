import torch

__all__ = ["colbert_scores"]


def colbert_scores(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    in_batch_negatives: bool = False,
) -> torch.Tensor:
    """ColBERT scoring function for training.

    Parameters
    ----------
    anchor_embeddings
        Anchor embeddings.
    positive_embeddings
        Positive embeddings.
    negative_embeddings
        Negative embeddings.
    in_batch_negatives
        Whether to use in batch negatives or not. Defaults to False.


    Examples
    --------
    >>> from neural_cherche import models, utils
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.ColBERT(
    ...     model_name_or_path="raphaelsty/neural-cherche-colbert",
    ...     device="mps",
    ... )

    >>> anchor_embeddings = model(
    ...     ["Paris", "Toulouse"],
    ...     query_mode=True,
    ... )

    >>> positive_embeddings = model(
    ...    ["Paris", "Toulouse"],
    ...    query_mode=False,
    ... )

    >>> negative_embeddings = model(
    ...    ["Toulouse", "Paris"],
    ...    query_mode=False,
    ... )

    >>> scores = utils.colbert_scores(
    ...     anchor_embeddings=anchor_embeddings["embeddings"],
    ...     positive_embeddings=positive_embeddings["embeddings"],
    ...     negative_embeddings=negative_embeddings["embeddings"],
    ... )

    >>> scores
    {'positive_scores': tensor([24.7555, 26.4455], device='mps:0', grad_fn=<SumBackward1>), 'negative_scores': tensor([18.3089, 17.1017], device='mps:0', grad_fn=<SumBackward1>)}

    """
    positive_scores = (
        torch.einsum("bsh,bth->bst", anchor_embeddings, positive_embeddings)
        .max(axis=2)
        .values.sum(axis=1)
    )

    negative_scores = (
        torch.einsum("bsh,bth->bst", anchor_embeddings, negative_embeddings)
        .max(axis=2)
        .values.sum(axis=1)
    )

    if in_batch_negatives:
        in_batch_negative_scores = (
            torch.einsum(
                "bsh,bth->bst",
                anchor_embeddings,
                positive_embeddings.roll(shifts=1, dims=0),
            )
            .max(axis=2)
            .values.sum(axis=1)
        )

        negative_scores = torch.stack(
            tensors=[negative_scores, in_batch_negative_scores], axis=1
        )

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
    }
