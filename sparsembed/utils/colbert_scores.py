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
    """
    positive_scores = torch.einsum(
        "bsh,bth->bst", anchor_embeddings, positive_embeddings
    )

    negative_scores = torch.einsum(
        "bsh,bth->bst", anchor_embeddings, negative_embeddings
    )

    return {
        "positive_scores": torch.max(positive_scores, axis=2).values.sum(axis=1),
        "negative_scores": torch.max(negative_scores, axis=2).values.sum(axis=1),
    }
