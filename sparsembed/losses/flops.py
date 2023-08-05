import torch

__all__ = ["Flops"]


class Flops(torch.nn.Module):
    """Flops loss, act as regularization loss over sparse activations.

    Example
    -------
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import model, losses
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = model.SparsEmbed(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased"),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ... )

    >>> anchor_queries_embeddings = model.encode(
    ...     ["Paris", "Toulouse"],
    ...     k=64
    ... )

    >>> positive_documents_embeddings = model.encode(
    ...     ["France", "France"],
    ...     k=64
    ... )

    >>> negative_documents_embeddings = model.encode(
    ...     ["Canada", "Espagne"],
    ...     k=64
    ... )

    >>> flops = losses.Flops()

    >>> loss = flops(
    ...     sparse_activations = anchor_queries_embeddings["sparse_activations"]
    ... )

    >>> loss += flops(
    ...     sparse_activations = torch.cat([
    ...         positive_documents_embeddings["sparse_activations"],
    ...         negative_documents_embeddings["sparse_activations"],
    ...     ], dim=0)
    ... )


    References
    ----------
    1. [MINIMIZING FLOPS TO LEARN EFFICIENT SPARSE REPRESENTATIONS](https://arxiv.org/pdf/2004.05665.pdf)
    2. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/pdf/2107.05720.pdf)

    """

    def __init__(self):
        super(Flops, self).__init__()

    def __call__(
        self,
        sparse_activations: torch.Tensor,
    ) -> torch.Tensor:
        """Loss which tend to reduce sparse activation."""
        return torch.sum(torch.mean(sparse_activations, dim=0) ** 2, dim=0)
