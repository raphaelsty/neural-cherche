import torch

__all__ = ["Cosine"]


class Cosine(torch.nn.Module):
    """Cosine similarity loss function between sparse vectors.

    Parameters
    ----------
    l
        Lambda to ponderate the Cosine loss.

    Example
    -------
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import losses, model, utils
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> loss = 0

    >>> model = model.SparsEmbed(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased"),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ... )

    >>> queries_embeddings = model(
    ...     ["Paris", "Toulouse"],
    ...     k=96
    ... )

    >>> documents_embeddings = model(
    ...    ["Paris is a city located in France.", "Toulouse is a city located in France."],
    ...     k=256
    ... )

    >>> scores = utils.scores(
    ...     queries_activations=queries_embeddings["activations"],
    ...     queries_embeddings=queries_embeddings["embeddings"],
    ...     documents_activations=documents_embeddings["activations"],
    ...     documents_embeddings=documents_embeddings["embeddings"],
    ...     device="cpu",
    ... )

    >>> cosine_loss = losses.Cosine()

    >>> loss += cosine_loss.sparse(
    ...     queries_sparse_activations=queries_embeddings["sparse_activations"],
    ...     documents_sparse_activations=documents_embeddings["sparse_activations"],
    ...     labels=torch.tensor([1,1]),
    ... )

    >>> loss += cosine_loss.dense(
    ...     scores=scores,
    ...     labels=torch.tensor([1, 1]),
    ... )

    References
    ----------
    1. [Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation](https://arxiv.org/pdf/2010.02666.pdf)

    """

    def __init__(
        self,
    ) -> None:
        super(Cosine, self).__init__()
        self.mse = torch.nn.MSELoss(reduction="none")

    def sparse(
        self,
        queries_sparse_activations: torch.Tensor,
        documents_sparse_activations: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Sparse CosineSimilarity loss."""
        similarity = torch.cosine_similarity(
            queries_sparse_activations, documents_sparse_activations, dim=1
        )
        errors = self.mse(similarity, labels)
        if weights is not None:
            errors *= weights
        return errors.mean()

    def dense(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        errors = self.mse(scores, labels)
        if weights is not None:
            errors *= weights
        return errors.mean()
