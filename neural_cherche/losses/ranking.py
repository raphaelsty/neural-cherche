import torch

__all__ = ["Ranking"]


class Ranking(torch.nn.Module):
    """Ranking loss.

    Examples
    --------
    >>> from neural_cherche import models, utils, losses
    >>> from pprint import pprint as print

    >>> model = models.Splade(
    ...     model_name_or_path="raphaelsty/splade-max",
    ...     device="cpu",
    ... )

    >>> queries_activations = model(
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

    >>> scores = utils.sparse_scores(
    ...     anchor_activations=queries_activations["sparse_activations"],
    ...     positive_activations=positive_activations["sparse_activations"],
    ...     negative_activations=negative_activations["sparse_activations"],
    ...     in_batch_negatives=True,
    ... )

    >>> losses.Ranking()(**scores)
    tensor(6.8423e-05, grad_fn=<NllLossBackward0>)

    References
    ----------
    1. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/pdf/2107.05720.pdf)

    """

    def __init__(self) -> None:
        super(Ranking, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def __call__(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Ranking loss."""
        positive_scores = positive_scores.unsqueeze(dim=-1)
        if negative_scores.ndim == 1:
            negative_scores = negative_scores.unsqueeze(dim=-1)

        scores = torch.cat(
            tensors=[
                positive_scores,
                negative_scores,
            ],
            dim=-1,
        )

        return self.cross_entropy(
            scores,
            torch.zeros(
                scores.shape[0],
                dtype=torch.long,
                device=scores.device,
            ),
        )
