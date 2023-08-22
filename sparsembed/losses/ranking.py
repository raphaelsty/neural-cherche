import torch

__all__ = ["Ranking"]


class Ranking(torch.nn.Module):
    """Ranking loss.

    Examples
    --------
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import model, utils, losses
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

    >>> scores = utils.sparse_scores(
    ...     anchor_activations=queries_activations["sparse_activations"],
    ...     positive_activations=positive_activations["sparse_activations"],
    ...     negative_activations=negative_activations["sparse_activations"],
    ...     in_batch_negatives=True,
    ... )

    >>> losses.Ranking()(**scores)
    tensor(1., device='mps:0', grad_fn=<ClampBackward1>)

    References
    ----------
    1. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/pdf/2107.05720.pdf)

    """

    def __init__(self):
        super(Ranking, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def __call__(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Ranking loss."""
        scores = torch.stack(
            [
                positive_scores,
                negative_scores,
            ],
            dim=1,
        )

        loss = torch.index_select(
            input=-self.log_softmax(scores),
            dim=1,
            index=torch.zeros(1, dtype=torch.int64).to(scores.device),
        ).mean()

        return torch.clip(loss, min=0.0, max=1.0)
