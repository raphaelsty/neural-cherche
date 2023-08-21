import torch

__all__ = ["Flops", "FlopsScheduler"]


class FlopsScheduler:
    """Flops scheduler."""

    def __init__(self, weight: float = 3e-5, steps: int = 10000):
        self._weight = weight
        self.weight = 0
        self.steps = steps
        self.step = 0

    def __call__(self):
        if self.step >= self.steps:
            pass
        else:
            self.step += 1
            self.weight = self._weight * (self.step / self.steps) ** 2
        return self.weight


class Flops(torch.nn.Module):
    """Flops loss, act as regularization loss over sparse activations.

    Example
    -------
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import model, utils, losses
    >>> from pprint import pprint as print
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> device = "mps"

    >>> model = model.Splade(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ...     device=device
    ... )

    >>> anchor_activations = model(
    ...     ["Sports", "Music"],
    ... )

    >>> positive_activations = model(
    ...    ["Sports", "Music"],
    ... )

    >>> negative_activations = model(
    ...    ["Cinema", "Movie"],
    ... )

    >>> losses.Flops()(
    ...     anchor_activations=anchor_activations["sparse_activations"],
    ...     positive_activations=positive_activations["sparse_activations"],
    ...     negative_activations=negative_activations["sparse_activations"],
    ... )
    tensor(1880.5656, device='mps:0', grad_fn=<SumBackward1>)

    References
    ----------
    1. [MINIMIZING FLOPS TO LEARN EFFICIENT SPARSE REPRESENTATIONS](https://arxiv.org/pdf/2004.05665.pdf)
    2. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/pdf/2107.05720.pdf)

    """

    def __init__(self):
        super(Flops, self).__init__()

    def __call__(
        self,
        anchor_activations: torch.Tensor,
        positive_activations: torch.Tensor,
        negative_activations: torch.Tensor,
    ) -> torch.Tensor:
        """Loss which tend to reduce sparse activation."""
        activations = torch.cat(
            [anchor_activations, positive_activations, negative_activations], dim=0
        )
        return torch.sum(torch.mean(torch.abs(activations), dim=0) ** 2, dim=0)
