import torch

__all__ = ["Flops", "FlopsScheduler"]


class FlopsScheduler:
    """Flops scheduler.

    References
    ----------
    1. [MINIMIZING FLOPS TO LEARN EFFICIENT SPARSE REPRESENTATIONS](https://arxiv.org/pdf/2004.05665.pdf)
    2. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/pdf/2107.05720.pdf)
    """

    def __init__(self, weight: float = 3e-6, steps: int = 10000):
        self._weight = weight
        self.weight = 0.0
        self.steps = steps
        self._step = 1

    def get(self) -> float:
        if self._step >= self.steps:
            pass
        else:
            self._step += 1
            self.weight = self._weight * (self._step / self.steps) ** 2
        return self.weight


class Flops(torch.nn.Module):
    """Flops loss, act as regularization loss over sparse activations.

    Examples
    --------
    >>> from neural_cherche import models, utils, losses
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.Splade(
    ...     model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
    ...     device="cpu",
    ... )

    >>> anchor_activations = model(
    ...     ["Sports", "Music"],
    ...     query_mode=True,
    ... )

    >>> positive_activations = model(
    ...    ["Sports", "Music"],
    ...     query_mode=False,
    ... )

    >>> negative_activations = model(
    ...    ["Cinema", "Movie"],
    ...     query_mode=False,
    ... )

    >>> losses.Flops()(
    ...     anchor_activations=anchor_activations["sparse_activations"],
    ...     positive_activations=positive_activations["sparse_activations"],
    ...     negative_activations=negative_activations["sparse_activations"],
    ... )
    tensor(1., grad_fn=<ClampBackward1>)

    References
    ----------
    1. [MINIMIZING FLOPS TO LEARN EFFICIENT SPARSE REPRESENTATIONS](https://arxiv.org/pdf/2004.05665.pdf)
    2. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/pdf/2107.05720.pdf)

    """

    def __init__(self) -> None:
        super(Flops, self).__init__()

    def __call__(
        self,
        anchor_activations: torch.Tensor,
        positive_activations: torch.Tensor,
        negative_activations: torch.Tensor,
        threshold: float = 30.0,
        max_flops_loss: float = 1.0,
    ) -> torch.Tensor:
        """Loss which tend to reduce sparse activation."""
        activations = torch.cat(
            tensors=[anchor_activations, positive_activations, negative_activations],
            dim=0,
        )

        return torch.abs(
            input=threshold
            - torch.sum(
                input=torch.mean(input=torch.abs(input=activations), dim=0) ** 2, dim=0
            )
        ).clip(min=0.0, max=max_flops_loss)
