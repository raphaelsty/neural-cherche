import torch

from .. import losses, utils

__all__ = ["train_splade"]


def train_splade(
    model,
    optimizer,
    anchor: list[str],
    positive: list[str],
    negative: list[str],
    flops_loss_weight: float = 3e-6,
    sparse_loss_weight: float = 1.0,
    in_batch_negatives: bool = False,
    threshold_flops: float = 100.0,
    max_flops_loss: float = 10.0,
    step: int = None,
    gradient_accumulation_steps: int = 50,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Compute the ranking loss and the flops loss for a single step.

    Parameters
    ----------
    model
        Splade model.
    optimizer
        Optimizer.
    anchor
        Anchor.
    positive
        Positive.
    negative
        Negative.
    flops_loss_weight
        Flops loss weight. Defaults to 1e-4.
    in_batch_negatives
        Whether to use in batch negatives or not. Defaults to True.
    step
        Training step, if specified, will enable gradient_accumulation_steps.
    gradient_accumulation_steps
        Gradient accumulation steps. Defaults to 50.

    Examples
    --------
    >>> from neural_cherche import models, utils, train
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.Splade(
    ...     model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
    ...     device="cpu",
    ... )

    >>> optimizer = torch.optim.AdamW(
    ...     model.parameters(),
    ...     lr=1e-6,
    ... )

    >>> X = [
    ...     ("Sports", "Music", "Cinema"),
    ...     ("Sports", "Music", "Cinema"),
    ...     ("Sports", "Music", "Cinema"),
    ... ]

    >>> flops_scheduler = losses.FlopsScheduler()

    >>> for anchor, positive, negative in utils.iter(
    ...         X,
    ...         epochs=3,
    ...         batch_size=3,
    ...         shuffle=False
    ...     ):
    ...     loss = train.train_splade(
    ...         model=model,
    ...         optimizer=optimizer,
    ...         anchor=anchor,
    ...         positive=positive,
    ...         negative=negative,
    ...         flops_loss_weight=flops_scheduler.get(),
    ...         in_batch_negatives=True,
    ...     )

    >>> loss
    {'sparse': tensor(0.7054, grad_fn=<NllLossBackward0>), 'flops': tensor(10., grad_fn=<ClampBackward1>), 'loss': tensor(0.7054, grad_fn=<AddBackward0>)}

    """

    anchor_activations = model(
        anchor,
        query_mode=True,
        **kwargs,
    )

    positive_activations = model(
        positive,
        query_mode=False,
        **kwargs,
    )

    negative_activations = model(
        negative,
        query_mode=False,
        **kwargs,
    )

    scores = utils.sparse_scores(
        anchor_activations=anchor_activations["sparse_activations"],
        positive_activations=positive_activations["sparse_activations"],
        negative_activations=negative_activations["sparse_activations"],
        in_batch_negatives=in_batch_negatives,
    )

    sparse_loss = losses.Ranking()(**scores)

    flops_loss = losses.Flops()(
        anchor_activations=anchor_activations["sparse_activations"],
        positive_activations=positive_activations["sparse_activations"],
        negative_activations=negative_activations["sparse_activations"],
        threshold=threshold_flops,
        max_flops_loss=max_flops_loss,
    )

    loss = sparse_loss_weight * sparse_loss + flops_loss_weight * flops_loss

    if step is not None:
        (loss / gradient_accumulation_steps).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    else:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {"sparse": sparse_loss, "flops": flops_loss, "loss": loss}
