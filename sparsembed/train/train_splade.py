from .. import losses, utils

__all__ = ["train_splade"]


def train_splade(
    model,
    optimizer,
    anchor: list[str],
    positive: list[str],
    negative: list[str],
    flops_loss_weight: float = 1e-5,
    sparse_loss_weight: float = 1.0,
    in_batch_negatives: bool = False,
    threshold_flops: float = 30,
    max_loss: float = 10.0,
    **kwargs,
):
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

    Examples
    --------
    >>> from sparsembed import models, utils, train
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.Splade(
    ...     model_name_or_path="distilbert-base-uncased",
    ...     device="mps",
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
    ...         in_batch_negatives=False,
    ...     )
    ...     flops_scheduler.step()

    >>> loss
    {'sparse': tensor(0., device='mps:0', grad_fn=<ClampBackward1>), 'flops': tensor(10., device='mps:0', grad_fn=<ClampBackward1>)}

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
        max_loss=max_loss,
    )

    loss = sparse_loss_weight * sparse_loss + flops_loss_weight * flops_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {"sparse": sparse_loss, "flops": flops_loss}
