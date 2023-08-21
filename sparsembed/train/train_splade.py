from .. import losses, utils

__all__ = ["train_splade"]


def train_splade(
    model,
    optimizer,
    anchor: list[str],
    positive: list[str],
    negative: list[str],
    flops_loss_weight: float = 1e-4,
    sparse_loss_weight: float = 1.0,
    in_batch_negatives: bool = True,
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
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import model, utils, train
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> device = "mps"

    >>> model = model.Splade(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ...     device=device
    ... )

    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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
    ...         flops_loss_weight=flops_scheduler(),
    ...         in_batch_negatives=True,
    ...     )

    >>> loss
    {'sparse': tensor(1582.0349, device='mps:0', grad_fn=<MeanBackward0>), 'flops': tensor(384.5024, device='mps:0', grad_fn=<SumBackward1>)}

    """

    anchor_activations = model(
        anchor,
        **kwargs,
    )

    positive_activations = model(
        positive,
        **kwargs,
    )

    negative_activations = model(
        negative,
        **kwargs,
    )

    scores = utils.sparse_scores(
        anchor_activations=anchor_activations["sparse_activations"],
        positive_activations=positive_activations["sparse_activations"],
        negative_activations=negative_activations["sparse_activations"],
        in_batch_negatives=in_batch_negatives,
    )

    ranking_loss = losses.Ranking()(**scores)

    flops_loss = losses.Flops()(
        anchor_activations=anchor_activations["sparse_activations"],
        positive_activations=positive_activations["sparse_activations"],
        negative_activations=negative_activations["sparse_activations"],
    )

    loss = sparse_loss_weight * ranking_loss + flops_loss_weight * flops_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {"sparse": ranking_loss, "flops": flops_loss}
