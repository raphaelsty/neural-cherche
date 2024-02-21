from .. import losses, utils

__all__ = ["train_colbert"]


def train_colbert(
    model,
    optimizer,
    anchor: list[str],
    positive: list[str],
    negative: list[str],
    in_batch_negatives: bool = False,
    backward: bool = True,
    **kwargs,
):
    """Compute the ranking loss and the flops loss for a single step.

    Parameters
    ----------
    model
        Colbert model.
    optimizer
        Optimizer.
    anchor
        Anchor.
    positive
        Positive.
    negative
        Negative.
    in_batch_negatives
        Whether to use in batch negatives or not. Defaults to True.

    Examples
    --------
    >>> from neural_cherche import models, utils, train
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> device = "mps"

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    ...     device=device
    ... )

    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    >>> X = [
    ...     ("Sports", "Football", "Cinema"),
    ...     ("Sports", "Rugby", "Cinema"),
    ...     ("Sports", "Tennis", "Cinema"),
    ... ]

    >>> for anchor, positive, negative in utils.iter(
    ...         X,
    ...         epochs=3,
    ...         batch_size=3,
    ...         shuffle=False
    ...     ):
    ...     loss = train.train_colbert(
    ...         model=model,
    ...         optimizer=optimizer,
    ...         anchor=anchor,
    ...         positive=positive,
    ...         negative=negative,
    ...         in_batch_negatives=False,
    ...     )

    >>> loss
    {'loss': tensor(0.0054, device='mps:0', grad_fn=<ClampBackward1>)}

    """

    anchor_embeddings = model(
        anchor,
        query_mode=True,
        **kwargs,
    )

    positive_embeddings = model(
        positive,
        query_mode=False,
        **kwargs,
    )

    negative_embeddings = model(
        negative,
        query_mode=False,
        **kwargs,
    )

    scores = utils.colbert_scores(
        anchor_embeddings=anchor_embeddings["embeddings"],
        positive_embeddings=positive_embeddings["embeddings"],
        negative_embeddings=negative_embeddings["embeddings"],
        in_batch_negatives=in_batch_negatives,
    )

    loss = losses.Ranking()(**scores)

    if backward:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return {"loss": loss}
