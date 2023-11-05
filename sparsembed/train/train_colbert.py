from .. import losses, utils

__all__ = ["train_colbert"]


def train_colbert(
    model,
    optimizer,
    anchor: list[str],
    positive: list[str],
    negative: list[str],
    in_batch_negatives: bool = False,
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
    >>> from sparsembed import models, utils, train
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
    {'loss': tensor(0.0007, device='mps:0', grad_fn=<ClampBackward1>)}

    """

    anchor_embeddings = model(
        anchor,
        **kwargs,
    )

    positive_embeddings = model(
        positive,
        **kwargs,
    )

    negative_embeddings = model(
        negative,
        **kwargs,
    )

    scores = utils.colbert_scores(
        anchor_embeddings=anchor_embeddings,
        positive_embeddings=positive_embeddings,
        negative_embeddings=negative_embeddings,
        in_batch_negatives=in_batch_negatives,
    )

    loss = losses.Ranking()(**scores)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {"loss": loss}
