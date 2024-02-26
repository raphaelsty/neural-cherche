import torch

from .. import losses, utils

__all__ = ["train_colbert"]


def train_colbert(
    model,
    optimizer,
    anchor: list[str],
    positive: list[str],
    negative: list[str],
    in_batch_negatives: bool = False,
    step: int = None,
    gradient_accumulation_steps: int = 50,
    **kwargs,
) -> dict[str, torch.Tensor]:
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
    step
        Training step, if specified, will enable gradient_accumulation_steps.
    gradient_accumulation_steps
        Gradient accumulation steps. Defaults to 50.

    Examples
    --------
    >>> from neural_cherche import models, utils, train
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.ColBERT(
    ...     model_name_or_path="raphaelsty/neural-cherche-colbert",
    ...     device="cpu",
    ... )

    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    >>> X = [
    ...     ("Sports", "Football", "Cinema"),
    ...     ("Sports", "Rugby", "Cinema"),
    ...     ("Sports", "Tennis", "Cinema"),
    ... ]

    >>> for step, (anchor, positive, negative) in enumerate(utils.iter(
    ...         X,
    ...         epochs=3,
    ...         batch_size=3,
    ...         shuffle=False
    ...     )):
    ...     loss = train.train_colbert(
    ...         model=model,
    ...         optimizer=optimizer,
    ...         anchor=anchor,
    ...         positive=positive,
    ...         negative=negative,
    ...         step=step,
    ...         gradient_accumulation_steps=2,
    ...         in_batch_negatives=True,
    ...     )

    >>> loss
    {'loss': tensor(1.0986, grad_fn=<NllLossBackward0>)}

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

    if step is not None:
        (loss / gradient_accumulation_steps).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    else:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {"loss": loss}
