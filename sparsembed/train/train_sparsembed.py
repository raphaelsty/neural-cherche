import torch

from .. import losses, utils

__all__ = ["train_sparsembed"]


def train_sparsembed(
    model,
    optimizer,
    anchor: list[str],
    positive: list[str],
    negative: list[str],
    k_tokens: int = 256,
    flops_loss_weight: float = 1e-4,
    sparse_loss_weight: float = 0.1,
    dense_loss_weight: float = 1.0,
    in_batch_negatives: bool = False,
    threshold_flops: float = 10,
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
    sparse_loss_weight
        Sparse loss weight. Defaults to 0.1.
    dense_loss_weight
        Dense loss weight. Defaults to 1.0.
    in_batch_negatives
        Whether to use in batch negatives or not. Defaults to True.
    threshold_flops
        Threshold margin for the flops loss. Defaults to 10.

    Examples
    --------
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import model, utils, train
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> device = "mps"

    >>> model = model.SparsEmbed(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ...     device=device
    ... )

    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

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
    ...     loss = train.train_sparsembed(
    ...         model=model,
    ...         optimizer=optimizer,
    ...         k_tokens=96,
    ...         anchor=anchor,
    ...         positive=positive,
    ...         negative=negative,
    ...         flops_loss_weight=flops_scheduler.get(),
    ...         in_batch_negatives=False,
    ...     )
    ...     flops_scheduler.step()

    >>> loss
    {'dense': tensor(1., device='mps:0', grad_fn=<ClampBackward1>), 'sparse': tensor(0.0101, device='mps:0', grad_fn=<ClampBackward1>), 'flops': tensor(157.9349, device='mps:0', grad_fn=<AbsBackward0>)}

    """

    anchor_activations = model(
        anchor,
        k_tokens=k_tokens,
        **kwargs,
    )

    positive_activations = model(
        positive,
        k_tokens=k_tokens,
        **kwargs,
    )

    negative_activations = model(
        negative,
        k_tokens=k_tokens,
        **kwargs,
    )

    sparse_scores = utils.sparse_scores(
        anchor_activations=anchor_activations["sparse_activations"],
        positive_activations=positive_activations["sparse_activations"],
        negative_activations=negative_activations["sparse_activations"],
        in_batch_negatives=in_batch_negatives,
    )

    dense_scores = utils.dense_scores(
        anchor_activations=anchor_activations["activations"],
        positive_activations=positive_activations["activations"],
        negative_activations=negative_activations["activations"],
        anchor_embeddings=anchor_activations["embeddings"],
        positive_embeddings=positive_activations["embeddings"],
        negative_embeddings=negative_activations["embeddings"],
        func=torch.sum,
    )

    if (
        dense_scores["positive_scores"] is None
        or dense_scores["negative_scores"] is None
    ):
        dense_ranking_loss = torch.tensor(0.0, device=model.device)
    else:
        dense_ranking_loss = losses.Ranking()(**dense_scores)

    sparse_ranking_loss = losses.Ranking()(**sparse_scores)

    flops_loss = losses.Flops()(
        anchor_activations=anchor_activations["sparse_activations"],
        positive_activations=positive_activations["sparse_activations"],
        negative_activations=negative_activations["sparse_activations"],
        threshold=threshold_flops,
    )

    loss = (
        dense_loss_weight * dense_ranking_loss
        + sparse_loss_weight * sparse_ranking_loss
        + flops_loss_weight * flops_loss
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {
        "dense": dense_ranking_loss,
        "sparse": sparse_ranking_loss,
        "flops": flops_loss,
    }
