import torch

from .. import losses, utils

__all__ = ["train_sparsembed"]


def train_sparsembed(
    model,
    optimizer,
    anchor: list[str],
    positive: list[str],
    negative: list[str],
    flops_loss_weight: float = 1e-4,
    sparse_loss_weight: float = 0.1,
    dense_loss_weight: float = 1.0,
    in_batch_negatives: bool = True,
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

    >>> device = "mps"

    >>> model = model.SparsEmbed(
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

    >>> for anchor, positive, negative in utils.iter(
    ...         X,
    ...         epochs=3,
    ...         batch_size=3,
    ...         shuffle=False
    ...     ):
    ...     loss = train.train_sparsembed(
    ...         model=model,
    ...         optimizer=optimizer,
    ...         anchor=anchor,
    ...         positive=positive,
    ...         negative=negative,
    ...         flops_loss_weight=1e-4,
    ...         in_batch_negatives=True,
    ...     )

    >>> {'dense': tensor(4.9316, device='mps:0', grad_fn=<MeanBackward0>), 'ranking': tensor(3456.6538, device='mps:0', grad_fn=<MeanBackward0>), 'flops': tensor(796.2637, device='mps:0', grad_fn=<SumBackward1>)}
    """

    anchor_activations = model(
        anchor,
    )

    positive_activations = model(
        positive,
    )

    negative_activations = model(
        negative,
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

    sparse_ranking_loss = losses.Ranking()(**sparse_scores)

    flops_loss = losses.Flops()(
        anchor_activations=anchor_activations["sparse_activations"],
        positive_activations=positive_activations["sparse_activations"],
        negative_activations=negative_activations["sparse_activations"],
    )

    dense_ranking_loss = losses.Ranking()(**dense_scores)

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
        "ranking": sparse_ranking_loss,
        "flops": flops_loss,
    }
