import torch

__all__ = ["scores"]


def _build_index(activations: torch.Tensor, embeddings: torch.Tensor) -> dict:
    """Build index to score documents using activated tokens and embeddings."""
    index = []
    for tokens_activation, tokens_embeddings in zip(activations, embeddings):
        index.append(
            {
                token.item(): embedding
                for token, embedding in zip(tokens_activation, tokens_embeddings)
            }
        )
    return index


def _intersection(t1: torch.Tensor, t2: torch.Tensor) -> list:
    """Retrieve intersection between two tensors."""
    t1, t2 = t1.flatten(), t2.flatten()
    combined = torch.cat((t1, t2), dim=0)
    uniques, counts = combined.unique(return_counts=True, sorted=False)
    return uniques[counts > 1].tolist()


def _get_intersection(queries_activations: list, documents_activations: list) -> list:
    """Retrieve intersection of activated tokens between queries and documents."""
    return [
        _intersection(query_activations, document_activations)
        for query_activations, document_activations in zip(
            queries_activations,
            documents_activations,
        )
    ]


def _get_scores(
    queries_embeddings_index: torch.Tensor,
    documents_embeddings_index: torch.Tensor,
    intersections: torch.Tensor,
    device: str,
    func,
) -> list:
    """Computes similarity scores between queries and documents based on activated tokens embeddings"""
    return torch.stack(
        [
            func(
                torch.stack(
                    [document_embeddings_index[token] for token in intersection], dim=0
                )
                * torch.stack(
                    [query_embeddings_index[token] for token in intersection], dim=0
                )
            )
            if len(intersection) > 0
            else torch.tensor(0.0, device=device)
            for query_embeddings_index, document_embeddings_index, intersection in zip(
                queries_embeddings_index, documents_embeddings_index, intersections
            )
        ],
        dim=0,
    )


def scores(
    queries_activations: torch.Tensor,
    queries_embeddings: torch.Tensor,
    documents_activations: torch.Tensor,
    documents_embeddings: torch.Tensor,
    device: str,
    func=torch.mean,
) -> list:
    """Computes score between queries and documents intersected activated tokens.

    Parameters
    ----------
    queries_activations
        Queries activated tokens.
    queries_embeddings
        Queries activated tokens embeddings.
    documents_activations
        Documents activated tokens.
    documents_embeddings
        Documents activated tokens embeddings.
    func
        Either torch.sum or torch.mean. torch.mean is dedicated to training and
        torch.sum is dedicated to inference.

    Example
    ----------
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import model, utils
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = model.SparsEmbed(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased"),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ... )

    >>> queries_embeddings = model(
    ...     ["Paris", "Toulouse"],
    ...     k=96
    ... )

    >>> documents_embeddings = model(
    ...    ["Toulouse is a city located in France.", "Paris is a city located in France."],
    ...     k=256
    ... )

    >>> scores = utils.scores(
    ...     queries_activations=queries_embeddings["activations"],
    ...     queries_embeddings=queries_embeddings["embeddings"],
    ...     documents_activations=documents_embeddings["activations"],
    ...     documents_embeddings=documents_embeddings["embeddings"],
    ...     func=torch.sum, # torch.sum is dedicated to training
    ...     device="cpu",
    ... )

    """
    queries_embeddings_index = _build_index(
        activations=queries_activations, embeddings=queries_embeddings
    )

    documents_embeddings_index = _build_index(
        activations=documents_activations, embeddings=documents_embeddings
    )

    intersections = _get_intersection(
        queries_activations=queries_activations,
        documents_activations=documents_activations,
    )

    return _get_scores(
        queries_embeddings_index=queries_embeddings_index,
        documents_embeddings_index=documents_embeddings_index,
        intersections=intersections,
        func=func,
        device=device,
    )
