from typing import Any

import torch

from ..models import BLP
from .faiss_index import FaissIndex

__all__ = ["BLPRetriever"]


class BLPRetriever:
    """BLP Retriever

    Parameters
    ----------
    key
        Key identifier of the documents.
    on
        Columns to use for the documents.
    model
        BLP model.

    Example
    -------
    >>> from sparsembed import models, retrieve
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.BLP(
    ...     model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    ...     device="mps",
    ...     relations=["author", "genre"]
    ... )

    >>> retriever = retrieve.BLPRetriever(
    ...     key="id",
    ...     on=["title", "description"],
    ...     model=model,
    ... )

    >>> documents = [
    ...     {"id": 0, "title": "Michael Jackson", "description": "pop"},
    ...     {"id": 1, "title": "Victor Hugo", "description": "classic"},
    ... ]

    >>> embeddings_documents = retriever.encode_documents(
    ...     documents=documents,
    ...     batch_size=2,
    ... )

    >>> retriever = retriever.add(
    ...     documents=documents,
    ...     embeddings_documents=embeddings_documents,
    ... )

    >>> scores = retriever(
    ...     q=["pop", "book"],
    ...     relations=["genre", "author"],
    ...     search_for_tail=True,
    ...     k=2,
    ...     batch_size=2,
    ... )

    >>> assert len(scores) == 2
    >>> assert len(scores[0]) == 2

    >>> scores = retriever(
    ...     q="pop",
    ...     relations="genre",
    ...     search_for_tail=True,
    ...     k=2,
    ...     batch_size=2,
    ... )

    >>> assert len(scores) == 2
    >>> assert isinstance(scores[0], dict)

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        model: BLP,
        index: FaissIndex = None,
        normalize: bool = True,
    ) -> None:
        self.key = key
        self.on = [on] if isinstance(on, str) else on
        self.model = model
        self.index = FaissIndex(
            key=key,
            index=index,
            normalize=normalize,
        )

    def encode_documents(
        self, documents: list[dict[str, str]], batch_size: int = 32, **kwargs
    ) -> torch.Tensor:
        """Encode documents.

        Parameters
        ----------
        documents
            Documents to encode.
        batch_size
            Batch size.
        """
        return (
            self.model.encode_entities(
                entities=[
                    " ".join([document.get(field, "") for field in self.on])
                    for document in documents
                ],
                batch_size=batch_size,
                **kwargs,
            )
            .cpu()
            .detach()
            .numpy()
        )

    def encode_queries(
        self, q: list[str], batch_size: int = 32, **kwargs
    ) -> torch.Tensor:
        """Encode queries.

        Parameters
        ----------
        q
            Queries to encode.
        batch_size
            Batch size.
        """
        return (
            self.model.encode_entities(entities=q, batch_size=batch_size, **kwargs)
            .cpu()
            .detach()
            .numpy()
        )

    def add(
        self, embeddings_documents: list[torch.Tensor], documents: list[dict[str, str]]
    ) -> "BLPRetriever":
        """Add documents embeddings to the index."""
        self.index.add(documents=documents, embeddings=embeddings_documents)
        return self

    def __call__(
        self,
        q: list[str],
        relations: list[str],
        search_for_tail: bool = True,
        k: int = 100,
        batch_size: int = 32,
        **kwargs,
    ) -> list[list[dict]]:
        """Retrieve entities.

        Parameters
        ----------
        q
            Queries.
        relations
            List of relations.
        search_for_tail
            Whether to search for tail entities or head entities.
        k
            Number of entities to retrieve.
        batch_size
            Batch size.
        """
        entities_embeddings = self.encode_queries(q=q, batch_size=batch_size, **kwargs)

        relations_embeddings = (
            self.model.encode_relations(relations=relations).cpu().detach().numpy()
        )

        scores = self.index(
            embeddings=entities_embeddings + relations_embeddings
            if search_for_tail
            else relations_embeddings - entities_embeddings,
            k=k,
        )

        return scores[0] if isinstance(q, str) else scores
