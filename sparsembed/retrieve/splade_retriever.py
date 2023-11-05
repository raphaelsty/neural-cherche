import os

from scipy.sparse import csr_matrix

from ..models import Splade
from .tfidf_retriever import TfIdfRetriever

__all__ = ["SpladeRetriever"]


class SpladeRetriever(TfIdfRetriever):
    """Retriever class.

    Parameters
    ----------
    key
        Document unique identifier.
    on
        Document texts.
    model
        SparsEmbed model.

    Example
    -------
    >>> from sparsembed import models, retrieve
    >>> from pprint import pprint
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> documents = [
    ...     {"id": 0, "document": "Food"},
    ...     {"id": 1, "document": "Sports"},
    ...     {"id": 2, "document": "Cinema"},
    ... ]

    >>> model = models.Splade(
    ...     model_name_or_path="distilbert-base-uncased",
    ...     device="mps",
    ... )

    >>> retriever = retrieve.SpladeRetriever(key="id", on="document", model=model)

    >>> retriever = retriever.add(
    ...     documents=documents,
    ...     batch_size=32,
    ... )

    >>> matchs = retriever(
    ...     ["Food", "Sports", "Cinema"],
    ...     batch_size=32,
    ...     k=3,
    ... )

    >>> pprint(matchs)
    [[{'id': 0, 'similarity': 380.16464},
      {'id': 1, 'similarity': 318.81836},
      {'id': 2, 'similarity': 318.04926}],
     [{'id': 1, 'similarity': 356.52582},
      {'id': 2, 'similarity': 312.32935},
      {'id': 0, 'similarity': 262.64456}],
     [{'id': 2, 'similarity': 362.6682},
      {'id': 1, 'similarity': 334.7304},
      {'id': 0, 'similarity': 295.53903}]]

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        model: Splade,
        tokenizer_parallelism: str = "false",
    ) -> None:
        super().__init__(
            key=key,
            on=on,
        )

        self.model = model

        # TfIdf Retriever.
        self.fit = True

        # Documents embeddings and activations store.
        self.documents_embeddings = []
        os.environ["TOKENIZERS_PARALLELISM"] = tokenizer_parallelism

    def transform_queries(self, texts: list[str], **kwargs) -> csr_matrix:
        """Transform queries into sparse matrix."""
        return csr_matrix(
            self.model.encode(
                texts=texts,
                query_mode=True,
                **kwargs,
            )["sparse_activations"]
            .detach()
            .cpu()
        )

    def transform_documents(self, texts: list[str], **kwargs) -> csr_matrix:
        """Transform queries into sparse matrix."""
        return csr_matrix(
            self.model.encode(
                texts=texts,
                query_mode=False,
                **kwargs,
            )["sparse_activations"]
            .detach()
            .cpu()
        )

    def add(
        self,
        documents: list,
        batch_size: int = 32,
        tqdm_bar: bool = False,
        **kwargs,
    ) -> "SpladeRetriever":
        """Add new documents to the retriever.

        Computes documents embeddings and activations and update the sparse matrix.

        Parameters
        ----------
        documents
            Documents to add.
        batch_size
            Batch size.
        """
        super().add(
            documents=documents,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            transform=self.transform_documents,
            **kwargs,
        )

        return self

    def __call__(
        self,
        q: list[str],
        k: int = 100,
        batch_size: int = 32,
        tqdm_bar: bool = False,
        **kwargs,
    ) -> list:
        """Retrieve documents.

        Parameters
        ----------
        q
            Queries.
        k_sparse
            Number of documents to retrieve.
        """
        return super().__call__(
            q=q,
            k=k,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            transform=self.transform_queries,
            **kwargs,
        )
