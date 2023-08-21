import os
import warnings

import torch
import tqdm

from ..model import Splade

__all__ = ["SpladeRetriever"]


class SpladeRetriever:
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
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import model, retrieve
    >>> from pprint import pprint as print
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> device = "cpu"

    >>> model = model.Splade(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ...     device=device,
    ... )

    >>> retriever = retrieve.SpladeRetriever(key="id", on="document", model=model)

    >>> documents = [
    ...     {"id": 0, "document": "Food"},
    ...     {"id": 1, "document": "Sports"},
    ...     {"id": 2, "document": "Cinema"},
    ... ]
    >>> retriever = retriever.add(
    ...     documents=documents,
    ...     batch_size=1
    ... )

    >>> print(retriever(["Food", "Sports", "Cinema"], batch_size=32))
    [[{'id': 0, 'similarity': 2005.1702880859375},
      {'id': 1, 'similarity': 1866.706787109375},
      {'id': 2, 'similarity': 1690.898681640625}],
     [{'id': 1, 'similarity': 2534.69140625},
      {'id': 2, 'similarity': 1875.5230712890625},
      {'id': 0, 'similarity': 1866.70654296875}],
     [{'id': 2, 'similarity': 1934.9771728515625},
      {'id': 1, 'similarity': 1875.521484375},
      {'id': 0, 'similarity': 1690.8975830078125}]]

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        model: Splade,
        tokenizer_parallelism: str = "false",
    ) -> None:
        self.key = key
        self.on = [on] if isinstance(on, str) else on
        self.model = model
        self.vocabulary_size = len(model.tokenizer.get_vocab())

        # Mapping between sparse matrix index and document key.
        self.sparse_matrix = None
        self.documents_keys = {}

        # Documents embeddings and activations store.
        self.documents_embeddings, self.documents_activations = [], []
        os.environ["TOKENIZERS_PARALLELISM"] = tokenizer_parallelism
        warnings.filterwarnings(
            "ignore", ".*Sparse CSR tensor support is in beta state.*"
        )

    def add(
        self,
        documents: list,
        batch_size: int = 32,
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
        sparse_matrix = self._build_index(
            X=[
                " ".join([document[field] for field in self.on])
                for document in documents
            ],
            batch_size=batch_size,
            **kwargs,
        )

        self.sparse_matrix = (
            sparse_matrix.T
            if self.sparse_matrix is None
            else torch.cat([self.sparse_matrix.to_sparse(), sparse_matrix.T], dim=1)
        )

        self.documents_keys = {
            **self.documents_keys,
            **{
                len(self.documents_keys) + index: document[self.key]
                for index, document in enumerate(documents)
            },
        }

        return self

    def __call__(
        self,
        q: list[str],
        k: int = 100,
        batch_size: int = 3,
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
        sparse_matrix = self._build_index(
            X=[q] if isinstance(q, str) else q,
            batch_size=batch_size,
            **kwargs,
        )

        sparse_scores = (sparse_matrix @ self.sparse_matrix).to_dense()

        return self._rank(
            sparse_scores=sparse_scores,
            k=k,
        )

    def _rank(self, sparse_scores: torch.Tensor, k: int) -> list:
        """Rank documents by scores.

        Parameters
        ----------
        scores
            Scores between queries and documents.
        matchs
            Documents matchs.
        """
        sparse_scores, sparse_matchs = torch.topk(
            input=sparse_scores, k=min(k, len(self.documents_keys)), dim=-1
        )

        sparse_scores = sparse_scores.tolist()
        sparse_matchs = sparse_matchs.tolist()

        return [
            [
                {
                    self.key: self.documents_keys[document],
                    "similarity": score,
                }
                for score, document in zip(query_scores, query_matchs)
            ]
            for query_scores, query_matchs in zip(sparse_scores, sparse_matchs)
        ]

    def _build_index(
        self,
        X: list[str],
        batch_size: int,
        **kwargs,
    ) -> tuple[list, list, torch.Tensor]:
        """Build a sparse matrix index."""
        sparse_activations = []

        for batch in self._to_batch(X, batch_size=batch_size):
            batch_embeddings = self.model.encode(batch, **kwargs)

            sparse_activations.append(
                batch_embeddings["sparse_activations"].to_sparse()
            )

        return torch.cat(sparse_activations)

    @staticmethod
    def _to_batch(X: list, batch_size: int) -> list:
        """Convert input list to batch."""
        for X in tqdm.tqdm(
            [X[pos : pos + batch_size] for pos in range(0, len(X), batch_size)],
            position=0,
            total=1 + len(X) // batch_size,
        ):
            yield X
