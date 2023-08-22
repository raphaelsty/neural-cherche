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
    ...     k_tokens=256,
    ...     batch_size=32,
    ... )

    >>> print(retriever(["Food", "Sports", "Cinema"],  k_tokens=256, batch_size=32))
    [[{'id': 0, 'similarity': 422.4058532714844},
      {'id': 2, 'similarity': 239.1786651611328},
      {'id': 1, 'similarity': 237.7996826171875}],
     [{'id': 1, 'similarity': 473.9842224121094},
      {'id': 0, 'similarity': 237.7996826171875},
      {'id': 2, 'similarity': 231.45362854003906}],
     [{'id': 2, 'similarity': 382.86102294921875},
      {'id': 0, 'similarity': 239.1786651611328},
      {'id': 1, 'similarity': 231.45362854003906}]]

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
        k_tokens: int = 256,
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

        for X in self._to_batch(documents, batch_size=batch_size):
            sparse_matrix = self._build_index(
                X=[" ".join([document[field] for field in self.on]) for document in X],
                k_tokens=k_tokens,
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
                    for index, document in enumerate(X)
                },
            }

        return self

    def __call__(
        self,
        q: list[str],
        k: int = 100,
        batch_size: int = 3,
        k_tokens: int = 256,
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
        q = [q] if isinstance(q, str) else q

        ranked = []

        for X in self._to_batch(q, batch_size=batch_size):
            sparse_matrix = self._build_index(
                X=X,
                k_tokens=k_tokens,
                **kwargs,
            )

            sparse_scores = (sparse_matrix @ self.sparse_matrix).to_dense()

            ranked += self._rank(
                sparse_scores=sparse_scores,
                k=k,
            )

        return ranked

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
        k_tokens: int,
        **kwargs,
    ) -> tuple[list, list, torch.Tensor]:
        """Build a sparse matrix index."""
        batch_embeddings = self.model.encode(X, k_tokens=k_tokens, **kwargs)
        return batch_embeddings["sparse_activations"].to_sparse()

    @staticmethod
    def _to_batch(X: list, batch_size: int) -> list:
        """Convert input list to batch."""
        for X in tqdm.tqdm(
            [X[pos : pos + batch_size] for pos in range(0, len(X), batch_size)],
            position=0,
            total=1 + len(X) // batch_size,
        ):
            yield X
