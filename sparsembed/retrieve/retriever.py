import os

import numpy as np
import torch
import tqdm
from scipy import sparse

from ..model import SparsEmbed

__all__ = ["Retriever"]


class Retriever:
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

    >>> model = model.SparsEmbed(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ...     device=device,
    ...     embedding_size=3,
    ... )

    >>> retriever = retrieve.Retriever(key="id", on="document", model=model)

    >>> documents = [
    ...     {"id": 0, "document": "Food is good."},
    ...     {"id": 1, "document": "Sports is great."},
    ... ]
    >>> retriever = retriever.add(
    ...     documents=documents,
    ...     k_token=256,
    ...     batch_size=24
    ... )

    >>> documents = [
    ...     {"id": 2, "document": "Cinema is great."},
    ...     {"id": 3, "document": "Music is amazing."},
    ... ]
    >>> retriever = retriever.add(
    ...     documents=documents,
    ...     k_token=256,
    ...     batch_size=24
    ... )

    >>> print(retriever(["Food", "Sports", "Cinema", "Music", "Hello World"], k_token=96))
    [[{'id': 0, 'similarity': 1.4686675071716309},
      {'id': 1, 'similarity': 1.345913052558899},
      {'id': 3, 'similarity': 1.304019808769226},
      {'id': 2, 'similarity': 1.1579231023788452}],
     [{'id': 1, 'similarity': 7.0373148918151855},
      {'id': 3, 'similarity': 3.528376817703247},
      {'id': 2, 'similarity': 2.4535036087036133},
      {'id': 0, 'similarity': 1.7893059253692627}],
     [{'id': 2, 'similarity': 2.3167333602905273},
      {'id': 3, 'similarity': 2.2312183380126953},
      {'id': 1, 'similarity': 2.0195937156677246},
      {'id': 0, 'similarity': 1.2890148162841797}],
     [{'id': 3, 'similarity': 2.4722704887390137},
      {'id': 2, 'similarity': 1.8648046255111694},
      {'id': 1, 'similarity': 1.732576608657837},
      {'id': 0, 'similarity': 1.3416467905044556}],
     [{'id': 3, 'similarity': 3.7778899669647217},
      {'id': 2, 'similarity': 3.198120355606079},
      {'id': 1, 'similarity': 3.1253902912139893},
      {'id': 0, 'similarity': 2.458303451538086}]]

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        model: SparsEmbed,
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

    def add(
        self,
        documents: list,
        k_token: int = 256,
        batch_size: int = 32,
    ) -> "Retriever":
        """Add new documents to the retriever.

        Computes documents embeddings and activations and update the sparse matrix.

        Parameters
        ----------
        documents
            Documents to add.
        k_token
            Number of tokens to activate.
        batch_size
            Batch size.
        """
        (
            documents_embeddings,
            documents_activations,
            sparse_matrix,
        ) = self._build_index(
            X=[
                " ".join([document[field] for field in self.on])
                for document in documents
            ],
            k_token=k_token,
            batch_size=batch_size,
        )

        self.documents_embeddings.extend(documents_embeddings)
        self.documents_activations.extend(documents_activations)

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
        k_sparse: int = 100,
        k_token: int = 96,
        batch_size: int = 3,
    ) -> list:
        """Retrieve documents.

        Parameters
        ----------
        q
            Queries.
        k_sparse
            Number of documents to retrieve.
        k_token
            Number of tokens to activate.
        """
        (
            queries_embeddings,
            queries_activations,
            sparse_matrix,
        ) = self._build_index(
            X=[q] if isinstance(q, str) else q,
            k_token=k_token,
            batch_size=batch_size,
        )

        sparse_scores = (sparse_matrix @ self.sparse_matrix).to_dense()

        _, sparse_matchs = torch.topk(
            input=sparse_scores, k=min(k_sparse, len(self.documents_keys)), dim=-1
        )

        sparse_matchs_idx = sparse_matchs.tolist()

        # Intersections between queries and documents activated tokens.
        intersections = self._get_intersection(
            queries_activations=queries_activations,
            documents_activations=[
                [self.documents_activations[document] for document in query_matchs]
                for query_matchs in sparse_matchs_idx
            ],
        )

        dense_scores = self._get_scores(
            queries_embeddings=queries_embeddings,
            documents_embeddings=[
                [self.documents_embeddings[document] for document in match]
                for match in sparse_matchs_idx
            ],
            intersections=intersections,
        )

        return self._rank(
            dense_scores=dense_scores, sparse_matchs=sparse_matchs, k_sparse=k_sparse
        )

    def _rank(
        self, dense_scores: torch.Tensor, sparse_matchs: torch.Tensor, k_sparse: int
    ) -> list:
        """Rank documents by scores.

        Parameters
        ----------
        scores
            Scores between queries and documents.
        matchs
            Documents matchs.
        """
        dense_scores, dense_matchs = torch.topk(
            input=dense_scores, k=min(k_sparse, len(self.documents_keys)), dim=-1
        )

        dense_scores = dense_scores.tolist()
        dense_matchs = (
            torch.gather(sparse_matchs, 1, dense_matchs).detach().cpu().tolist()
        )

        return [
            [
                {
                    self.key: self.documents_keys[document],
                    "similarity": score,
                }
                for score, document in zip(query_scores, query_matchs)
            ]
            for query_scores, query_matchs in zip(dense_scores, dense_matchs)
        ]

    def _build_index(
        self,
        X: list[str],
        batch_size: int,
        k_token: int,
    ) -> tuple[list, list, sparse.csc_matrix]:
        """Build a sparse matrix index."""
        index_embeddings, index_activations, sparse_activations = [], [], []

        for batch in self._to_batch(X, batch_size=batch_size):
            batch_embeddings = self.model.encode(batch, k=k_token)

            sparse_activations.append(
                batch_embeddings["sparse_activations"].to_sparse()
            )

            for activations, activations_idx, embeddings in zip(
                batch_embeddings["activations"],
                batch_embeddings["activations"].detach().cpu().tolist(),
                batch_embeddings["embeddings"],
            ):
                index_activations.append(activations)
                index_embeddings.append(
                    {
                        token: embedding
                        for token, embedding in zip(activations_idx, embeddings)
                    }
                )

        return (
            index_embeddings,
            index_activations,
            torch.cat(sparse_activations),
        )

    @staticmethod
    def _to_batch(X: list, batch_size: int) -> list:
        """Convert input list to batch."""
        for X in tqdm.tqdm(
            [X[pos : pos + batch_size] for pos in range(0, len(X), batch_size)],
            position=0,
            total=1 + len(X) // batch_size,
        ):
            yield X

    @classmethod
    def _get_intersection(
        cls,
        queries_activations: list[torch.Tensor],
        documents_activations: list[list[torch.Tensor]],
    ) -> list[list[torch.Tensor]]:
        """Retrieve intersection of activated tokens between queries and documents."""
        return [
            [
                cls._intersection(query_activations, document_activations)
                for document_activations in query_documents_activations
            ]
            for query_activations, query_documents_activations in zip(
                queries_activations, documents_activations
            )
        ]

    @staticmethod
    def _intersection(t1: torch.Tensor, t2: torch.Tensor) -> list[int]:
        t1, t2 = t1.flatten(), t2.flatten()
        combined = torch.cat((t1, t2), dim=0)
        uniques, counts = combined.unique(return_counts=True, sorted=False)
        return uniques[counts > 1].tolist()

    @staticmethod
    def _get_scores(
        queries_embeddings: list[torch.Tensor],
        documents_embeddings: list[list[torch.Tensor]],
        intersections: list[torch.Tensor],
    ) -> list:
        """Computes similarity scores between queries and documents with activated tokens embeddings."""
        return torch.stack(
            [
                torch.stack(
                    [
                        torch.sum(
                            torch.stack(
                                [document_embddings[token] for token in intersection],
                                dim=0,
                            )
                            * torch.stack(
                                [query_embeddings[token] for token in intersection],
                                dim=0,
                            )
                        )
                        for intersection, document_embddings in zip(
                            query_intersections, query_documents_embddings
                        )
                    ],
                    dim=0,
                )
                for query_intersections, query_embeddings, query_documents_embddings in zip(
                    intersections, queries_embeddings, documents_embeddings
                )
            ]
        )
