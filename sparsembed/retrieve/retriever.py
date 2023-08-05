import os

import numpy as np
import torch
import tqdm
from scipy import sparse

from ..model import SparsEmbed

__all__ = ["Retriever"]


class Retriever:
    """Class dedicated to SparsEmbed model inference in order retrieve
    documents with queries.

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

    >>> device = "mps"

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

    >>> print(retriever(["Food", "Sports", "Cinema", "Music"], k_token=96))
    [[{'id': 0, 'similarity': 1.4686672687530518},
      {'id': 1, 'similarity': 1.3459084033966064},
      {'id': 3, 'similarity': 1.3040170669555664},
      {'id': 2, 'similarity': 1.157921314239502}],
     [{'id': 1, 'similarity': 7.03730583190918},
      {'id': 3, 'similarity': 3.5283799171447754},
      {'id': 2, 'similarity': 2.453505516052246},
      {'id': 0, 'similarity': 1.789308786392212}],
     [{'id': 2, 'similarity': 2.316730260848999},
      {'id': 3, 'similarity': 2.2312138080596924},
      {'id': 1, 'similarity': 2.0195863246917725},
      {'id': 0, 'similarity': 1.289013147354126}],
     [{'id': 3, 'similarity': 5.773364067077637},
      {'id': 1, 'similarity': 3.6177942752838135},
      {'id': 2, 'similarity': 3.3001816272735596},
      {'id': 0, 'similarity': 2.591763496398926}]]

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
            sparse_matrix
            if self.sparse_matrix is None
            else sparse.vstack((self.sparse_matrix, sparse_matrix))
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
        q: list[str] | str,
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

        # TODO: return torch tensor
        matchs, _ = self.top_k_by_partition(
            similarities=(sparse_matrix @ self.sparse_matrix.T).toarray(),
            k_sparse=k_sparse,
        )

        # Intersections between queries and documents activated tokens.
        intersections = self._get_intersection(
            queries_activations=queries_activations,
            documents_activations=[
                [self.documents_activations[document] for document in query_matchs]
                for query_matchs in matchs
            ],
        )

        # Optimize to handle batchs
        scores = self._get_scores(
            queries_embeddings=queries_embeddings,
            documents_embeddings=[
                [self.documents_embeddings[document] for document in match]
                for match in matchs
            ],
            intersections=intersections,
        )

        return self._rank(scores=scores, matchs=matchs)

    def _rank(self, scores: torch.Tensor, matchs: torch.Tensor) -> list:
        """Rank documents by scores.

        Parameters
        ----------
        scores
            Scores between queries and documents.
        matchs
            Documents matchs.
        """
        ranks = torch.argsort(scores, dim=-1, descending=True, stable=False)

        scores = [
            torch.index_select(input=query_documents_score, dim=0, index=query_ranks)
            for query_documents_score, query_ranks in zip(scores, ranks)
        ]

        matchs = [
            torch.index_select(
                input=torch.tensor(query_matchs).to(self.model.device),
                dim=0,
                index=query_ranks,
            )
            for query_matchs, query_ranks in zip(matchs, ranks)
        ]

        return [
            [
                {
                    self.key: self.documents_keys[document.item()],
                    "similarity": score.item(),
                }
                for score, document in zip(query_scores, query_matchs)
            ]
            for query_scores, query_matchs in zip(scores, matchs)
        ]

    def _build_index(
        self,
        X: list[str],
        batch_size: int,
        k_token: int,
    ) -> tuple[list, list, sparse.csr_matrix]:
        """Build"""
        index_embeddings, index_activations, rows, columns, values = [], [], [], [], []
        n = 0

        for batch in self._to_batch(X, batch_size=batch_size):
            batch_embeddings = self.model.encode(batch, k=k_token)

            for activations, embeddings, sparse_activations in zip(
                batch_embeddings["activations"],
                batch_embeddings["embeddings"],
                batch_embeddings["sparse_activations"],
            ):
                index_activations.append(activations)
                index_embeddings.append(
                    {
                        token.item(): embedding
                        for token, embedding in zip(activations, embeddings)
                    }
                )

                tokens_scores = torch.index_select(
                    sparse_activations, dim=-1, index=activations
                )

                rows.extend([n for _ in range(len(activations))])
                columns.extend(activations.tolist())
                values.extend(tokens_scores.tolist())
                n += 1

        sparse_matrix = sparse.csc_matrix(
            (values, (rows, columns)), shape=(len(X), self.vocabulary_size)
        )

        return index_embeddings, index_activations, sparse_matrix

    def top_k_by_partition(
        self, similarities: np.ndarray, k_sparse: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top k elements by partition."""
        similarities *= -1

        if k_sparse < len(self.documents_keys):
            ind = np.argpartition(similarities, k_sparse, axis=-1)

            # k non-sorted indices
            ind = np.take(ind, np.arange(k_sparse), axis=-1)

            # k non-sorted values
            similarities = np.take_along_axis(similarities, ind, axis=-1)

            # sort within k elements
            ind_part = np.argsort(similarities, axis=-1)
            ind = np.take_along_axis(ind, ind_part, axis=-1)

        else:
            ind_part = np.argsort(similarities, axis=-1)
            ind = ind_part

        similarities *= -1
        val = np.take_along_axis(similarities, ind_part, axis=-1)
        return ind, val

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
