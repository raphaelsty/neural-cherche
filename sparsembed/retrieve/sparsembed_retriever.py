import os
import warnings

import torch
import tqdm

from ..model import SparsEmbed

__all__ = ["SparsEmbedRetriever"]


class SparsEmbedRetriever:
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
    ...     embedding_size=64,
    ... )

    >>> retriever = retrieve.SparsEmbedRetriever(key="id", on="document", model=model)

    >>> documents = [
    ...     {"id": 0, "document": "Food"},
    ...     {"id": 1, "document": "Sports"},
    ...     {"id": 2, "document": "Cinema"},
    ... ]
    >>> retriever = retriever.add(
    ...     documents=documents,
    ...     k_tokens=96,
    ...     batch_size=32
    ... )

    >>> print(retriever(["Food", "Sports", "Cinema"], k_tokens=96, batch_size=32))
    [[{'id': 0, 'similarity': 317.099365234375},
      {'id': 2, 'similarity': 83.53012084960938},
      {'id': 1, 'similarity': 74.39786529541016}],
     [{'id': 1, 'similarity': 430.7840576171875},
      {'id': 2, 'similarity': 88.93757629394531},
      {'id': 0, 'similarity': 74.39787292480469}],
     [{'id': 2, 'similarity': 290.1988220214844},
      {'id': 1, 'similarity': 88.93758392333984},
      {'id': 0, 'similarity': 83.53012084960938}]]

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
        warnings.filterwarnings(
            "ignore", ".*Sparse CSR tensor support is in beta state.*"
        )

    def add(
        self,
        documents: list,
        batch_size: int = 32,
        k_tokens: int = 96,
    ) -> "SparsEmbedRetriever":
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
            (
                documents_embeddings,
                documents_activations,
                sparse_matrix,
            ) = self._build_index(
                X=[" ".join([document[field] for field in self.on]) for document in X],
                k_tokens=k_tokens,
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
                    for index, document in enumerate(X)
                },
            }

        return self

    def __call__(
        self,
        q: list[str],
        k: int = 100,
        batch_size: int = 64,
        k_tokens: int = 96,
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
            (
                queries_embeddings,
                queries_activations,
                sparse_matrix,
            ) = self._build_index(
                X=X,
                k_tokens=k_tokens,
                **kwargs,
            )

            sparse_scores = (sparse_matrix @ self.sparse_matrix).to_dense()

            sparse_scores, sparse_matchs = torch.topk(
                input=sparse_scores, k=min(k, len(self.documents_keys)), dim=-1
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

            ranked += self._rank(
                dense_scores=dense_scores,
                sparse_matchs=sparse_matchs,
                k_dense=k,
            )

        return ranked

    def _rank(
        self, dense_scores: torch.Tensor, sparse_matchs: torch.Tensor, k_dense: int
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
            input=dense_scores, k=min(k_dense, len(self.documents_keys)), dim=-1
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
        k_tokens: int,
        **kwargs,
    ) -> tuple[list, list, torch.Tensor]:
        """Build a sparse matrix index."""
        index_embeddings, index_activations, sparse_activations = [], [], []
        batch_embeddings = self.model.encode(X, k_tokens=k_tokens, **kwargs)
        sparse_activations.append(batch_embeddings["sparse_activations"].to_sparse())

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

    def _get_scores(
        self,
        queries_embeddings: list[torch.Tensor],
        documents_embeddings: list[list[torch.Tensor]],
        intersections: list[torch.Tensor],
    ) -> torch.Tensor:
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
                        if len(intersection) > 0
                        else torch.tensor(0.0, device=self.model.device)
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
