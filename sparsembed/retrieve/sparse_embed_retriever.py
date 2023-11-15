import collections
import os

import numpy as np
import torch
from scipy.sparse import csc_matrix, csr_matrix, hstack, vstack

from ..models import SparseEmbed
from ..utils import batchify
from .tfidf_retriever import TfIdfRetriever

__all__ = ["SparseEmbedRetriever"]


class SparseEmbedRetriever(TfIdfRetriever):
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

    >>> device = "mps"

    >>> model = models.SparseEmbed(
    ...     model_name_or_path="distilbert-base-uncased",
    ...     device=device,
    ...     embedding_size=64,
    ... )

    >>> retriever = retrieve.SparseEmbedRetriever(
    ...     key="id",
    ...     on="document",
    ...     model=model,
    ... )

    >>> documents = [
    ...     {"id": 0, "document": "Food"},
    ...     {"id": 1, "document": "Sports"},
    ...     {"id": 2, "document": "Cinema"},
    ... ]

    >>> queries = ["Food", "Sports", "Cinema"]

    >>> documents_embeddings = retriever.encode_documents(
    ...     documents=documents,
    ...     batch_size=1,
    ... )

    >>> queries_embeddings = retriever.encode_queries(
    ...     queries=queries,
    ...     batch_size=1,
    ... )

    >>> retriever = retriever.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> scores = retriever(
    ...     queries_embeddings=queries_embeddings,
    ...     batch_size=32
    ... )

    >>> pprint(scores)
    [[{'id': 0, 'similarity': 127.2161636352539},
      {'id': 2, 'similarity': 112.07202911376953},
      {'id': 1, 'similarity': 107.36328887939453}],
     [{'id': 1, 'similarity': 167.22268676757812},
      {'id': 2, 'similarity': 87.65172576904297},
      {'id': 0, 'similarity': 84.95704650878906}],
     [{'id': 2, 'similarity': 141.13333129882812},
      {'id': 0, 'similarity': 111.82447814941406},
      {'id': 1, 'similarity': 107.57138061523438}]]

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        model: SparseEmbed,
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
        self.documents_embeddings, self.documents_activations = [], []

        os.environ["TOKENIZERS_PARALLELISM"] = tokenizer_parallelism

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        query_mode: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Encode queries.

        Parameters
        ----------
        queries
            List of queries to encode.
        batch_size
            Batch size.
        tqdm_bar
            Display a tqdm bar.
        """
        embeddings = collections.defaultdict(dict)

        for batch in batchify(
            queries,
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} encoder",
            tqdm_bar=tqdm_bar,
        ):
            documents_embeddings = self.model.encode(
                texts=batch,
                query_mode=query_mode,
                **kwargs,
            )

            for field, output in documents_embeddings.items():
                for query in batch:
                    activation = output.detach().cpu().numpy().astype(np.float32)
                    if field == "sparse_activations":
                        activation = csr_matrix(activation)
                    embeddings[query][field] = activation

        return embeddings

    def encode_documents(
        self,
        documents: list[dict],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        query_mode: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Encode documents.

        Parameters
        ----------
        documents
            List of documents to encode.
        """
        embeddings = collections.defaultdict(dict)

        for batch in batchify(
            documents,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
        ):
            documents_embeddings = self.model.encode(
                texts=[
                    " ".join([doc.get(field, "") for field in self.on]) for doc in batch
                ],
                query_mode=query_mode,
                **kwargs,
            )

            for field, output in documents_embeddings.items():
                output = output.detach().cpu().numpy().astype(np.float32)
                for document, activation in zip(batch, output):
                    if field == "sparse_activations":
                        activation = csc_matrix(activation).T
                    embeddings[document[self.key]][field] = activation

        return embeddings

    def add(
        self,
        documents_embeddings: dict[dict[str, torch.Tensor]],
    ) -> "SparseEmbedRetriever":
        """Add documents embeddings and activations to the retriever.

        Parameters
        ----------
        documents
            Documents to add.
        documents_embeddings
            Documents embeddings.
        build_index
            Build a sparse matrix index.
        matrix
            List of sparse matrix.
        """
        matrix = hstack(
            [
                embeddings["sparse_activations"]
                for embeddings in documents_embeddings.values()
            ]
        )
        self.matrix = matrix if self.matrix is None else hstack((self.matrix, matrix))

        for key, embeddings in documents_embeddings.items():
            self.documents_embeddings.append(
                {
                    token.item(): token_embedding
                    for token, token_embedding in zip(
                        embeddings["activations"], embeddings["embeddings"]
                    )
                }
            )
            self.documents_activations.append(embeddings["activations"])

        for key in documents_embeddings:
            self.documents.append({self.key: key})
            self.n_documents += 1

        return self

    def __call__(
        self,
        queries_embeddings,
        k: int = None,
        batch_size: int = 64,
        tqdm_bar: bool = True,
    ) -> list:
        """Retrieve documents.

        Parameters
        ----------
        q
            Queries.
        k
            Number of documents to retrieve.
        batch_size
            Batch size.
        tqdm_bar
            Display a tqdm bar.
        """
        k = k if k is not None else self.n_documents

        ranked = []

        for queries_batch in batchify(
            list(queries_embeddings.values()),
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} retriever",
            tqdm_bar=tqdm_bar,
        ):
            embeddings = {
                "activations": torch.cat(
                    [
                        torch.tensor(query["activations"], device=self.model.device)
                        for query in queries_batch
                    ],
                    dim=0,
                ),
                "embeddings": torch.cat(
                    [
                        torch.tensor(query["embeddings"], device=self.model.device)
                        for query in queries_batch
                    ],
                    dim=0,
                ),
                "sparse_activations": vstack(
                    [query["sparse_activations"] for query in queries_batch]
                ),
            }

            ranked.extend(
                self._retrieve(
                    embeddings=embeddings,
                    k=k,
                )
            )

        return ranked

    def _retrieve(self, embeddings: dict, k: int) -> list[list[dict]]:
        """Retrieve documents from input embeddings.

        Parameters
        ----------
        embeddings
            Input embeddings.
        k
            Number of documents to retrieve.
        """
        similarities = -1 * embeddings["sparse_activations"].dot(self.matrix)
        sparse_matchs, _ = self.top_k(similarities=similarities, k=k)

        documents_activations = [
            [
                torch.tensor(
                    self.documents_activations[document], device=self.model.device
                )
                for document in query_matchs
            ]
            for query_matchs in sparse_matchs
        ]

        intersections = self._get_intersection(
            queries_activations=embeddings["activations"],
            documents_activations=documents_activations,
        )

        queries_embeddings = [
            {
                token.item(): token_embedding
                for token, token_embedding in zip(query_activations, query_embeddings)
            }
            for query_activations, query_embeddings in zip(
                embeddings["activations"],
                embeddings["embeddings"],
            )
        ]

        dense_scores = self._get_scores(
            queries_embeddings=queries_embeddings,
            documents_embeddings=[
                [self.documents_embeddings[document] for document in match]
                for match in sparse_matchs
            ],
            intersections=intersections,
        )

        return self._rank(
            dense_scores=dense_scores,
            sparse_matchs=sparse_matchs,
            k=k,
        )

    def _rank(
        self, dense_scores: torch.Tensor, sparse_matchs: torch.Tensor, k: int
    ) -> list[dict]:
        """Rank documents by scores.

        Parameters
        ----------
        scores
            Scores between queries and documents.
        matchs
            Documents matchs.
        """
        ranked = []

        for query_scores, query_sparse_matchs in zip(dense_scores, sparse_matchs):
            query_scores, query_matchs = torch.topk(
                input=query_scores,
                k=min(k, query_scores.shape[0]),
                dim=-1,
            )

            query_matchs = np.take(
                a=query_sparse_matchs, indices=query_matchs.cpu().detach().numpy()
            )

            query_scores = query_scores.tolist()

            ranked.append(
                [
                    {
                        **self.documents[document],
                        "similarity": score,
                    }
                    for score, document in zip(query_scores, query_matchs)
                ]
            )

        return ranked

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
        queries_embeddings: list[dict[int, torch.Tensor]],
        documents_embeddings: list[list[dict[int, torch.Tensor]]],
        intersections: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Computes similarity scores between queries and documents with activated tokens embeddings.

        Parameters
        ----------
        queries_embeddings
            List of dictionaries, with tokens as keys and embeddings as values.
        documents_embeddings
            List of list of dictionnaries with tokens as keys and embeddings as values.

        """
        queries_documents_scores = []

        for intersection, query_embeddings, document_embeddings in zip(
            intersections, queries_embeddings, documents_embeddings
        ):
            query_documents_scores = []

            for intersection, document_embedding in zip(
                intersection, document_embeddings
            ):
                if len(intersection) > 0:
                    query_documents_scores.append(
                        torch.sum(
                            torch.stack(
                                [
                                    torch.tensor(
                                        document_embedding[token],
                                        device=self.model.device,
                                    )
                                    for token in intersection
                                ],
                                dim=0,
                            )
                            * torch.stack(
                                [query_embeddings[token] for token in intersection],
                                dim=0,
                            )
                        )
                    )

                else:
                    query_documents_scores.append(
                        torch.tensor(0.0, device=self.model.device)
                    )

            queries_documents_scores.append(torch.stack(query_documents_scores, dim=0))

        return queries_documents_scores
