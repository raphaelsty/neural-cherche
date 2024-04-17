import collections
import os

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack, vstack

from .. import models, utils
from .tfidf import TfIdf

__all__ = ["SparseEmbed"]


class SparseEmbed(TfIdf):
    """SparseEmbed retriever.

    Parameters
    ----------
    key
        Document unique identifier.
    on
        Document texts.
    model
        SparseEmbed model.

    Examples
    --------
    >>> from neural_cherche import models, retrieve
    >>> from pprint import pprint
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.SparseEmbed(
    ...     model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
    ...     device="cpu",
    ...     embedding_size=64,
    ... )

    >>> retriever = retrieve.SparseEmbed(
    ...     key="id",
    ...     on="document",
    ...     model=model,
    ... )

    >>> documents = [
    ...     {"id": 0, "document": " ".join(["Food Hello world"] * 200)},
    ...     {"id": 1, "document": "Sports"},
    ...     {"id": 2, "document": "Cinema"},
    ... ]

    >>> queries = ["Food", "Sports", "Cinema"]

    >>> documents_embeddings = retriever.encode_documents(
    ...     documents=documents,
    ...     batch_size=2,
    ... )

    >>> queries_embeddings = retriever.encode_queries(
    ...     queries=queries,
    ...     batch_size=2,
    ... )

    >>> retriever = retriever.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> scores = retriever(
    ...     queries_embeddings=queries_embeddings,
    ...     batch_size=32
    ... )

    >>> pprint(scores)
    [[{'id': 0, 'similarity': 650.8814697265625},
      {'id': 2, 'similarity': 63.36189270019531},
      {'id': 1, 'similarity': 55.061676025390625}],
     [{'id': 1, 'similarity': 139.74667358398438},
      {'id': 2, 'similarity': 63.8810920715332},
      {'id': 0, 'similarity': 42.96449661254883}],
     [{'id': 2, 'similarity': 158.86715698242188},
      {'id': 0, 'similarity': 68.95826721191406},
      {'id': 1, 'similarity': 63.872554779052734}]]

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        model: models.SparseEmbed,
        tokenizer_parallelism: str = "false",
        device: str = None,
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
        self.device = device if device is not None else self.model.device

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        query_mode: bool = True,
        warn_duplicates: bool = True,
        desc: str = "embeddings queries",
        **kwargs,
    ) -> dict[str, dict[str, torch.Tensor]]:
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

        for batch in utils.batchify(
            queries,
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} {desc}",
            tqdm_bar=tqdm_bar,
        ):
            queries_embeddings = self.model.encode(
                texts=batch,
                query_mode=query_mode,
                **kwargs,
            )

            queries_embeddings = {
                field: array.detach().cpu().numpy().astype(np.float32)
                for field, array in queries_embeddings.items()
            }

            queries_embeddings["sparse_activations"] = csr_matrix(
                queries_embeddings["sparse_activations"]
            )

            for query, sparse_activations, activations, tokens_embeddings in zip(
                batch,
                queries_embeddings["sparse_activations"],
                queries_embeddings["activations"],
                queries_embeddings["embeddings"],
            ):
                embeddings[query]["sparse_activations"] = csr_matrix(sparse_activations)
                embeddings[query]["activations"] = activations
                embeddings[query]["embeddings"] = tokens_embeddings

        if len(embeddings) != len(queries) and warn_duplicates:
            utils.duplicates_queries_warning()

        return embeddings

    def encode_documents(
        self,
        documents: list[dict],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        query_mode: bool = False,
        desc: str = "embeddings documents",
        **kwargs,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Encode documents.

        Parameters
        ----------
        documents
            List of documents to encode.
        """
        embeddings = collections.defaultdict(dict)

        for batch in utils.batchify(
            X=documents,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            desc=f"{self.__class__.__name__} {desc}",
        ):
            documents_embeddings = self.model.encode(
                texts=[
                    " ".join([doc.get(field, "") for field in self.on]) for doc in batch
                ],
                query_mode=query_mode,
                **kwargs,
            )

            documents_embeddings = {
                field: array.detach().cpu().numpy().astype(np.float32)
                for field, array in documents_embeddings.items()
            }

            documents_embeddings["sparse_activations"] = csr_matrix(
                documents_embeddings["sparse_activations"]
            )

            for document, sparse_activations, activations, tokens_embeddings in zip(
                batch,
                documents_embeddings["sparse_activations"],
                documents_embeddings["activations"],
                documents_embeddings["embeddings"],
            ):
                key = document[self.key]
                embeddings[key]["sparse_activations"] = sparse_activations
                embeddings[key]["activations"] = activations
                embeddings[key]["embeddings"] = tokens_embeddings

        return embeddings

    def add(
        self,
        documents_embeddings: dict[dict[str, torch.Tensor]],
    ) -> "SparseEmbed":
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
        matrix = vstack(
            [
                embeddings["sparse_activations"]
                for embeddings in documents_embeddings.values()
            ]
        ).T.tocsr()

        self.matrix = matrix if self.matrix is None else hstack((self.matrix, matrix))

        for key, embeddings in documents_embeddings.items():
            self.documents_embeddings.append(
                {
                    token: token_embedding
                    for token, token_embedding in zip(
                        embeddings["activations"],
                        embeddings["embeddings"],
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
        queries_embeddings: dict[str, dict[str, torch.Tensor]],
        k: int = None,
        k_rank: int = None,
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
        k_rank = k_rank if k_rank is not None else k

        ranked = []

        for queries_batch in utils.batchify(
            X=list(queries_embeddings.values()),
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} retriever",
            tqdm_bar=tqdm_bar,
        ):
            embeddings = {
                "activations": torch.stack(
                    tensors=[
                        torch.tensor(data=query["activations"], device=self.device)
                        for query in queries_batch
                    ],
                    dim=0,
                ),
                "embeddings": torch.stack(
                    tensors=[
                        torch.tensor(data=query["embeddings"], device=self.device)
                        for query in queries_batch
                    ],
                    dim=0,
                ),
                "sparse_activations": vstack(
                    blocks=[query["sparse_activations"] for query in queries_batch]
                ),
            }

            ranked.extend(
                self._retrieve(
                    embeddings=embeddings,
                    k=k,
                    k_rank=k_rank,
                )
            )

        return ranked

    def _retrieve(
        self, embeddings: dict[str, torch.Tensor], k: int, k_rank: int
    ) -> list[list[dict]]:
        """Retrieve documents from input embeddings.

        Parameters
        ----------
        embeddings
            Input embeddings.
        k
            Number of documents to retrieve.
        """
        sparse_matchs = [
            activations.indices
            for activations in (embeddings["sparse_activations"].dot(self.matrix))
        ]

        documents_activations = [
            [
                torch.tensor(
                    data=self.documents_activations[document], device=self.device
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
                token: token_embedding
                for token, token_embedding in zip(query_activations, query_embeddings)
            }
            for query_activations, query_embeddings in zip(
                embeddings["activations"].tolist(),
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
            k=k_rank,
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

            if len(query_scores) == 0:
                ranked.append([])
                continue

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
                cls._intersection(t1=query_activations, t2=document_activations)
                for document_activations in query_documents_activations
            ]
            for query_activations, query_documents_activations in zip(
                queries_activations, documents_activations
            )
        ]

    @staticmethod
    def _intersection(t1: torch.Tensor, t2: torch.Tensor) -> list[int]:
        t1, t2 = t1.flatten(), t2.flatten()
        combined = torch.cat(tensors=(t1, t2), dim=0)
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
                            input=torch.stack(
                                tensors=[
                                    torch.tensor(
                                        data=document_embedding[token],
                                        device=self.device,
                                    )
                                    for token in intersection
                                ],
                                dim=0,
                            )
                            * torch.stack(
                                tensors=[
                                    query_embeddings[token] for token in intersection
                                ],
                                dim=0,
                            )
                        )
                    )

                else:
                    query_documents_scores.append(
                        torch.tensor(data=0.0, device=self.device)
                    )

            if not query_documents_scores:
                queries_documents_scores.append([])
                continue

            queries_documents_scores.append(
                torch.stack(tensors=query_documents_scores, dim=0)
            )

        return queries_documents_scores
