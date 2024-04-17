import numpy as np
import torch

from .. import models, utils
from ..retrieve.sparse_embed import SparseEmbed as SparseEmbedRetriever

__all__ = ["SparseEmbed"]


class SparseEmbed(SparseEmbedRetriever):
    """SparseEmbed ranker.

    Parameters
    ----------
    key
        Document unique identifier.
    on
        Document texts.
    model
        SparseEmbed model.
    device
        Device to use, default is model device.

    Examples
    --------
    >>> from neural_cherche import models, rank
    >>> from pprint import pprint
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.SparseEmbed(
    ...     model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
    ...     device="cpu",
    ...     embedding_size=64,
    ... )

    >>> ranker = rank.SparseEmbed(
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

    >>> documents_embeddings = ranker.encode_documents(
    ...     documents=documents,
    ...     batch_size=2,
    ... )

    >>> queries_embeddings = ranker.encode_queries(
    ...     queries=queries,
    ...     batch_size=2,
    ... )

    >>> scores = ranker(
    ...     documents=[documents for _ in queries],
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     k=3,
    ...     batch_size=32
    ... )

    >>> pprint(scores)
    [[{'id': 0, 'similarity': 650.8815},
      {'id': 2, 'similarity': 63.361893},
      {'id': 1, 'similarity': 55.061672}],
     [{'id': 1, 'similarity': 139.74667},
      {'id': 2, 'similarity': 63.88109},
      {'id': 0, 'similarity': 42.964497}],
     [{'id': 2, 'similarity': 158.86716},
      {'id': 0, 'similarity': 68.95827},
      {'id': 1, 'similarity': 63.872555}]]

    >>> scores = ranker(
    ...     documents=[documents, [], documents],
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     k=3,
    ...     batch_size=32
    ... )

    >>> pprint(scores)
    [[{'id': 0, 'similarity': 650.8815},
      {'id': 2, 'similarity': 63.361893},
      {'id': 1, 'similarity': 55.061672}],
     [],
     [{'id': 2, 'similarity': 158.86716},
      {'id': 0, 'similarity': 68.95827},
      {'id': 1, 'similarity': 63.872555}]]

    >>> documents_embeddings = ranker.encode_documents(
    ...     documents=[documents for _ in queries],
    ...     batch_size=3,
    ... )

    >>> assert len(documents_embeddings) == len(documents)

    >>> documents_embeddings = ranker.encode_candidates_documents(
    ...     documents=documents,
    ...     candidates=[[{"id": 0}, {"id": 1}], [{"id": 1}], []],
    ...     batch_size=3,
    ... )

    >>> assert len(documents_embeddings) == 2

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
            model=model,
            tokenizer_parallelism=tokenizer_parallelism,
            device=device,
        )

    def encode_documents(
        self,
        documents: list[dict] | list[list[dict]],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        query_mode: bool = False,
        desc: str = "documents embeddings",
        **kwargs,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Encode documents.

        Parameters
        ----------
        documents
            Documents to encode.
        batch_size
            Batch size.
        tqdm_bar
            Display a tqdm bar.
        query_mode
            Query mode. True if documents are queries.
        """
        if not documents:
            return {}

        # Flatten documents if necessary
        if isinstance(documents[0], list):
            documents_flatten, duplicates = [], {}
            for query_documents in documents:
                for document in query_documents:
                    if document[self.key] not in duplicates:
                        duplicates[document[self.key]] = True
                        documents_flatten.append(document)
            documents = documents_flatten

        return super().encode_documents(
            documents=documents,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            query_mode=query_mode,
            desc=desc,
            **kwargs,
        )

    def encode_candidates_documents(
        self,
        documents: list[dict],
        candidates: list[list[dict]],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        query_mode: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Map documents contents to candidates and encode them.
        This method is useful when you have a list of candidates for each query without
        their contents and you want to encode them.

        Parameters
        ----------
        documents
            Documents.
        candidates
            List of candidates for each query.
        batch_size
            Batch size.
        tqdm_bar
            Show tqdm bar.
        query_mode
            Query mode.
        """
        mapping_documents = {document[self.key]: document for document in documents}

        candidates = [
            {**candidate, **mapping_documents[candidate[self.key]]}
            for query_candidates in candidates
            for candidate in query_candidates
        ]

        return self.encode_documents(
            documents=candidates,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            query_mode=query_mode,
            **kwargs,
        )

    def __call__(
        self,
        documents: list[list[dict]],
        queries_embeddings: dict[str, torch.Tensor],
        documents_embeddings: dict[str, torch.Tensor],
        tqdm_bar: bool = True,
        k: int = None,
        batch_size: int = 32,
    ) -> list:
        """Retrieve documents.

        Parameters
        ----------
        documents
            Documents to rerank.
        queries_embeddings
            Queries embeddings.
        documents_embeddings
            Documents embeddings.
        tqdm_bar
            Display a tqdm bar.
        k
            Number of documents to retrieve.
        """
        k = k if k is not None else self.n_documents

        ranked = []

        for queries_batch, documents_batch in zip(
            utils.batchify(
                X=list(queries_embeddings.values()),
                batch_size=batch_size,
                desc=f"{self.__class__.__name__} ranker",
                tqdm_bar=tqdm_bar,
            ),
            utils.batchify(
                X=documents,
                batch_size=batch_size,
                tqdm_bar=False,
            ),
        ):
            queries_embeddings_batch = {
                "activations": torch.stack(
                    tensors=[
                        torch.tensor(
                            data=query["activations"],
                            device=self.model.device,
                            dtype=torch.int32,
                        )
                        for query in queries_batch
                    ],
                    dim=0,
                ),
                "embeddings": torch.stack(
                    tensors=[
                        torch.tensor(data=query["embeddings"], device=self.model.device)
                        for query in queries_batch
                    ],
                    dim=0,
                ),
            }

            documents_embeddings_batch = {}
            for query_documents in documents_batch:
                for document in query_documents:
                    documents_embeddings_batch[document[self.key]] = {
                        "activations": torch.tensor(
                            data=documents_embeddings[document[self.key]][
                                "activations"
                            ],
                            device=self.model.device,
                            dtype=torch.int32,
                        ),
                        "embeddings": documents_embeddings[document[self.key]][
                            "embeddings"
                        ],
                    }

            ranked.extend(
                self._rank(
                    queries_embeddings=queries_embeddings_batch,
                    documents_embeddings=documents_embeddings_batch,
                    documents=documents_batch,
                    k=k,
                )
            )

        return ranked

    def _rank(
        self,
        documents: list[list[dict]],
        queries_embeddings: dict[str, torch.Tensor],
        documents_embeddings: dict[str, torch.Tensor],
        k: int,
    ) -> list[list[dict]]:
        """Retrieve documents from input embeddings.

        Parameters
        ----------
        documents
            Documents to rerank.
        queries_embeddings
            Queries embeddings.
        documents_embeddings
            Documents embeddings.
        k
            Number of documents to retrieve.
        """
        documents_activations = [
            [
                documents_embeddings[document[self.key]]["activations"]
                for document in query_matchs
            ]
            for query_matchs in documents
        ]

        intersections = self._get_intersection(
            queries_activations=queries_embeddings["activations"],
            documents_activations=documents_activations,
        )

        queries_tokens_embeddings = [
            {
                token: token_embedding
                for token, token_embedding in zip(query_activations, query_embeddings)
            }
            for query_activations, query_embeddings in zip(
                queries_embeddings["activations"].cpu().detach().tolist(),
                queries_embeddings["embeddings"],
            )
        ]

        documents_tokens_embeddings = [
            [
                {
                    token: token_embedding
                    for token, token_embedding in zip(
                        document_activations.cpu().detach().tolist(),
                        documents_embeddings[document[self.key]]["embeddings"],
                    )
                }
                for document_activations, document in zip(
                    query_documents_activations, query_documents
                )
            ]
            for query_documents_activations, query_documents in zip(
                documents_activations, documents
            )
        ]

        dense_scores = self._get_scores(
            queries_embeddings=queries_tokens_embeddings,
            documents_embeddings=documents_tokens_embeddings,
            intersections=intersections,
        )

        ranked = []

        for query_scores, query_documents in zip(dense_scores, documents):
            if not query_documents:
                ranked.append([])
                continue

            query_scores, query_matchs = torch.topk(
                input=query_scores,
                k=min(k, query_scores.shape[0]),
                dim=-1,
            )

            query_matchs = np.take(
                a=query_documents, indices=query_matchs.cpu().detach().numpy()
            )

            query_scores = query_scores.cpu().detach().numpy()

            ranked.append(
                [
                    {self.key: document[self.key], "similarity": score}
                    for document, score in zip(query_matchs, query_scores)
                ]
            )

        return ranked
