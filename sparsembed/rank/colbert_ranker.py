import torch

from .. import utils
from ..models import ColBERT

__all__ = ["ColBERTRanker"]


class ColBERTRanker:
    """ColBERT ranker.

    Parameters
    ----------
    model
        ColBERT model.

    Example
    -------
    >>> from sparsembed import models, rank
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> encoder = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    ...     embedding_size=128,
    ...     device="mps",
    ...     max_length_query=32,
    ...     max_length_document=350,
    ... )

    >>> ranker = rank.ColBERTRanker(
    ...    key="id",
    ...    on=["text"],
    ...    model=encoder,
    ... )

    >>> q = ["Berlin", "Paris", "London"]

    >>> documents = [
    ...     {"id": 1, "text": "Berlin is the capital of Germany"},
    ...     {"id": 2, "text": "Paris is the capital of France and France is in Europe"},
    ...     {"id": 3, "text": "London is the capital of England"},
    ... ]

    >>> embeddings_queries = ranker.encode_queries(
    ...     q=q,
    ...     batch_size=2,
    ... )

    >>> embeddings_documents = ranker.encode_documents(
    ...     documents=documents,
    ...     batch_size=1,
    ... )

    >>> for query, query_embedding in embeddings_queries.items():
    ...     assert query_embedding.shape[0] == 32

    >>> for query, documents_embedding in embeddings_documents.items():
    ...     assert documents_embedding.shape[0] == 350

    >>> matchs = ranker(
    ...     q=q,
    ...     documents=[documents for _ in q],
    ...     embeddings_queries=embeddings_queries,
    ...     embeddings_documents=embeddings_documents,
    ...     batch_size=2,
    ...     tqdm_bar=True,
    ...     k=2,
    ... )

    >>> assert len(matchs) == 3
    >>> assert len(matchs[0]) == 2

    >>> ranker(
    ...     q=q,
    ...     documents=[documents for _ in q],
    ...     embeddings_queries=embeddings_queries,
    ...     embeddings_documents=embeddings_documents,
    ...     batch_size=1,
    ...     tqdm_bar=True,
    ...     k=1,
    ... )
    [[{'id': 1, 'text': 'Berlin is the capital of Germany', 'similarity': 20.214763641357422}], [{'id': 2, 'text': 'Paris is the capital of France and France is in Europe', 'similarity': 16.75994873046875}], [{'id': 3, 'text': 'London is the capital of England', 'similarity': 18.290054321289062}]]

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        model: ColBERT,
    ) -> None:
        self.key = key
        self.on = on if isinstance(on, list) else [on]
        self.model = model
        self.device = self.model.device

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = False,
        query_mode: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Encode documents.

        Parameters
        ----------
        documents
            Documents.
        batch_size
            Batch size.
        tqdm_bar
            Show tqdm bar.
        truncation
            Truncate the inputs.
        add_special_tokens
            Add special tokens.
        max_length
            Maximum length of the inputs.
        """
        # Documents embeddings must be composed of more tokens than queries embeddings
        embeddings = self.encode_queries(
            q=[
                " ".join([document[field] for field in self.on])
                for document in documents
            ],
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            query_mode=query_mode,
            **kwargs,
        )

        for _, embedding in embeddings.items():
            if embedding.shape[0] < self.model.max_length_document:
                raise ValueError(embedding.shape)

        return {
            document[self.key]: embedding
            for document, embedding in zip(documents, embeddings.values())
        }

    def encode_queries(
        self,
        q: list[str],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = False,
        query_mode: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Encode queries.

        Parameters
        ----------
        q
            Queries.
        batch_size
            Batch size.
        tqdm_bar
            Show tqdm bar.
        truncation
            Truncate the inputs.
        add_special_tokens
            Add special tokens.
        query
            Wether the model encode queries or documents.
        """
        embeddings = {}

        for batch_texts in utils.batchify(
            X=q, batch_size=batch_size, tqdm_bar=tqdm_bar
        ):
            batch_embeddings = self.model.encode(
                texts=batch_texts,
                truncation=truncation,
                add_special_tokens=add_special_tokens,
                query_mode=query_mode,
                **kwargs,
            )

            batch_embeddings = batch_embeddings.cpu().detach().numpy().astype("float32")

            for query, embedding in zip(batch_texts, batch_embeddings):
                embeddings[query] = embedding

        return embeddings

    def __call__(
        self,
        q: list[str],
        documents: list[list[dict]],
        embeddings_queries: dict[str, torch.Tensor],
        embeddings_documents: dict[str, torch.Tensor],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        k: int = None,
    ) -> list[list[str]]:
        """Rank documents  givent queries.

        Parameters
        ----------
        queries
            Queries.
        documents
            Documents.
        embeddings_queries
            Queries embeddings.
        embeddings_documents
            Documents embeddings.
        batch_size
            Batch size.
        tqdm_bar
            Show tqdm bar.
        k
            Number of documents to retrieve.
        """
        queries, docs, missing_documents = self._sanitize_input(
            q=q,
            documents=documents,
        )

        if not queries:
            return self._sanitize_output(
                q=q, ranked=[], missing_documents=missing_documents
            )

        scores = []

        for query, query_documents in zip(queries, docs):
            query_scores = []

            embedding_query = torch.tensor(
                embeddings_queries[query],
                device=self.device,
                dtype=torch.float32,
            )

            for batch_query_documents in utils.batchify(
                X=query_documents,
                batch_size=batch_size,
                tqdm_bar=tqdm_bar,
            ):
                embeddings_batch_documents = torch.stack(
                    [
                        torch.tensor(
                            embeddings_documents[document[self.key]],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        for document in batch_query_documents
                    ],
                    dim=0,
                )

                query_documents_scores = torch.einsum(
                    "sh,bth->bst",
                    embedding_query,
                    embeddings_batch_documents,
                )

                query_scores.append(
                    query_documents_scores.max(dim=2).values.sum(axis=1)
                )

            scores.append(torch.cat(query_scores, dim=0))

        return self._sanitize_output(
            q=q,
            ranked=self._rank(scores=scores, documents=docs, k=k),
            missing_documents=missing_documents,
        )

    def _rank(
        self, scores: torch.Tensor, documents: list[list[dict]], k: int
    ) -> list[list[dict]]:
        """Rank documents by scores.

        Parameters
        ----------
        scores
            Scores.
        documents
            Documents.
        k
            Number of documents to retrieve.
        """
        ranked = []

        for query_scores, query_documents in zip(scores, documents):
            top_k = torch.topk(
                input=query_scores,
                k=min(k, len(query_documents))
                if k is not None
                else len(query_documents),
                dim=-1,
            )

            ranked.append(
                [
                    {**query_documents[indice], "similarity": similarity}
                    for indice, similarity in zip(top_k.indices, top_k.values.tolist())
                ]
            )

        return ranked

    @staticmethod
    def _sanitize_input(q: list[str], documents):
        """Sanitize queries and documents.

        Parameters
        ----------
        q
            Queries.
        documents
            Documents.
        """
        queries, docs = [], []
        missing_documents = {}
        for index, (query, query_documents) in enumerate(
            zip(
                [q] if isinstance(q, str) else q,
                [documents] if isinstance(documents[0], dict) else documents,
            )
        ):
            if query_documents:
                queries.append(query)
                docs.append(query_documents)
            else:
                missing_documents[index] = True

        return queries, docs, missing_documents

    @staticmethod
    def _sanitize_output(
        q: list[str], ranked: list[list[dict]], missing_documents: dict[int, bool]
    ):
        """Sanitize output ranked documents.

        Parameters
        ----------
        q
            Queries.
        ranked
            Ranked documents.
        missing_documents
            Missing documents.
        """
        for index in missing_documents:
            ranked.insert(index, [])

        return ranked[0] if isinstance(q, str) else ranked
