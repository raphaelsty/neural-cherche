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
    ...     embedding_size=64,
    ...     device="mps",
    ... )

    >>> ranker = rank.ColBERTRanker(
    ...    key="id",
    ...    on=["text"],
    ...    model=encoder,
    ... )

    >>> q = ["Berlin", "Paris", "London"]

    >>> documents = [
    ...     {"id": 1, "text": "Berlin is the capital of Germany"},
    ...     {"id": 2, "text": "Paris is the capital of France"},
    ...     {"id": 3, "text": "London is the capital of England"},
    ... ]

    >>> embeddings_queries = ranker.encode_queries(
    ...     q=q,
    ...     batch_size=2,
    ... )

    >>> embeddings_documents = ranker.encode_documents(
    ...     documents=documents,
    ... )

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

    >>> matchs = ranker(
    ...     q=q[0],
    ...     documents=documents,
    ...     embeddings_queries=embeddings_queries,
    ...     embeddings_documents=embeddings_documents,
    ...     batch_size=2,
    ...     tqdm_bar=True,
    ...     k=2,
    ... )

    >>> matchs
    [{'id': 1, 'text': 'Berlin is the capital of Germany', 'similarity': 0.44193798303604126}, {'id': 2, 'text': 'Paris is the capital of France', 'similarity': 0.37626248598098755}]

    """

    def __init__(self, key: str, on: list[str], model: ColBERT) -> None:
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
        padding: bool = True,
        add_special_tokens: bool = False,
        max_length: int = 256,
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
        padding
            Pad the inputs.
        add_special_tokens
            Add special tokens.
        max_length
            Maximum length of the inputs.
        """
        embeddings = self.encode_queries(
            q=[
                " ".join([document[field] for field in self.on])
                for document in documents
            ],
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            truncation=truncation,
            padding=padding,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            **kwargs,
        )

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
        padding: bool = True,
        add_special_tokens: bool = False,
        max_length: int = 256,
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
        padding
            Pad the inputs.
        add_special_tokens
            Add special tokens.
        max_length
            Maximum length of the inputs.
        """
        embeddings = {}

        for batch_texts in utils.batchify(
            X=q, batch_size=batch_size, tqdm_bar=tqdm_bar
        ):
            batch_embeddings = self.model.encode(
                texts=batch_texts,
                truncation=truncation,
                padding=padding,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                **kwargs,
            )

            for query, embedding in zip(batch_texts, batch_embeddings):
                embeddings[query] = embedding

        return embeddings

    def rank_embeddings(
        self,
        embeddings_queries: torch.Tensor,
        embeddings_documents: torch.Tensor,
        batch_size: int,
        tqdm_bar: bool = False,
    ) -> None:
        """Rank documents given embeddings.

        Parameters
        ----------
        embeddings_queries
            Queries embeddings.
        embeddings_documents
            Documents embeddings.
        batch_size
            Batch size.
        tqdm_bar
            Show tqdm bar.
        """
        scores = []

        for embedding_query, embeddings_query_documents in zip(
            embeddings_queries, embeddings_documents
        ):
            query_scores = []

            for batch_embeddings_query_documents in utils.batchify(
                X=embeddings_query_documents,
                batch_size=batch_size,
                tqdm_bar=tqdm_bar,
            ):
                # The query must contain less tokens than the documents.
                # We are adding null tensors to the documents embeddings.
                if embedding_query.shape[0] > batch_embeddings_query_documents.shape[1]:
                    n, s, h = batch_embeddings_query_documents.shape
                    
                    null_tensor = torch.zeros(
                        n, embedding_query.shape[0] - s, h, device=self.model.device
                    )
                    
                    batch_embeddings_query_documents = torch.cat(
                        (batch_embeddings_query_documents, null_tensor), dim=1
                    )

                query_scores.append(
                    torch.einsum(
                        "sh,nsh->n",
                        embedding_query,
                        batch_embeddings_query_documents,
                    )
                )

            scores.append(torch.cat(query_scores, dim=0))

        return scores

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

        embeddings_queries = [embeddings_queries[query] for query in queries]

        embeddings_documents = [
            torch.stack(
                [
                    embeddings_documents[document[self.key]]
                    for document in query_documents
                ],
                dim=0,
            )
            for query_documents in docs
        ]

        scores = self.rank_embeddings(
            embeddings_queries=embeddings_queries,
            embeddings_documents=embeddings_documents,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
        )

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
