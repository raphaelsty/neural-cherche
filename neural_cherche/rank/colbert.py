import torch

from .. import models, utils

__all__ = ["ColBERT"]


class ColBERT:
    """ColBERT ranker.

    Parameters
    ----------
    key
        Document unique identifier.
    on
        Document texts.
    model
        ColBERT model.

    Examples
    --------
    >>> from neural_cherche import models, rank
    >>> from pprint import pprint
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> encoder = models.ColBERT(
    ...     model_name_or_path="raphaelsty/neural-cherche-colbert",
    ...     device="mps",
    ... )

    >>> documents = [
    ...     {"id": 0, "document": "Food"},
    ...     {"id": 1, "document": "Sports"},
    ...     {"id": 2, "document": "Cinema"},
    ... ]

    >>> queries = ["Food", "Sports", "Cinema"]

    >>> ranker = rank.ColBERT(
    ...    key="id",
    ...    on=["document"],
    ...    model=encoder,
    ... )

    >>> queries_embeddings = ranker.encode_queries(
    ...     queries=queries,
    ...     batch_size=3,
    ... )

    >>> documents_embeddings = ranker.encode_documents(
    ...     documents=documents,
    ...     batch_size=3,
    ... )

    >>> scores = ranker(
    ...     documents=[documents for _ in queries],
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     batch_size=3,
    ...     tqdm_bar=True,
    ...     k=3,
    ... )

    >>> pprint(scores)
    [[{'document': 'Food', 'id': 0, 'similarity': 4.7243194580078125},
      {'document': 'Cinema', 'id': 2, 'similarity': 2.403003692626953},
      {'document': 'Sports', 'id': 1, 'similarity': 2.286036252975464}],
     [{'document': 'Sports', 'id': 1, 'similarity': 4.792296886444092},
      {'document': 'Cinema', 'id': 2, 'similarity': 2.6001152992248535},
      {'document': 'Food', 'id': 0, 'similarity': 2.312016487121582}],
     [{'document': 'Cinema', 'id': 2, 'similarity': 5.069696426391602},
      {'document': 'Sports', 'id': 1, 'similarity': 2.5587477684020996},
      {'document': 'Food', 'id': 0, 'similarity': 2.4474282264709473}]]

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        model: models.ColBERT,
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
        """
        # Documents embeddings must be composed of more tokens than queries embeddings
        embeddings = self.encode_queries(
            queries=[
                " ".join([document[field] for field in self.on])
                for document in documents
            ],
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            query_mode=query_mode,
            **kwargs,
        )

        return {
            document[self.key]: embedding
            for document, embedding in zip(documents, embeddings.values())
        }

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
            Queries.
        batch_size
            Batch size.
        tqdm_bar
            Show tqdm bar.
        """
        embeddings = {}

        for batch_texts in utils.batchify(
            X=queries, batch_size=batch_size, tqdm_bar=tqdm_bar
        ):
            batch_embeddings = self.model.encode(
                texts=batch_texts,
                query_mode=query_mode,
                **kwargs,
            )

            batch_embeddings = (
                batch_embeddings["embeddings"].cpu().detach().numpy().astype("float32")
            )

            for query, embedding in zip(batch_texts, batch_embeddings):
                embeddings[query] = embedding

        if len(embeddings) != len(queries):
            utils.duplicates_queries_warning()

        return embeddings

    def __call__(
        self,
        documents: list[list[dict]],
        queries_embeddings: dict[str, torch.Tensor],
        documents_embeddings: dict[str, torch.Tensor],
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
        queries_embeddings
            Queries embeddings.
        documents_embeddings
            Documents embeddings.
        batch_size
            Batch size.
        tqdm_bar
            Show tqdm bar.
        k
            Number of documents to retrieve.
        """
        scores = []

        for (query, query_embedding), query_documents in zip(
            queries_embeddings.items(), documents
        ):
            query_scores = []

            embedding_query = torch.tensor(
                query_embedding,
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
                            documents_embeddings[document[self.key]],
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

        return self._rank(scores=scores, documents=documents, k=k)

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
