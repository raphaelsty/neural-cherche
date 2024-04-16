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
    device
        Device to use. default is model device.

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

    >>> scores = ranker(
    ...     documents=[documents, [], documents],
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
     [],
     [{'document': 'Cinema', 'id': 2, 'similarity': 5.069696426391602},
      {'document': 'Sports', 'id': 1, 'similarity': 2.5587477684020996},
      {'document': 'Food', 'id': 0, 'similarity': 2.4474282264709473}]]

    >>> documents_embeddings = ranker.encode_candidates_documents(
    ...     documents=documents,
    ...     candidates=[documents for _ in queries],
    ...     batch_size=3,
    ... )

    >>> assert len(documents_embeddings) == 3
    """

    def __init__(
        self,
        key: str,
        on: list[str],
        model: models.ColBERT,
        device: str = None,
    ) -> None:
        self.key = key
        self.on = on if isinstance(on, list) else [on]
        self.model = model
        self.device = self.model.device if device is None else device

    def encode_documents(
        self,
        documents: list[dict] | list[list[dict]],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        query_mode: bool = False,
        desc: str = "documents embeddings",
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
        if not documents:
            return {}

        embeddings = self.encode_queries(
            queries=[
                " ".join([document[field] for field in self.on])
                for document in documents
            ],
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            query_mode=query_mode,
            desc=desc,
            warn_duplicates=False,
            **kwargs,
        )

        return {
            document[self.key]: embedding
            for document, embedding in zip(documents, embeddings.values())
        }

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

        documents_flatten, duplicates = [], {}
        for query_documents in candidates:
            for document in query_documents:
                if document[self.key] not in duplicates:
                    duplicates[document[self.key]] = True
                    documents_flatten.append(mapping_documents[document[self.key]])
        candidates = documents_flatten

        return self.encode_documents(
            documents=candidates,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            query_mode=query_mode,
            **kwargs,
        )

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        query_mode: bool = True,
        warn_duplicates: bool = True,
        desc: str = "queries embeddings",
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
            X=queries,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            desc=f"{self.__class__.__name__} {desc}",
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

        if len(embeddings) != len(queries) and warn_duplicates:
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
        bar = utils.batchify(
            X=documents,
            batch_size=1,
            tqdm_bar=tqdm_bar,
            desc=f"{self.__class__.__name__} ranker",
        )

        scores = []

        for (query, query_embedding), query_documents in zip(
            queries_embeddings.items(), bar
        ):
            query_scores = []

            embedding_query = torch.tensor(
                data=query_embedding,
                device=self.device,
                dtype=torch.float32,
            )

            for batch_query_documents in utils.batchify(
                X=query_documents[0],
                batch_size=batch_size,
                tqdm_bar=False,
            ):
                embeddings_batch_documents = torch.stack(
                    tensors=[
                        torch.tensor(
                            data=documents_embeddings[document[self.key]],
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

            if not query_scores:
                scores.append([])
                continue

            scores.append(torch.cat(tensors=query_scores, dim=0))

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
            if not query_documents:
                ranked.append([])
                continue

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
