import itertools

import torch
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from ..rank import ColBERT
from ..retrieve import BM25 as BM25Retriever

__all__ = ["BM25"]


class BM25(BM25Retriever):
    """BM25 explorer.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    documents
        Documents in TFIdf retriever are static. The retriever must be reseted to index new
        documents.
    CountVectorizer
        CountVectorizer class of Sklearn to create a custom CountVectorizer counter.
    b
        The impact of document length normalization.  Default is `0.75`, Higher will
        penalize longer documents more.
    k1
        How quickly the impact of term frequency saturates.  Default is `1.5`, Higher
        will make term frequency more influential.
    epsilon
        Smoothing term. Default is `0`.
    fit
        Fit the CountVectorizer on the documents. Default is `True`.

    Examples
    --------
    >>> from neural_cherche import explore, models, rank
    >>> from pprint import pprint

    >>> documents = [
    ...     {"id": 0, "document": "Food"},
    ...     {"id": 1, "document": "Sports"},
    ...     {"id": 2, "document": "Cinema"},
    ... ]

    >>> model = models.ColBERT(
    ...      model_name_or_path="raphaelsty/neural-cherche-colbert",
    ... )

    >>> ranker = rank.ColBERT(
    ...     model=model,
    ...     key="id",
    ...     on=["document"],
    ... )

    >>> explorer = explore.BM25(
    ...     key="id",
    ...     on=["document"],
    ...     ranker=ranker,
    ... )

    >>> documents_embeddings = explorer.encode_documents(
    ...     documents=documents
    ... )

    >>> explorer = explorer.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> queries = ["Food", "Sports", "Cinema food sports", "cinema"]
    >>> queries_embeddings = explorer.encode_queries(
    ...     queries=queries
    ... )

    >>> scores = explorer(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     k=10,
    ...     ranker_batch_size=32,
    ...     retriever_batch_size=2000,
    ...     max_step=3,
    ...     beam_size=3,
    ... )

    >>> pprint(scores)

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        ranker: ColBERT,
        count_vectorizer: CountVectorizer = None,
        b: float = 0.75,
        k1: float = 1.5,
        epsilon: float = 0,
        fit: bool = True,
    ) -> None:
        super().__init__(
            key=key,
            on=on,
            count_vectorizer=count_vectorizer,
            b=b,
            k1=k1,
            epsilon=epsilon,
            fit=fit,
        )

        self.ranker = ranker
        self.mapping_documents = {}

    def encode_documents(
        self,
        documents: list[dict],
        ranker_embeddings: bool = False,
        batch_size: int = 32,
        query_mode: bool = False,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> dict[str, csr_matrix]:
        """Encode documents."""
        embeddings = {
            "retriever": super().encode_documents(
                documents=documents,
            ),
            "ranker": {},
        }

        for document in documents:
            self.mapping_documents[document[self.key]] = document

        if ranker_embeddings:
            embeddings["ranker"] = self.ranker.encode_documents(
                documents=documents,
                batch_size=batch_size,
                query_mode=query_mode,
                tqdm_bar=tqdm_bar,
                **kwargs,
            )

        return embeddings

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        query_mode: bool = True,
        tqdm_bar: bool = True,
        warn_duplicates: bool = True,
        **kwargs,
    ) -> dict[str, csr_matrix]:
        """Encode queries."""
        return {
            "retriever": super().encode_queries(
                queries=queries, warn_duplicates=warn_duplicates
            ),
            "ranker": self.ranker.encode_queries(
                queries=queries,
                batch_size=batch_size,
                query_mode=query_mode,
                tqdm_bar=tqdm_bar,
                **kwargs,
            ),
        }

    def add(self, documents_embeddings: dict[dict[str, torch.Tensor]]) -> "BM25":
        """Add new documents to the BM25 retriever."""
        super().add(documents_embeddings=documents_embeddings["retriever"])

        return self

    def __call__(
        self,
        queries_embeddings: dict[str, dict[str, csr_matrix] | dict[str, torch.Tensor]],
        documents_embeddings: dict[
            str, dict[str, csr_matrix] | dict[str, torch.Tensor]
        ],
        k: int = 30,
        beam_size: int = 3,
        max_step: int = 3,
        retriever_batch_size: int = 2000,
        ranker_batch_size: int = 32,
        early_stopping: bool = False,
        tqdm_bar: bool = False,
        queries: list[str] = None,
        actual_step: int = 0,
        scores: list[dict] = None,
    ) -> list[list[dict]]:
        """Explore the documents."""
        scores = (
            [{} for _ in queries_embeddings["retriever"]] if scores is None else scores
        )

        queries = (
            queries
            if queries is not None
            else [[query] for query in list(queries_embeddings["retriever"].keys())]
        )

        retriever_queries_embeddings = super().encode_queries(
            queries=list(
                set([query for group_queries in queries for query in group_queries])
            ),
            warn_duplicates=False,
        )

        # Retrieve the top k documents
        candidates = super().__call__(
            queries_embeddings=retriever_queries_embeddings,
            k=k,
            batch_size=retriever_batch_size,
            tqdm_bar=tqdm_bar,
        )

        # Start post-process candidates retriever.
        mapping_position = {
            query: position
            for position, query in enumerate(
                iterable=list(retriever_queries_embeddings.keys())
            )
        }

        # Map candidates back to queries and avoid duplicates and avoid already scored documents
        candidates = [
            [
                [
                    document
                    for document in candidates[mapping_position[query]]
                    if document[self.key] not in query_scores
                ]
                if query in mapping_position
                else []
                for query in group_queries
            ]
            for group_queries, query_scores in zip(queries, scores)
        ]

        candidates = list(itertools.chain.from_iterable(candidates))

        # Drop duplicates
        distinct_candidates = []
        for query_candidates in candidates:
            distinct_candidates_query, duplicates = [], {}
            for document in query_candidates:
                if document[self.key] not in duplicates:
                    distinct_candidates_query.append(document)
                    duplicates[document[self.key]] = True
            distinct_candidates.append(distinct_candidates_query)
        candidates = distinct_candidates

        print(candidates, queries)
        # End post-process candidates retriever.

        # Encoding documents
        documents_to_encode, duplicates = [], {}
        for query_documents in candidates:
            for document in query_documents:
                if (
                    document[self.key] not in documents_embeddings["ranker"]
                    and document[self.key] not in duplicates
                ):
                    documents_to_encode.append(
                        self.mapping_documents[document[self.key]]
                    )

                    duplicates[document[self.key]] = True

        if documents_to_encode:
            documents_embeddings["ranker"].update(
                self.ranker.encode_documents(
                    documents=documents_to_encode,
                    batch_size=ranker_batch_size,
                    tqdm_bar=False,
                )
            )
        # End encoding documents

        # Rank the candidates and take the top k
        candidates = self.ranker(
            documents=candidates,
            queries_embeddings=queries_embeddings["ranker"],
            documents_embeddings=documents_embeddings["ranker"],
            batch_size=ranker_batch_size,
            tqdm_bar=tqdm_bar,
        )

        scores = [
            {
                **query_scores,
                **{
                    document[self.key]: document["similarity"]
                    for document in query_documents
                },
            }
            for query_scores, query_documents in zip(scores, candidates)
        ]

        if (actual_step - 1) > max_step:
            return scores

        # Add early stopping
        # Take beam_size top candidates which are not in query_explored
        # Create query explored
        top_candidates = candidates

        queries = [
            [
                f"{query} {' '.join([self.mapping_documents[document[self.key]][field] for field in self.on])}"
                for document in query_documents
            ][:beam_size]
            for query, query_documents in zip(
                list(queries_embeddings["retriever"].keys()), top_candidates
            )
        ]

        print(len(scores), len(queries))

        return self(
            queries_embeddings=queries_embeddings,
            documents_embeddings=documents_embeddings,
            k=k,
            beam_size=beam_size,
            max_step=max_step,
            retriever_batch_size=retriever_batch_size,
            ranker_batch_size=ranker_batch_size,
            tqdm_bar=tqdm_bar,
            queries=queries,
            actual_step=actual_step + 1,
            scores=scores,
        )
