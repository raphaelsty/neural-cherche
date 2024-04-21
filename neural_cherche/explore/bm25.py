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

    >>> queries = ["Food", "Sports", "Cinema food sports", "cinema"]

    >>> documents = [
    ...     {"id": id, "document": queries[id%4]} for id in range(100)
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
    ...     documents=documents,
    ...     ranker_embeddings=False,
    ... )

    >>> explorer = explorer.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> queries_embeddings = explorer.encode_queries(
    ...     queries=queries
    ... )

    >>> scores = explorer(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     k=10,
    ...     early_stopping=True,
    ...     ranker_batch_size=32,
    ...     retriever_batch_size=2000,
    ...     max_step=3,
    ...     beam_size=3,
    ... )

    >>> pprint(scores)
    [[{'id': 96, 'similarity': 4.7243194580078125},
      {'id': 24, 'similarity': 4.7243194580078125},
      {'id': 60, 'similarity': 4.7243194580078125},
      {'id': 20, 'similarity': 4.7243194580078125},
      {'id': 56, 'similarity': 4.7243194580078125},
      {'id': 52, 'similarity': 4.7243194580078125},
      {'id': 0, 'similarity': 4.7243194580078125},
      {'id': 48, 'similarity': 4.7243194580078125},
      {'id': 36, 'similarity': 4.7243194580078125},
      {'id': 40, 'similarity': 4.7243194580078125}],
     [{'id': 97, 'similarity': 4.792297840118408},
      {'id': 25, 'similarity': 4.792297840118408},
      {'id': 61, 'similarity': 4.792297840118408},
      {'id': 21, 'similarity': 4.792297840118408},
      {'id': 57, 'similarity': 4.792297840118408},
      {'id': 53, 'similarity': 4.792297840118408},
      {'id': 1, 'similarity': 4.792297840118408},
      {'id': 49, 'similarity': 4.792297840118408},
      {'id': 37, 'similarity': 4.792297840118408},
      {'id': 41, 'similarity': 4.792297840118408}],
     [{'id': 74, 'similarity': 7.377876281738281},
      {'id': 82, 'similarity': 7.377876281738281},
      {'id': 62, 'similarity': 7.377876281738281},
      {'id': 94, 'similarity': 7.377876281738281},
      {'id': 70, 'similarity': 7.377876281738281},
      {'id': 66, 'similarity': 7.377876281738281},
      {'id': 78, 'similarity': 7.377876281738281},
      {'id': 2, 'similarity': 7.377876281738281},
      {'id': 90, 'similarity': 7.377876281738281},
      {'id': 46, 'similarity': 7.377876281738281}],
     [{'id': 31, 'similarity': 5.06969690322876},
      {'id': 23, 'similarity': 5.06969690322876},
      {'id': 55, 'similarity': 5.069695472717285},
      {'id': 47, 'similarity': 5.069695472717285},
      {'id': 43, 'similarity': 5.069695472717285},
      {'id': 39, 'similarity': 5.069695472717285},
      {'id': 35, 'similarity': 5.069695472717285},
      {'id': 63, 'similarity': 5.069695472717285},
      {'id': 27, 'similarity': 5.069695472717285},
      {'id': 11, 'similarity': 5.069695472717285}]]

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

    def _encode_queries_retriever(
        self, queries: list, queries_embeddings: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Encode queries for the retriever."""
        return super().encode_queries(
            queries=list(
                set(
                    [
                        query
                        for group_queries in queries
                        for query in group_queries
                        if query
                    ]
                )
            ),
            warn_duplicates=False,
        )

    def _encode_documents_ranker(
        self,
        candidates: list[list[dict]],
        documents_embeddings: dict[str, csr_matrix],
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Encode documents for the ranker."""
        documents_to_encode, duplicates = [], {}
        for query_documents in candidates:
            for document in query_documents:
                if (
                    document[self.key] not in documents_embeddings
                    and document[self.key] not in duplicates
                ):
                    documents_to_encode.append(
                        self.mapping_documents[document[self.key]]
                    )

                    duplicates[document[self.key]] = True

        if documents_to_encode:
            documents_embeddings.update(
                self.ranker.encode_documents(
                    documents=documents_to_encode,
                    batch_size=batch_size,
                    tqdm_bar=False,
                )
            )

        return documents_embeddings

    def _post_process_candidates_retriever(
        self,
        queries_embeddings: dict,
        queries: list[str],
        candidates: list[list[dict]],
        documents_explored: list[dict],
        k: int,
    ) -> list[list[dict]]:
        """Post-process candidates from the retriever."""
        mapping_position = {
            query: position
            for position, query in enumerate(iterable=list(queries_embeddings.keys()))
        }

        # Gather all the documents retrieved for the same query
        candidates = [
            list(
                itertools.chain.from_iterable(
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
                )
            )
            for group_queries, query_scores in zip(queries, documents_explored)
        ]

        # Drop duplicates documents retrieved for the same query.
        distinct_candidates = []
        for query_candidates in candidates:
            distinct_candidates_query, duplicates = [], {}
            for document in query_candidates:
                if document[self.key] not in duplicates:
                    distinct_candidates_query.append(document)
                    duplicates[document[self.key]] = True
            distinct_candidates.append(distinct_candidates_query)

        return distinct_candidates

    def _get_next_queries(
        self,
        candidates: list[list[dict]],
        queries_embeddings: dict[str, csr_matrix],
        documents_explored: list[dict],
        beam_size: int,
        max_scores: list[float],
        early_stopping: bool,
    ) -> tuple[list[str], list[float], list[dict]]:
        """Get the next queries to explore."""
        next_queries, next_max_scores = [], []

        for query, query_documents, query_documents_explored, query_max_score in zip(
            list(queries_embeddings.keys()), candidates, documents_explored, max_scores
        ):
            query_next_queries = []
            early_stopping_condition = False

            for document in query_documents:
                if document[self.key] not in query_documents_explored:
                    if (
                        document["similarity"] >= query_max_score and early_stopping
                    ) or (early_stopping_condition and early_stopping):
                        if document["similarity"] > query_max_score:
                            query_max_score = document["similarity"]

                        early_stopping_condition = True
                        query_documents_explored[document[self.key]] = True
                        query_next_queries.append(
                            f"{query} {' '.join([self.mapping_documents[document[self.key]][field] for field in self.on])}"
                        )

                if len(query_next_queries) >= beam_size:
                    break

            next_max_scores.append(query_max_score)
            next_queries.append(query_next_queries)

        return next_queries, next_max_scores, documents_explored

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
        documents_explored: list[dict] = None,
        max_scores: list[float] = None,
    ) -> list[list[dict]]:
        """Explore the documents.

        Parameters
        ----------
        queries_embeddings
            Queries embeddings.
        documents_embeddings
            Documents embeddings.
        k
            Number of documents to retrieve.
        beam_size
            Among the top k documents retrieved, how many documents to explore.
        max_step
            Maximum number of steps to explore.
        retriever_batch_size
            Batch size for the retriever.
        ranker_batch_size
            Batch size for the ranker.
        early_stopping
            Number of step to perform the exploration until the ranker did not spot better
            documents.
        """
        queries = (
            queries
            if queries is not None
            else [[query] for query in list(queries_embeddings["retriever"].keys())]
        )

        scores = (
            [{} for _ in queries_embeddings["retriever"]] if scores is None else scores
        )

        max_scores = (
            [0 for _ in queries_embeddings["retriever"]]
            if max_scores is None
            else max_scores
        )

        documents_explored = (
            documents_explored
            if documents_explored is not None
            else [{} for _ in queries_embeddings["retriever"]]
        )

        retriever_queries_embeddings = self._encode_queries_retriever(
            queries=queries,
            queries_embeddings=queries_embeddings["retriever"],
        )

        # Retrieve the top k documents
        candidates = super().__call__(
            queries_embeddings=retriever_queries_embeddings,
            k=k,
            batch_size=retriever_batch_size,
            tqdm_bar=tqdm_bar,
        )

        # Start post-process candidates retriever.
        candidates = self._post_process_candidates_retriever(
            queries_embeddings=retriever_queries_embeddings,
            queries=queries,
            candidates=candidates,
            documents_explored=documents_explored,
            k=k,
        )

        # Encoding documents
        documents_embeddings["ranker"] = self._encode_documents_ranker(
            candidates=candidates,
            documents_embeddings=documents_embeddings["ranker"],
            batch_size=ranker_batch_size,
        )

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
            return self._rank(scores=scores, k=k)

        # Add early stopping
        queries, max_scores, documents_explored = self._get_next_queries(
            queries_embeddings=queries_embeddings["retriever"],
            candidates=candidates,
            documents_explored=documents_explored,
            beam_size=beam_size,
            max_scores=max_scores,
            early_stopping=early_stopping,
        )

        return self(
            queries_embeddings=queries_embeddings,
            documents_embeddings=documents_embeddings,
            k=k,
            early_stopping=early_stopping,
            beam_size=beam_size,
            max_step=max_step,
            retriever_batch_size=retriever_batch_size,
            ranker_batch_size=ranker_batch_size,
            tqdm_bar=tqdm_bar,
            queries=queries,
            actual_step=actual_step + 1,
            scores=scores,
            documents_explored=documents_explored,
            max_scores=max_scores,
        )

    def _rank(self, scores: list[dict], k: int) -> list[dict]:
        """Rank the scores."""
        return [
            [
                {self.key: key, "similarity": similarity}
                for key, similarity in sorted(
                    query_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ][:k]
            for query_scores in scores
        ]
