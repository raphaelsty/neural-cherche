__all__ = ["TfIdf"]

import typing

import numpy as np
from scipy.sparse import csc_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils import batchify


class TfIdfRetriever:
    """TfIdf retriever based on cosine similarities.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    documents
        Documents in TFIdf retriever are static. The retriever must be reseted to index new
        documents.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
        will be retrieved.
    tfidf
        TfidfVectorizer class of Sklearn to create a custom TfIdf retriever.

    Examples
    --------

    >>> from sparsembed import retrieve
    >>> from pprint import pprint

    >>> documents = [
    ...     {"id": 0, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 1, "title": "Montreal", "article": "Montreal is in Canada."},
    ...     {"id": 2, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 3, "title": "Montreal", "article": "Montreal is in Canada."},
    ... ]

    >>> retriever = retrieve.TfIdfRetriever(
    ...     key="id",
    ...     on=["title", "article"],
    ... )

    >>> retriever = retriever.add(documents)

    >>> matrix = retriever.transform(texts=["paris", "canada"])
    >>> matrix.shape
    (2, 179)

    >>> pprint(retriever(q=["paris", "canada"], k=4))
    [[{'id': 2, 'similarity': 0.29395219465280986},
      {'id': 0, 'similarity': 0.29395219465280986}],
     [{'id': 3, 'similarity': 0.23284916979662118},
      {'id': 1, 'similarity': 0.23284916979662118}]]

    >>> pprint(retriever(q="paris"))
    [{'id': 2, 'similarity': 0.29395219465280986},
     {'id': 0, 'similarity': 0.29395219465280986}]

    References
    ----------
    1. [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        tfidf: TfidfVectorizer = None,
    ) -> None:
        self.key = key
        self.on = [on] if isinstance(on, str) else on

        self.tfidf = (
            TfidfVectorizer(lowercase=True, ngram_range=(3, 7), analyzer="char")
            if tfidf is None
            else tfidf
        )

        self.matrix = None
        self.fit = False
        self.duplicates = {}
        self.documents = []
        self.n_documents = 0

    def add(
        self,
        documents: list,
        batch_size: int = 100_000,
        tqdm_bar: bool = False,
        fit: bool = True,
        transform: any = None,
        **kwargs,
    ):
        """Add new documents to the TFIDF retriever. The tfidf won't be refitted."""
        transform = self.tfidf.transform if transform is None else transform

        for batch in batchify(
            documents,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
        ):
            batch = [
                document
                for document in batch
                if document[self.key] not in self.duplicates
            ]

            if not batch:
                continue

            texts = [
                " ".join([doc.get(field, "") for field in self.on]) for doc in batch
            ]

            if not self.fit and fit:
                self.tfidf.fit(texts)
                self.fit = True

            matrix = csc_matrix(
                transform(texts, **kwargs),
                dtype=np.float32,
            ).T

            self.matrix = (
                matrix if self.matrix is None else hstack((self.matrix, matrix))
            )

            for document in batch:
                self.documents.append({self.key: document[self.key]})
                self.duplicates[document[self.key]] = True

            self.n_documents += len(batch)

        return self

    def top_k(self, similarities: csc_matrix, k: int):
        """Return the top k documents for each query."""
        matchs, scores = [], []
        for row in similarities:
            _k = min(row.data.shape[0] - 1, k)
            ind = np.argpartition(row.data, kth=_k, axis=0)[:k]
            similarity = np.take_along_axis(row.data, ind, axis=0)
            indices = np.take_along_axis(row.indices, ind, axis=0)
            ind = np.argsort(similarity, axis=0)
            scores.append(-1 * np.take_along_axis(similarity, ind, axis=0))
            matchs.append(np.take_along_axis(indices, ind, axis=0))
        return matchs, scores

    def __call__(
        self,
        q: list[str],
        k: int = None,
        batch_size: int = 2000,
        tqdm_bar: bool = True,
        transform: any = None,
        **kwargs,
    ) -> list[list[dict]]:
        """Retrieve documents from batch of queries.

        Parameters
        ----------
        q
            Either a single query or a list of queries.
        k
            Number of documents to retrieve. Default is `None`, i.e all documents that match the
            query will be retrieved.
        batch_size
            Batch size to use to retrieve documents.
        """
        queries = [q] if isinstance(q, str) else q
        k = k if k is not None else self.n_documents
        transform = self.tfidf.transform if transform is None else transform

        ranked = []

        for batch in batchify(
            queries,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            desc=f"{self.__class__.__name__} retriever",
            tqdm_bar=tqdm_bar,
        ):
            similarities = -1 * transform(batch, **kwargs).dot(self.matrix)
            batch_match, batch_similarities = self.top_k(similarities=similarities, k=k)

            for match, similarities in zip(batch_match, batch_similarities):
                ranked.append(
                    [
                        {**self.documents[idx], "similarity": similarity}
                        for idx, similarity in zip(match, similarities)
                        if similarity > 0
                    ]
                )

        return ranked[0] if isinstance(q, str) else ranked
