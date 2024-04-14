import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import CountVectorizer

from .. import utils

__all__ = ["BM25"]


class BM25:
    """BM25 .

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
        Number of documents to retrieve. Default is `None`, i.e all documents that match
        the query will be retrieved.
    b
        The impact of document length normalization.  Default is `0.75`, Higher will
        penalize longer documents more.
    k1
        How quickly the impact of term frequency saturates.  Default is `1.5`, Higher
        will make term frequency more influential.
    CountVectorizer
        CountVectorizer class of Sklearn to create a custom CountVectorizer counter.
    fit
        Fit the CountVectorizer on the documents. Default is `True`.

    Examples
    --------
    >>> from neural_cherche import retrieve
    >>> from pprint import pprint

    >>> documents = [
    ...     {"id": 0, "document": "Food"},
    ...     {"id": 1, "document": "Sports"},
    ...     {"id": 2, "document": "Cinema"},
    ... ]

    >>> queries = ["Food", "Sports", "Cinema food sports", "cinema"]

    >>> retriever = retrieve.BM25(
    ...     key="id",
    ...     on=["document"],
    ... )

    >>> documents_embeddings = retriever.encode_documents(
    ...     documents=documents
    ... )

    >>> retriever = retriever.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> queries_embeddings = retriever.encode_queries(
    ...     queries=queries
    ... )

    >>> scores = retriever(
    ...     queries_embeddings=queries_embeddings,
    ...     k=4
    ... )

    >>> pprint(scores)
    [[{'id': 0, 'similarity': 4.0525277853012085}],
     [{'id': 1, 'similarity': 8.626796007156372}],
     [{'id': 1, 'similarity': 8.626796007156372},
      {'id': 2, 'similarity': 8.626796007156372},
      {'id': 0, 'similarity': 4.0525277853012085}],
     [{'id': 2, 'similarity': 8.626796007156372}]]

    >>> documents = [
    ...     {"id": 3, "document": "Food"},
    ...     {"id": 4, "document": "Sports"},
    ...     {"id": 5, "document": "Cinema"},
    ... ]

    >>> documents_embeddings = retriever.encode_documents(
    ...     documents=documents
    ... )

    >>> retriever = retriever.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> scores = retriever(
    ...     queries_embeddings=queries_embeddings,
    ...     k=4
    ... )

    >>> pprint(scores)
    [[{'id': 3, 'similarity': 4.254116535186768},
      {'id': 0, 'similarity': 4.0525277853012085}],
     [{'id': 4, 'similarity': 9.055926203727722},
      {'id': 1, 'similarity': 8.626796007156372}],
     [{'id': 5, 'similarity': 9.055926203727722},
      {'id': 4, 'similarity': 9.055926203727722},
      {'id': 2, 'similarity': 8.626796007156372},
      {'id': 1, 'similarity': 8.626796007156372}],
     [{'id': 5, 'similarity': 9.055926203727722},
      {'id': 2, 'similarity': 8.626796007156372}]]

    References
    ----------
    1. [sklearn.feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
    2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        b: float = 0.75,
        k1: float = 1.5,
        epsilon: float = 0,
        count_vectorizer: CountVectorizer = None,
        fit: bool = True,
    ) -> None:
        self.key = key
        self.on = [on] if isinstance(on, str) else on
        self.count_vectorizer = (
            CountVectorizer(lowercase=True, ngram_range=(3, 7), analyzer="char")
            if count_vectorizer is None
            else count_vectorizer
        )
        self.b = b
        self.k1 = k1
        self.epsilon = epsilon
        self.matrix = None
        self.tf = None
        self.fit = fit
        self.documents = []
        self.n_documents = 0

    def encode_documents(self, documents: list[dict]) -> dict[str, csr_matrix]:
        """Encode queries into sparse matrix.

        Parameters
        ----------
        documents
            Documents to encode.

        """
        content = [
            " ".join([doc.get(field, "") for field in self.on]) for doc in documents
        ]

        matrix = None
        if self.fit:
            matrix = self.count_vectorizer.fit_transform(raw_documents=content)
            self.fit = False

        # matrix is a csr matrix of shape (n_documents, n_features)
        if matrix is None:
            matrix = self.count_vectorizer.transform(raw_documents=content)
        return {document[self.key]: row for document, row in zip(documents, matrix)}

    def encode_queries(self, queries: list[str]) -> dict[str, csr_matrix]:
        """Encode queries into sparse matrix.

        Parameters
        ----------
        queries
            Queries to encode.

        """
        if self.fit:
            raise ValueError("You must call the `encode_documents` method first.")

        # matrix is a csr matrix of shape (n_queries, n_features)
        matrix = self.count_vectorizer.transform(raw_documents=queries)
        embeddings = {query: row for query, row in zip(queries, matrix)}

        if len(embeddings) != len(queries):
            utils.duplicates_queries_warning()

        return embeddings

    def add(
        self,
        documents_embeddings: dict[str, csr_matrix],
    ) -> "BM25":
        """Add new documents to the TFIDF retriever. The tfidf won't be refitted."""
        matrix = vstack(
            blocks=[row for row in documents_embeddings.values()], dtype=np.float32
        )

        self.tf = (
            matrix.sum(axis=0) if self.tf is None else self.tf + matrix.sum(axis=0)
        )

        len_documents = (matrix).sum(axis=1)

        average_len_documents = len_documents.mean()

        regularization = np.squeeze(
            np.asarray(
                (
                    self.k1
                    * (1 - self.b + self.b * (len_documents / average_len_documents))
                ).flatten()
            )
        )

        numerator = matrix.copy()
        denominator = matrix.copy().tocsc()

        numerator.data = numerator.data * (self.k1 + 1)
        denominator.data += np.take(a=regularization, indices=denominator.indices)
        matrix.data = (numerator.data / denominator.tocsr().data) + self.epsilon

        for document_key in documents_embeddings:
            self.documents.append({self.key: document_key})

        self.n_documents += len(documents_embeddings)

        idf = np.squeeze(
            a=np.asarray(
                a=np.log((self.n_documents - self.tf + 0.5) / (self.tf + 0.5) + 1)
            )
        )

        matrix = matrix.multiply(idf).T.tocsr()

        self.matrix = (
            matrix if self.matrix is None else hstack(blocks=(self.matrix, matrix))
        )

        return self

    def top_k(self, similarities: csc_matrix, k: int) -> tuple[list, list]:
        """Return the top k documents for each query."""
        matchs, scores = [], []
        for row in similarities:
            _k = min(row.data.shape[0] - 1, k)
            ind = np.argpartition(a=row.data, kth=_k, axis=0)[:k]
            similarity = np.take_along_axis(arr=row.data, indices=ind, axis=0)
            indices = np.take_along_axis(arr=row.indices, indices=ind, axis=0)
            ind = np.argsort(a=similarity, axis=0)
            scores.append(-1 * np.take_along_axis(arr=similarity, indices=ind, axis=0))
            matchs.append(np.take_along_axis(arr=indices, indices=ind, axis=0))
        return matchs, scores

    def __call__(
        self,
        queries_embeddings: dict[str, csr_matrix],
        k: int = None,
        batch_size: int = 2000,
        tqdm_bar: bool = True,
    ) -> list[list[dict]]:
        """Retrieve documents from batch of queries.

        Parameters
        ----------
        queries_embeddings
            Queries embeddings.
        k
            Number of documents to retrieve. Default is `None`, i.e all documents that match the
            query will be retrieved.
        batch_size
            Batch size to use to retrieve documents.
        """
        k = k if k is not None else self.n_documents

        ranked = []

        for batch_embeddings in utils.batchify(
            X=list(queries_embeddings.values()),
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} retriever",
            tqdm_bar=tqdm_bar,
        ):
            # self.matrix is a csr matrix of shape (n_features, n_documents)
            # Transform output a csr matrix of shape (n_queries, n_features)
            similarities = -1 * vstack(blocks=batch_embeddings).dot(self.matrix)
            batch_match, batch_similarities = self.top_k(similarities=similarities, k=k)

            for match, similarities in zip(batch_match, batch_similarities):
                ranked.append(
                    [
                        {**self.documents[idx], "similarity": similarity}
                        for idx, similarity in zip(match, similarities)
                        if similarity > 0
                    ]
                )

        return ranked
