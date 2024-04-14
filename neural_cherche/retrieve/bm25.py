import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import CountVectorizer

from .tfidf import TfIdf

__all__ = ["BM25"]


class BM25(TfIdf):
    """BM25 retriever.

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
    epsilon
        Smoothing term. Default is `0`.
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
        super().__init__(
            key=key,
            on=on,
            tfidf=(
                CountVectorizer(lowercase=True, ngram_range=(3, 7), analyzer="char")
                if count_vectorizer is None
                else count_vectorizer
            ),
            fit=fit,
        )

        self.b = b
        self.k1 = k1
        self.epsilon = epsilon
        self.tf = None

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
