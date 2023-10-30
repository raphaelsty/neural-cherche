from cherche import retrieve
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["TfIdfRetriever"]


class TfIdfRetriever(retrieve.TfIdf):
    """TfIdfRetriever model.

    Parameters
    ----------
    key
        Key of the documents.
    on
        Columns to use for the documents.
    documents
        Documents to index.
    tfidf
        TfIdfVectorizer.
    k
        Number of documents to retrieve.
    batch_size
        Batch size.
    fit
        Whether to fit the model.


    Examples
    --------
    >>> from sparsembed import retrieve
    >>> from pprint import pprint as print

    >>> documents = [
    ...     {"id": 0, "document": "Food"},
    ...     {"id": 1, "document": "Sports"},
    ...     {"id": 2, "document": "Cinema"},
    ... ]

    >>> retriever = retrieve.TfIdfRetriever(
    ...     key="id",
    ...     on="document",
    ...     documents=documents,
    ... )

    >>> retriever = retriever.add(
    ...     documents=documents,
    ...     batch_size=32,
    ... )

    >>> candidates = retriever(
    ...     ["Food", "Sports", "Cinema"],
    ...     batch_size=32,
    ...     k=2,
    ... )

    >>> print(candidates)
    [[{'id': 0, 'similarity': 0.9999999932782064}],
     [{'id': 1, 'similarity': 0.9999999880261985}],
     [{'id': 2, 'similarity': 0.9999999880261985}]]

    """

    def __init__(
        self,
        key: str,
        on: str | list,
        documents: list[dict[str, str]] = None,
        tfidf: TfidfVectorizer = None,
        k: int | None = None,
        batch_size: int = 1024,
        fit: bool = True,
    ) -> None:
        super().__init__(key, on, documents, tfidf, k, batch_size, fit)
