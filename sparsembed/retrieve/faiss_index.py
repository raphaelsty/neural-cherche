import typing

import numpy as np

__all__ = ["FaissIndex"]


class FaissIndex:
    """Faiss index dedicated to vector search.

    Parameters
    ----------
    key
        Identifier field for each document.
    index
        Faiss index to use.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from sparsembed import retrieve
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"id": 0, "title": "Paris France"},
    ...    {"id": 1, "title": "Madrid Spain"},
    ...    {"id": 2, "title": "Montreal Canada"}
    ... ]

    >>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    >>> faiss_index = retrieve.FaissIndex(key="id")

    >>> faiss_index = faiss_index.add(
    ...    documents = documents,
    ...    embeddings = encoder.encode([document["title"] for document in documents]),
    ... )

    >>> print(faiss_index(embeddings=encoder.encode(["Spain", "Montreal"])))
    [[{'id': 1, 'similarity': 0.6544566197822951},
      {'id': 0, 'similarity': 0.5405466290777285},
      {'id': 2, 'similarity': 0.48717489472604614}],
     [{'id': 2, 'similarity': 0.7372165680578416},
      {'id': 0, 'similarity': 0.5185646665953703},
      {'id': 1, 'similarity': 0.4834444940712032}]]

    References
    ----------
    1. [Faiss](https://github.com/facebookresearch/faiss)

    """

    def __init__(self, key, index=None, normalize: bool = True) -> None:
        self.key = key
        self.index = index
        self.documents = []
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.documents)

    def _build(self, embeddings: np.ndarray):
        """Build faiss index.

        Parameters
        ----------
        index
            faiss index.
        embeddings
            Embeddings of the documents.

        """
        if self.index is None:
            try:
                import faiss

                self.index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
            except:
                raise ImportError(
                    "Run `pip install faiss-cpu` or `pip install faiss-gpu` to use this retriever."
                )
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        if not self.index.is_trained and embeddings:
            self.index.train(embeddings)

        self.index.add(embeddings)
        return self.index

    def add(self, documents: list, embeddings: np.ndarray) -> "FaissIndex":
        """Add documents to the faiss index and export embeddings if the path is provided.
        Streaming friendly.

        Parameters
        ----------
        documents
            List of documents as json or list of string to pre-compute queries embeddings.

        """
        array = []
        for document, embedding in zip(documents, embeddings):
            self.documents.append({self.key: document[self.key]})
            array.append(embedding)

        embeddings = np.array(array)
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=-1)[:, None]
        self.index = self._build(embeddings=embeddings)
        return self

    def __call__(
        self,
        embeddings: np.ndarray,
        k: typing.Optional[int] = None,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, typing.Any]]],
        typing.List[typing.Dict[str, typing.Any]],
    ]:
        if k is None:
            k = len(self)

        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=-1)[:, None]

        distances, indexes = self.index.search(embeddings, k)

        # Filter -1 indexes
        matchs = np.take(self.documents, np.where(indexes < 0, 0, indexes))

        rank = []
        for distance, index, match in zip(distances, indexes, matchs):
            rank.append(
                [
                    {
                        **m,
                        "similarity": 1 / (1 + d),
                    }
                    for d, idx, m in zip(distance, index, match)
                    if idx > -1
                ]
            )

        return rank
