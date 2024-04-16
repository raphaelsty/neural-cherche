import os

from scipy.sparse import csr_matrix

from .. import models, utils
from .tfidf import TfIdf

__all__ = ["Splade"]


class Splade(TfIdf):
    """Retriever class.

    Parameters
    ----------
    key
        Document unique identifier.
    on
        Document texts.
    model
        SparsEmbed model.

    Examples
    --------
    >>> from neural_cherche import models, retrieve
    >>> from pprint import pprint
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> documents = [
    ...     {"id": 0, "document": "Food"},
    ...     {"id": 1, "document": "Sports"},
    ...     {"id": 2, "document": "Cinema"},
    ... ]

    >>> queries = ["Food", "Sports", "Cinema"]

    >>> model = models.Splade(
    ...     model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
    ...     device="mps",
    ... )

    >>> retriever = retrieve.Splade(
    ...     key="id",
    ...     on="document",
    ...     model=model
    ... )

    >>> documents_embeddings = retriever.encode_documents(
    ...     documents=documents,
    ...     batch_size=32,
    ... )

    >>> queries_embeddings = retriever.encode_queries(
    ...     queries=queries,
    ...     batch_size=32,
    ... )

    >>> retriever = retriever.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> scores = retriever(
    ...     queries_embeddings=queries_embeddings,
    ...     k=3,
    ... )

    >>> pprint(scores)
    [[{'id': 0, 'similarity': 19.365147},
      {'id': 1, 'similarity': 4.440977},
      {'id': 2, 'similarity': 3.9301493}],
     [{'id': 1, 'similarity': 18.055237},
      {'id': 0, 'similarity': 4.440977},
      {'id': 2, 'similarity': 3.6559556}],
     [{'id': 2, 'similarity': 16.21696},
      {'id': 0, 'similarity': 3.9301493},
      {'id': 1, 'similarity': 3.6559556}]]

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        model: models.Splade,
        tokenizer_parallelism: str = "false",
    ) -> None:
        super().__init__(
            key=key,
            on=on,
        )

        self.model = model
        os.environ["TOKENIZERS_PARALLELISM"] = tokenizer_parallelism

    def encode_documents(
        self,
        documents: list[dict],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        query_mode: bool = False,
        desc: str = "documents embeddings",
        **kwargs,
    ) -> dict[str, csr_matrix]:
        """Encode queries into sparse matrix.

        Parameters
        ----------
        documents
            Documents to encode.
        batch_size
            Batch size.
        tqdm_bar
            Whether to show tqdm bar.

        """
        documents_embeddings = {}

        for batch_documents in utils.batchify(
            documents,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            desc=f"{self.__class__.__name__} {desc}",
        ):
            embeddings = self.model.encode(
                [
                    " ".join([doc.get(field, "") for field in self.on])
                    for doc in batch_documents
                ],
                query_mode=query_mode,
                **kwargs,
            )

            sparse_activations = csr_matrix(
                embeddings["sparse_activations"].detach().cpu()
            )
            for document, sparse_activation in zip(batch_documents, sparse_activations):
                documents_embeddings[document[self.key]] = sparse_activation

        return documents_embeddings

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        query_mode: bool = True,
        desc: str = "queries embeddings",
        warn_duplicates: bool = True,
        **kwargs,
    ) -> dict[str, csr_matrix]:
        """Encode queries into sparse matrix.

        Parameters
        ----------
        documents
            Documents to encode.
        batch_size
            Batch size.
        tqdm_bar
            Whether to show tqdm bar.

        """
        queries_embeddings = {}

        for batch_queries in utils.batchify(
            queries,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            desc=f"{self.__class__.__name__}  {desc}",
        ):
            embeddings = self.model.encode(
                batch_queries,
                query_mode=query_mode,
                **kwargs,
            )

            sparse_activations = csr_matrix(
                embeddings["sparse_activations"].detach().cpu()
            )
            for query, sparse_activation in zip(batch_queries, sparse_activations):
                queries_embeddings[query] = sparse_activation

        if len(embeddings) != len(queries) and warn_duplicates:
            utils.duplicates_queries_warning()

        return queries_embeddings
