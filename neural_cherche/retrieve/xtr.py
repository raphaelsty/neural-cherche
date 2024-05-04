from collections import defaultdict

import torch
import tqdm

from .. import models, utils
from ..retrieve import ColBERT

__all__ = ["XTR"]


class XTR(ColBERT):
    """XTR retriever.

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
    >>> from neural_cherche import models, retrieve
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

    >>> retriever = retrieve.XTR(
    ...    key="id",
    ...    on=["document"],
    ...    model=encoder,
    ... )

    >>> documents_embeddings = retriever.encode_documents(
    ...     documents=documents,
    ...     batch_size=3,
    ... )

    >>> retriever = retriever.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> queries_embeddings = retriever.encode_queries(
    ...     queries=queries,
    ...     batch_size=3,
    ... )

    >>> scores = retriever(
    ...     queries_embeddings=queries_embeddings,
    ...     batch_size=3,
    ...     tqdm_bar=True,
    ...     k=3,
    ... )

    >>> pprint(scores)
    [[{'id': 0, 'similarity': 4.7243194580078125},
      {'id': 2, 'similarity': 2.403003692626953},
      {'id': 1, 'similarity': 2.286036252975464}],
     [{'id': 1, 'similarity': 4.792296886444092},
      {'id': 2, 'similarity': 2.6001152992248535},
      {'id': 0, 'similarity': 2.312016487121582}],
     [{'id': 2, 'similarity': 5.069696426391602},
      {'id': 1, 'similarity': 2.5587477684020996},
      {'id': 0, 'similarity': 2.4474282264709473}]]

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
        self.documents = []
        self.documents_embeddings = {}

    def __call__(
        self,
        queries_embeddings: dict[str, torch.Tensor],
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

        bar = (
            tqdm.tqdm(iterable=queries_embeddings.items(), position=0)
            if tqdm_bar
            else queries_embeddings.items()
        )

        for query, query_embedding in bar:
            query_scores = []

            embedding_query = torch.tensor(
                data=query_embedding,
                device=self.device,
                dtype=torch.float32,
            )

            for batch_query_documents in utils.batchify(
                X=self.documents,
                batch_size=batch_size,
                tqdm_bar=False,
            ):
                embeddings_batch_documents = torch.stack(
                    tensors=[
                        torch.tensor(
                            data=self.documents_embeddings[document[self.key]],
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

                query_scores.append(query_documents_scores)
            scores.append(self.xtr_score(torch.cat(tensors=query_scores), k))

        return scores

    def xtr_score(self, all_socres, k:int
        )-> list[list[dict]]:
        num_tokens = all_socres.shape[1]
        sum_tokens_queries = defaultdict(float)
        for token_id in range(num_tokens):
            # Iterate through tokens
            tensor = all_socres[:, token_id, :]
            # Flatten the tensor
            flattened_tensor = tensor.flatten()
            # Use topk to get the indices of the top k` elements across the entire tensor
            top_values, top_indices = flattened_tensor.topk(1000)
            # Convert the flattened indices to their original shape
            index_top_doc = top_indices // tensor.shape[1]  # index of the doc
            # index_top_token= top_indices % tensor.shape[1]# index of  token embding doc
            # exact index for doc and token embding doc for a token query
            # index_doc_docToken = torch.stack([top_indices // tensor.shape[1], top_indices % tensor.shape[1]],1)
            # Iterate through same doc index and update using sum
            for idx, i_doc in enumerate(index_top_doc):
                sum_tokens_queries[self.documents[i_doc.item()]["index"]] += top_values[
                    idx
                ]
        # make it in the same format of {self.key: key_, 'similarity': value_} and stop at top k
        socres = []
        for n, (key_, value_) in enumerate(sum_tokens_queries.items()):
            if n > k - 1:
                break
            socres.append({self.key: key_, "similarity": value_ / num_tokens})
        return socres
