import json
import os
import string

import torch

from .. import utils
from .base import Base

__all__ = ["Splade"]


class Splade(Base):
    """SpladeV1 model.

    Parameters
    ----------
    tokenizer
        HuggingFace Tokenizer.
    model
        HuggingFace AutoModelForMaskedLM.
    kwargs
        Additional parameters to the SentenceTransformer model.

    Examples
    --------
    >>> from neural_cherche import models
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.Splade(
    ...     model_name_or_path="distilbert-base-uncased",
    ...     device="mps",
    ... )

    >>> queries_activations = model.encode(
    ...     ["Sports", "Music"],
    ... )

    >>> documents_activations = model.encode(
    ...    ["Music is great.", "Sports is great."],
    ...    query_mode=False,
    ... )

    >>> queries_activations["sparse_activations"].shape
    torch.Size([2, 30522])

    >>> model.scores(
    ...     queries=["Sports", "Music"],
    ...     documents=["Sports is great.", "Music is great."],
    ...     batch_size=1
    ... )
    tensor([318.1384, 271.8006], device='mps:0')

    >>> _ = model.save_pretrained("checkpoint")

    >>> model = models.Splade(
    ...     model_name_or_path="checkpoint",
    ...     device="mps",
    ... )

    >>> model.scores(
    ...     queries=["Sports", "Music"],
    ...     documents=["Sports is great.", "Music is great."],
    ...     batch_size=1
    ... )
    tensor([318.1384, 271.8006], device='mps:0')

    References
    ----------
    1. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)

    """

    def __init__(
        self,
        model_name_or_path: str = None,
        device: str = None,
        max_length_query: int = 128,
        max_length_document: int = 256,
        extra_files_to_load: list[str] = ["metadata.json"],
        accelerate: bool = False,
        query_prefix: str = "[Q] ",
        document_prefix: str = "[D] ",
        **kwargs,
    ) -> None:
        super(Splade, self).__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            extra_files_to_load=extra_files_to_load,
            accelerate=accelerate,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            **kwargs,
        )

        self.relu = torch.nn.ReLU().to(self.device)

        if os.path.exists(os.path.join(self.model_folder, "metadata.json")):
            with open(os.path.join(self.model_folder, "metadata.json"), "r") as file:
                metadata = json.load(file)

            max_length_query = metadata["max_length_query"]
            max_length_document = metadata["max_length_document"]
            self.query_prefix = metadata.get("query_prefix", self.query_prefix)
            self.document_prefix = metadata.get("document_prefix", self.document_prefix)

        self.max_length_query = max_length_query
        self.max_length_document = max_length_document

    def encode(
        self,
        texts: list[str],
        query_mode: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Encode documents

        Parameters
        ----------
        texts
            List of documents to encode.
        truncation
            Whether to truncate the documents.
        padding
            Whether to pad the documents.
        max_length
            Maximum length of the documents.
        """
        with torch.no_grad():
            return self(
                texts=texts,
                query_mode=query_mode,
                **kwargs,
            )

    def decode(
        self,
        sparse_activations: torch.Tensor,
        clean_up_tokenization_spaces: bool = False,
        skip_special_tokens: bool = True,
        k_tokens: int = 96,
    ) -> list[str]:
        """Decode activated tokens ids where activated value > 0.

        Parameters
        ----------
        sparse_activations
            Activated tokens.
        clean_up_tokenization_spaces
            Whether to clean up the tokenization spaces.
        skip_special_tokens
            Whether to skip special tokens.
        k_tokens
            Number of tokens to keep.
        """
        activations = self._filter_activations(
            sparse_activations=sparse_activations, k_tokens=k_tokens
        )

        # Decode
        return [
            " ".join(
                activation.translate(str.maketrans("", "", string.punctuation)).split()
            )
            for activation in self.tokenizer.batch_decode(
                activations,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                skip_special_tokens=skip_special_tokens,
            )
        ]

    def forward(
        self,
        texts: list[str],
        query_mode: bool,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Pytorch forward method.

        Parameters
        ----------
        texts
            List of documents to encode.
        query_mode
            Whether to encode queries or documents.
        """
        prefix = self.query_prefix if query_mode else self.document_prefix

        texts = [prefix + text for text in texts]

        self.tokenizer.pad_token = (
            self.query_pad_token if query_mode else self.original_pad_token
        )

        k_tokens = self.max_length_query if query_mode else self.max_length_document

        logits, _ = self._encode(
            texts=texts,
            truncation=True,
            padding="max_length",
            max_length=k_tokens,
            add_special_tokens=True,
            **kwargs,
        )

        activations = self._get_activation(logits=logits)

        activations = self._update_activations(
            **activations,
            k_tokens=k_tokens,
        )

        return {"sparse_activations": activations["sparse_activations"]}

    def save_pretrained(
        self,
        path: str,
    ):
        """Save model the model.

        Parameters
        ----------
        path
            Path to save the model.

        """
        self.model.save_pretrained(path)
        self.tokenizer.pad_token = self.original_pad_token

        if self.accelerate:
            self.save_tokenizer_accelerate(path)
        else:
            self.tokenizer.save_pretrained(path)

        with open(os.path.join(path, "metadata.json"), "w") as file:
            json.dump(
                fp=file,
                obj={
                    "max_length_query": self.max_length_query,
                    "max_length_document": self.max_length_document,
                    "query_prefix": self.query_prefix,
                    "document_prefix": self.document_prefix,
                },
                indent=4,
            )

        return self

    def scores(
        self,
        queries: list[str],
        documents: list[str],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Compute similarity scores between queries and documents.

        Parameters
        ----------
        queries
            List of queries.
        documents
            List of documents.
        batch_size
            Batch size.
        tqdm_bar
            Show a progress bar.
        """
        sparse_scores = []

        for batch_queries, batch_documents in zip(
            utils.batchify(
                X=queries,
                batch_size=batch_size,
                desc="Computing scores.",
                tqdm_bar=tqdm_bar,
            ),
            utils.batchify(X=documents, batch_size=batch_size, tqdm_bar=False),
        ):
            queries_embeddings = self.encode(
                batch_queries,
                query_mode=True,
                **kwargs,
            )

            documents_embeddings = self.encode(
                batch_documents,
                query_mode=False,
                **kwargs,
            )

            sparse_scores.append(
                torch.sum(
                    queries_embeddings["sparse_activations"]
                    * documents_embeddings["sparse_activations"],
                    axis=1,
                )
            )

        return torch.cat(sparse_scores, dim=0)

    def _get_activation(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns activated tokens."""
        return {"sparse_activations": torch.amax(torch.log1p(self.relu(logits)), dim=1)}

    def _filter_activations(
        self, sparse_activations: torch.Tensor, k_tokens: int
    ) -> list[torch.Tensor]:
        """Among the set of activations, select the ones with a score > 0."""
        scores, activations = torch.topk(input=sparse_activations, k=k_tokens, dim=-1)
        return [
            torch.index_select(
                activation, dim=-1, index=torch.nonzero(score, as_tuple=True)[0]
            )
            for score, activation in zip(scores, activations)
        ]

    def _update_activations(
        self, sparse_activations: torch.Tensor, k_tokens: int
    ) -> torch.Tensor:
        """Returns activated tokens."""
        activations = torch.topk(input=sparse_activations, k=k_tokens, dim=1).indices
        zero_tensor = torch.zeros_like(sparse_activations, dtype=int)
        updated_sparse_activations = sparse_activations * zero_tensor.scatter(
            dim=1, index=activations.long(), value=1
        )

        return {
            "activations": activations,
            "sparse_activations": updated_sparse_activations,
        }
