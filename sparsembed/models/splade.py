import string

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

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

    Example
    -------
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import models

    >>> device = "mps"

    >>> model = models.Splade(
    ...     model_name_or_path="distilbert-base-uncased",
    ...     device=device,
    ... )

    >>> queries_activations = model.encode(
    ...     ["Sports", "Music"],
    ... )

    >>> documents_activations = model.encode(
    ...    ["Music is great.", "Sports is great."],
    ... )

    >>> queries_activations["sparse_activations"].shape
    torch.Size([2, 30522])

    >>> model.scores(
    ...     queries=["Sports", "Music"],
    ...     documents=["Sports is great.", "Music is great."],
    ...     batch_size=1
    ... )
    tensor([301.4348, 214.5453], device='mps:0')

    >>> _ = model.save_pretrained("checkpoint")

    >>> model = models.Splade(
    ...     model_name_or_path="checkpoint",
    ...     device=device,
    ... )

    >>> queries_activations["sparse_activations"].shape
    torch.Size([2, 30522])

    References
    ----------
    1. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)

    """

    def __init__(
        self,
        model_name_or_path: str = None,
        device: str = None,
        **kwargs,
    ) -> None:
        super(Splade, self).__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            **kwargs,
        )

        self.relu = torch.nn.ReLU().to(self.device)

    def encode(
        self,
        texts: list[str],
        truncation: bool = True,
        padding: bool = True,
        max_length: int = 256,
        k_tokens: int = 256,
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
        k_tokens
            Number of tokens to keep.
        """
        with torch.no_grad():
            return self(
                texts=texts,
                k_tokens=k_tokens,
                truncation=truncation,
                padding=padding,
                max_length=max_length,
                **kwargs,
            )

    def decode(
        self,
        sparse_activations: torch.Tensor,
        clean_up_tokenization_spaces: bool = False,
        skip_special_tokens: bool = True,
        k_tokens: int = 96,
        **kwargs,
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
        truncation: bool = True,
        padding: bool = True,
        k_tokens: int = None,
        max_length: int = 256,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Pytorch forward method.

        Parameters
        ----------
        texts
            List of documents to encode.
        truncation
            Whether to truncate the documents.
        padding
            Whether to pad the documents.
        k_tokens
            Number of tokens to keep.
        max_length
            Maximum length of the documents.
        """
        kwargs = {
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
            **kwargs,
        }

        logits, _ = self._encode(texts=texts, **kwargs)

        activations = self._get_activation(logits=logits)

        if k_tokens is not None:
            activations = self._update_activations(
                **activations,
                k_tokens=k_tokens,
            )

        return {"sparse_activations": activations["sparse_activations"]}

    def save_pretrained(self, path: str):
        """Save model the model."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        return self

    def scores(
        self,
        queries: list[str],
        documents: list[str],
        batch_size: int = 32,
        k_tokens: int = 256,
        truncation: bool = True,
        padding: bool = True,
        max_length: int = 256,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Compute similarity scores between queries and documents."""
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
                k_tokens=k_tokens,
                truncation=truncation,
                padding=padding,
                max_length=max_length,
                **kwargs,
            )

            documents_embeddings = self.encode(
                batch_documents,
                k_tokens=k_tokens,
                truncation=truncation,
                padding=padding,
                max_length=max_length,
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
        return {
            "sparse_activations": torch.amax(torch.log1p(self.relu(logits)), dim=1),
        }

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

        # Set value of max sparse_activations which are not in top k to 0.
        sparse_activations = sparse_activations * torch.zeros(
            (sparse_activations.shape[0], sparse_activations.shape[1]), dtype=int
        ).to(self.device).scatter_(dim=1, index=activations.long(), value=1)

        return {
            "activations": activations,
            "sparse_activations": sparse_activations,
        }
