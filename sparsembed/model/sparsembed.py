import string

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

__all__ = ["SparseEmbed"]

from .splade import Splade


class SparsEmbed(Splade):
    """SparsEmbed model.

    Parameters
    ----------
    tokenizer
        HuggingFace Tokenizer.
    model
        HuggingFace AutoModelForMaskedLM.
    k_query
        Number of activated terms to retrieve for queries.
    k_documents
        Number of activated terms to retrieve for documents.
    embedding_size
        Size of the embeddings in output of SparsEmbed model.

    Example
    -------
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import model
    >>> from pprint import pprint as print

    >>> device = "mps"

    >>> model = model.SparsEmbed(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ...     k_tokens=96,
    ...     device=device,
    ... )

    >>> queries_embeddings = model.encode(
    ...     ["Sports", "Music"],
    ... )

    >>> documents_embeddings = model.encode(
    ...    ["Music is great.", "Sports is great."],
    ... )

    >>> query_expanded = model.decode(
    ...     sparse_activations=queries_embeddings["sparse_activations"]
    ... )

    >>> documents_expanded = model.decode(
    ...     sparse_activations=documents_embeddings["sparse_activations"]
    ... )

    >>> queries_embeddings["activations"].shape
    torch.Size([2, 96])

    >>> queries_embeddings["sparse_activations"].shape
    torch.Size([2, 30522])

    >>> queries_embeddings["embeddings"].shape
    torch.Size([2, 96, 64])

    References
    ----------
    1. [SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval](https://dl.acm.org/doi/pdf/10.1145/3539618.3592065)

    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForMaskedLM,
        k_tokens: int = 96,
        embedding_size: int = 64,
        device: str = None,
    ) -> None:
        super(SparsEmbed, self).__init__(
            tokenizer=tokenizer, model=model, device=device
        )

        self.k_tokens = k_tokens
        self.embedding_size = embedding_size

        self.softmax = torch.nn.Softmax(dim=2).to(self.device)

        # Input embedding size:
        with torch.no_grad():
            _, embeddings = self._encode(texts=["test"])
            in_features = embeddings.shape[2]

        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=embedding_size, bias=False
        ).to(self.device)

    def encode(
        self,
        texts: list[str],
        truncation: bool = True,
        padding: bool = True,
        max_length: int = 256,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """Encode documents"""
        with torch.no_grad():
            return self(
                texts=texts,
                truncation=truncation,
                padding=padding,
                max_length=max_length,
                **kwargs,
            )

    def forward(
        self,
        texts: list[str],
        truncation: bool = True,
        padding: bool = True,
        max_length: int = 256,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """Pytorch forward method."""
        kwargs = {
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
            **kwargs,
        }

        logits, embeddings = self._encode(texts=texts, **kwargs)

        activations = self._update_activations(
            **self._get_activation(logits=logits),
            k_tokens=self.k_tokens,
        )

        attention = self._get_attention(
            logits=logits,
            activations=activations["activations"],
        )

        embeddings = torch.bmm(
            attention,
            embeddings,
        )

        return {
            "embeddings": self.relu(self.linear(embeddings)),
            "sparse_activations": activations["sparse_activations"],
            "activations": activations["activations"],
        }

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

    def _get_attention(
        self, logits: torch.Tensor, activations: torch.Tensor
    ) -> torch.Tensor:
        """Extract attention scores from MLM logits based on activated tokens."""
        attention = logits.gather(
            dim=2,
            index=torch.stack(
                [
                    torch.stack([token for _ in range(logits.shape[1])])
                    for token in activations
                ]
            ),
        )

        return self.softmax(attention.transpose(1, 2))
