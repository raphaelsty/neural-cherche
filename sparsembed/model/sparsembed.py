import os

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .. import utils

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
    k_tokens
        Number of activated terms to retrieve.
    embedding_size
        Size of the embeddings in output of SparsEmbed model.

    Example
    -------
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import model
    >>> from pprint import pprint as print
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> device = "mps"

    >>> model = model.SparsEmbed(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ...     device=device,
    ...     embedding_size=32,
    ... )

    >>> queries_embeddings = model.encode(
    ...     ["Sports", "Music"],
    ...     k_tokens=96,
    ... )

    >>> documents_embeddings = model.encode(
    ...    ["Music is great.", "Sports is great."],
    ...     k_tokens=96,
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
    torch.Size([2, 96, 32])

    >>> model.scores(
    ...     queries=["Sports", "Music"],
    ...     documents=["Sports is great.", "Music is great."],
    ...     batch_size=1,
    ... )
    tensor([78.8720, 24.5763], device='mps:0')

    >>> _ = model.save_pretrained("checkpoint")

    >>> from sparsembed import model

    >>> model = model.SparsEmbed(
    ...     model_name_or_path="checkpoint",
    ...     device=device,
    ... )

    >>> queries_embeddings = model.encode(
    ...     ["Sports", "Music"],
    ...     k_tokens=96,
    ... )

    >>> queries_embeddings["embeddings"].shape
    torch.Size([2, 96, 32])

    References
    ----------
    1. [SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval](https://dl.acm.org/doi/pdf/10.1145/3539618.3592065)

    """

    def __init__(
        self,
        model_name_or_path: str = None,
        tokenizer: AutoTokenizer = None,
        model: AutoModelForMaskedLM = None,
        embedding_size: int = 64,
        device: str = None,
    ) -> None:
        super(SparsEmbed, self).__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        self.embedding_size = embedding_size

        self.softmax = torch.nn.Softmax(dim=2).to(self.device)

        if model_name_or_path is not None:
            linear = torch.load(os.path.join(model_name_or_path, "linear.pt"))
            self.embedding_size = linear["weight"].shape[0]
            in_features = linear["weight"].shape[1]
        else:
            with torch.no_grad():
                _, embeddings = self._encode(texts=["test"])
                in_features = embeddings.shape[2]

        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=self.embedding_size, bias=False
        ).to(self.device)

        if model_name_or_path is not None:
            self.linear.load_state_dict(linear)

    def forward(
        self,
        texts: list[str],
        k_tokens: int = 96,
        truncation: bool = True,
        padding: bool = True,
        max_length: int = 256,
        **kwargs,
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
            k_tokens=k_tokens,
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

    def save_pretrained(self, path: str):
        """Save model the model."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.linear.state_dict(), os.path.join(path, "linear.pt"))
        return self

    def scores(
        self,
        queries: list[str],
        documents: list[str],
        k_tokens: int = 96,
        batch_size: int = 32,
        truncation: bool = True,
        padding: bool = True,
        max_length: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        """Compute similarity scores between queries and documents."""
        dense_scores = []

        for batch_queries, batch_documents in zip(
            utils.batchify(X=queries, batch_size=batch_size, desc="Computing scores."),
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

            dense_scores.append(
                utils.pairs_dense_scores(
                    queries_activations=queries_embeddings["activations"],
                    documents_activations=documents_embeddings["activations"],
                    queries_embeddings=queries_embeddings["embeddings"],
                    documents_embeddings=documents_embeddings["embeddings"],
                )
            )

        return torch.cat(dense_scores, dim=0)
