import os

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .. import utils
from .base import Base

__all__ = ["ColBERT"]


class ColBERT(Base):
    """ColBERT model.

    Parameters
    ----------
    model_name_or_path
        Path to the model or the model name.
    embedding_size
        Size of the embeddings in output of ColBERT model.
    device
        Device to use for the model. CPU or CUDA.
    kwargs
        Additional parameters to the SentenceTransformer model.

    Example
    -------
    >>> from sparsembed import models
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> encoder = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    ...     embedding_size=64,
    ...     device="mps",
    ... )

    >>> embeddings = encoder(texts=["Sports Music Sports", "Music Sports Music"])
    >>> embeddings.shape
    torch.Size([2, 3, 64])

    >>> scores = encoder.scores(
    ...    queries=["football football football football football", "rugby rugby rugby"],
    ...    documents=["football", "rugby"],
    ... )

    >>> scores
    tensor([4.3269, 3.9620], device='mps:0', grad_fn=<CatBackward0>)

    >>> _ = encoder.save_pretrained("checkpoint")
    """

    def __init__(
        self,
        model_name_or_path: str,
        embedding_size: int = 64,
        device: str = None,
        **kwargs,
    ) -> None:
        """Initialize the model."""
        super(ColBERT, self).__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            extra_files_to_load=["linear.pt"],
            **kwargs,
        )

        self.embedding_size = embedding_size

        if os.path.exists(os.path.join(self.model_folder, "linear.pt")):
            linear = torch.load(
                os.path.join(self.model_folder, "linear.pt"), map_location=self.device
            )
            self.embedding_size = linear["weight"].shape[0]
            in_features = linear["weight"].shape[1]
        else:
            with torch.no_grad():
                _, embeddings = self._encode(texts=["test"])
                in_features = embeddings.shape[2]

        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=self.embedding_size, bias=False
        ).to(self.device)

        if os.path.exists(os.path.join(self.model_folder, "linear.pt")):
            self.linear.load_state_dict(linear)

    def encode(
        self,
        texts: list[str],
        truncation: bool = True,
        padding: bool = True,
        add_special_tokens: bool = False,
        max_length: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        """Encode documents

        Parameters
        ----------
        texts
            List of sentences to encode.
        truncation
            Truncate the inputs.
        padding
            Pad the inputs.
        add_special_tokens
            Add special tokens.
        max_length
            Maximum length of the inputs.
        """
        with torch.no_grad():
            embeddings = self(
                texts=texts,
                truncation=truncation,
                padding=padding,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                **kwargs,
            )
        return embeddings

    def forward(
        self,
        texts: list[str],
        truncation: bool = True,
        padding: bool = True,
        add_special_tokens: bool = False,
        max_length: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        """Pytorch forward method.

        Parameters
        ----------
        texts
            List of sentences to encode.
        truncation
            Truncate the inputs.
        padding
            Pad the inputs.
        add_special_tokens
            Add special tokens.
        max_length
            Maximum length of the inputs.
        """
        kwargs = {
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
            "add_special_tokens": add_special_tokens,
            **kwargs,
        }
        _, embeddings = self._encode(texts=texts, **kwargs)
        return self.linear(embeddings)

    def scores(
        self,
        queries: list[str],
        documents: list[str],
        batch_size: int = 2,
        truncation: bool = True,
        padding: bool = True,
        add_special_tokens: bool = False,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Score queries and documents.

        Parameters
        ----------
        queries
            List of queries.
        documents
            List of documents.
        batch_size
            Batch size.
        truncation
            Truncate the inputs.
        padding
            Pad the inputs.
        add_special_tokens
            Add special tokens.
        tqdm_bar
            Show tqdm bar.
        """
        list_scores = []

        for batch_queries, batch_documents in zip(
            utils.batchify(
                X=queries,
                batch_size=batch_size,
                desc="Computing scores.",
                tqdm_bar=tqdm_bar,
            ),
            utils.batchify(X=documents, batch_size=batch_size, tqdm_bar=False),
        ):
            queries_embeddings = self(
                texts=batch_queries,
                truncation=truncation,
                padding=padding,
                add_special_tokens=add_special_tokens,
                **kwargs,
            )

            documents_embeddings = self(
                texts=batch_documents,
                truncation=truncation,
                padding=padding,
                add_special_tokens=add_special_tokens,
                **kwargs,
            )

            late_interactions = torch.einsum(
                "bsh,bth->bst", queries_embeddings, documents_embeddings
            )

            late_interactions = torch.max(late_interactions, axis=2).values.sum(axis=1)

            list_scores.append(late_interactions)

        return torch.cat(list_scores, dim=0)

    def save_pretrained(self, path: str) -> "ColBERT":
        """Save model the model.

        Parameters
        ----------
        path
            Path to save the model.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        return self
