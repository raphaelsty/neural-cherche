import json
import os

import torch

from .. import utils

__all__ = ["SparseEmbed"]

from .splade import Splade


class SparseEmbed(Splade):
    """SparseEmbed model.

    Parameters
    ----------
    model_name_or_path
        Path to the model or the model name. It should be a SentenceTransformer model.
    embedding_size
        Size of the embeddings in output of SparsEmbed model.
    kwargs
        Additional parameters to the pre-trained model.

    Examples
    --------
    >>> from neural_cherche import models
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> device = "mps"

    >>> model = models.SparseEmbed(
    ...     model_name_or_path="distilbert-base-uncased",
    ...     device=device,
    ... )

    >>> queries_embeddings = model.encode(
    ...     ["Sports", "Music"],
    ... )

    >>> queries_embeddings["activations"].shape
    torch.Size([2, 128])

    >>> queries_embeddings["sparse_activations"].shape
    torch.Size([2, 30522])

    >>> queries_embeddings["embeddings"].shape
    torch.Size([2, 128, 128])

    >>> documents_embeddings = model.encode(
    ...    ["Music is great.", "Sports is great."],
    ...    query_mode=False,
    ... )

    >>> documents_embeddings["activations"].shape
    torch.Size([2, 256])

    >>> documents_embeddings["sparse_activations"].shape
    torch.Size([2, 30522])

    >>> documents_embeddings["embeddings"].shape
    torch.Size([2, 256, 128])

    >>> model.scores(
    ...     queries=["Sports", "Music"],
    ...     documents=["Sports is great.", "Music is great."],
    ...     batch_size=1,
    ... )
    tensor([64.2330, 54.0180], device='mps:0')

    >>> _ = model.save_pretrained("checkpoint")

    >>> model = models.SparseEmbed(
    ...     model_name_or_path="checkpoint",
    ...     device="cpu",
    ... )

    >>> model.scores(
    ...     queries=["Sports", "Music"],
    ...     documents=["Sports is great.", "Music is great."],
    ...     batch_size=2,
    ... )
    tensor([64.2330, 54.0180])

    References
    ----------
    1. [SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval](https://dl.acm.org/doi/pdf/10.1145/3539618.3592065)

    """

    def __init__(
        self,
        model_name_or_path: str = None,
        embedding_size: int = 128,
        max_length_query: int = 128,
        max_length_document: int = 256,
        device: str = None,
        accelerate: bool = False,
        **kwargs,
    ) -> None:
        super(SparseEmbed, self).__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            extra_files_to_load=["linear.pt", "metadata.json"],
            accelerate=accelerate,
            **kwargs,
        )

        self.embedding_size = embedding_size

        self.softmax = torch.nn.Softmax(dim=2).to(self.device)

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
            in_features=in_features,
            out_features=self.embedding_size,
            bias=False,
            device=self.device,
        )

        if os.path.exists(os.path.join(self.model_folder, "linear.pt")):
            self.linear.load_state_dict(linear)

        if os.path.exists(os.path.join(self.model_folder, "metadata.json")):
            with open(os.path.join(self.model_folder, "metadata.json"), "r") as file:
                metadata = json.load(file)

            max_length_query = metadata["max_length_query"]
            max_length_document = metadata["max_length_document"]

        self.max_length_query = max_length_query
        self.max_length_document = max_length_document

    def forward(
        self,
        texts: list[str],
        query_mode: bool = True,
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
        suffix = "[Q] " if query_mode else "[D] "

        texts = [suffix + text for text in texts]

        self.tokenizer.pad_token = (
            self.query_pad_token if query_mode else self.original_pad_token
        )

        k_tokens = self.max_length_query if query_mode else self.max_length_document

        logits, embeddings = self._encode(
            texts=texts,
            truncation=True,
            padding="max_length",
            max_length=k_tokens,
            add_special_tokens=True,
            **kwargs,
        )

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

        return self.softmax(attention)

    def save_pretrained(
        self,
        path: str,
    ):
        """Save model the model."""
        self.model.save_pretrained(path)
        self.tokenizer.pad_token = self.original_pad_token

        if self.accelerate:
            self.save_tokenizer_accelerate(path)
        else:
            self.tokenizer.save_pretrained(path)
        torch.save(self.linear.state_dict(), os.path.join(path, "linear.pt"))
        with open(os.path.join(path, "metadata.json"), "w") as file:
            json.dump(
                fp=file,
                obj={
                    "max_length_query": self.max_length_query,
                    "max_length_document": self.max_length_document,
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
        """Compute similarity scores between queries and documents."""
        dense_scores = []

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
                texts=batch_queries,
                query_mode=True,
                **kwargs,
            )

            documents_embeddings = self.encode(
                texts=batch_documents,
                query_mode=False,
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
