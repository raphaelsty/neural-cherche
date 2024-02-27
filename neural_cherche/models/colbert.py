import json
import os

import torch

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

    Examples
    --------
    >>> from neural_cherche import models
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> queries = [" ".join(["Berlin"] * 600), "Paris", "London"]

    >>> documents = [
    ...     "Berlin is the capital of Germany",
    ...     "Paris is the capital of France and France is in Europe",
    ...     "London is the capital of England",
    ... ]

    >>> encoder = models.ColBERT(
    ...     model_name_or_path="raphaelsty/neural-cherche-colbert",
    ...     device="cpu",
    ... )

    >>> scores = encoder.scores(
    ...    queries=queries,
    ...    documents=documents,
    ...    batch_size=10,
    ... )

    >>> scores
    tensor([19.2999,  3.6684,  3.5832])

    >>> _ = encoder.save_pretrained("checkpoint")

    >>> encoder = models.ColBERT(
    ...     model_name_or_path="checkpoint",
    ...     device="cpu",
    ... )

    >>> scores = encoder.scores(
    ...    queries=queries,
    ...    documents=documents,
    ...    batch_size=1,
    ... )

    >>> scores
    tensor([19.2999,  3.6684,  3.5832])

    >>> embeddings = encoder(
    ...     texts=queries,
    ...     query_mode=True
    ... )

    >>> embeddings["embeddings"].shape
    torch.Size([3, 32, 128])

    >>> embeddings = encoder(
    ...     texts=queries,
    ...     query_mode=False
    ... )

    >>> embeddings["embeddings"].shape
    torch.Size([3, 256, 128])

    """

    def __init__(
        self,
        model_name_or_path: str,
        embedding_size: int = 128,
        device: str = None,
        max_length_query: int = 64,
        max_length_document: int = 256,
        query_prefix: str = "[Q] ",
        document_prefix: str = "[D] ",
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True,
        freeze_layers_except_last_n: int = None,
        **kwargs,
    ) -> None:
        """Initialize the model."""
        super(ColBERT, self).__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            extra_files_to_load=["linear.pt", "metadata.json"],
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            freeze_layers_except_last_n=freeze_layers_except_last_n,
            **kwargs,
        )

        self.max_length_query = max_length_query
        self.max_length_document = max_length_document
        self.embedding_size = embedding_size

        if os.path.exists(path=os.path.join(self.model_folder, "linear.pt")):
            linear = torch.load(
                os.path.join(self.model_folder, "linear.pt"), map_location=self.device
            )
            self.embedding_size = linear["weight"].shape[0]
            in_features = linear["weight"].shape[1]
        else:
            with torch.no_grad():
                _, embeddings, _ = self._encode(texts=["test"])
                in_features = embeddings.shape[2]

        self.linear = torch.nn.Linear(
            in_features=in_features,
            out_features=self.embedding_size,
            dtype=torch.float32,
            bias=False,
            device=self.device,
        )

        torch.nn.init.xavier_uniform_(tensor=self.linear.weight)

        if os.path.exists(path=os.path.join(self.model_folder, "metadata.json")):
            with open(
                file=os.path.join(self.model_folder, "metadata.json"), mode="r"
            ) as f:
                metadata = json.load(fp=f)
            self.max_length_document = metadata["max_length_document"]
            self.max_length_query = metadata["max_length_query"]
            self.query_prefix = metadata.get("query_prefix", self.query_prefix)
            self.document_prefix = metadata.get("document_prefix", self.document_prefix)
            self.padding = metadata.get("padding", self.padding)
            self.truncation = metadata.get("truncation", self.truncation)
            self.add_special_tokens = metadata.get(
                "add_special_tokens", self.add_special_tokens
            )

        if os.path.exists(path=os.path.join(self.model_folder, "linear.pt")):
            self.linear.load_state_dict(state_dict=linear)

        self.query_pad_token = self.tokenizer.mask_token
        self.original_pad_token = self.tokenizer.pad_token

    def encode(
        self,
        texts: list[str],
        truncation: bool = True,
        add_special_tokens: bool = False,
        query_mode: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Encode documents

        Parameters
        ----------
        texts
            List of sentences to encode.
        truncation
            Truncate the inputs.
        add_special_tokens
            Add special tokens.
        max_length
            Maximum length of the inputs.
        """
        with torch.no_grad():
            embeddings = self(
                texts=texts,
                query_mode=query_mode,
                **kwargs,
            )
        return embeddings

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
            List of sentences to encode.
        query_mode
            Wether to encode query or not.
        """
        prefix = self.query_prefix if query_mode else self.document_prefix
        texts = [prefix + text for text in texts]

        _, embeddings, attention_mask = self._encode(
            texts=texts,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length_query
            if query_mode
            else self.max_length_document,
            add_special_tokens=self.add_special_tokens,
            **kwargs,
        )

        return {
            "embeddings": torch.nn.functional.normalize(
                input=self.linear(embeddings * attention_mask), p=2, dim=2
            )
        }

    def scores(
        self,
        queries: list[str],
        documents: list[str],
        batch_size: int = 2,
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

            late_interactions = torch.einsum(
                "bsh,bth->bst",
                queries_embeddings["embeddings"],
                documents_embeddings["embeddings"],
            )

            late_interactions = torch.max(late_interactions, axis=2).values.sum(axis=1)

            list_scores.append(late_interactions)

        return torch.cat(tensors=list_scores, dim=0)

    def save_pretrained(self, path: str) -> "ColBERT":
        """Save model the model.

        Parameters
        ----------
        path
            Path to save the model.
        """
        self.model.save_pretrained(path)
        torch.save(self.linear.state_dict(), os.path.join(path, "linear.pt"))
        self.tokenizer.pad_token = self.original_pad_token
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(
                obj={
                    "max_length_query": self.max_length_query,
                    "max_length_document": self.max_length_document,
                    "query_prefix": self.query_prefix,
                    "document_prefix": self.document_prefix,
                    "padding": self.padding,
                    "truncation": self.truncation,
                    "add_special_tokens": self.add_special_tokens,
                },
                fp=f,
            )
        return self
