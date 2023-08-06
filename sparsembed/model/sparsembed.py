import string

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

__all__ = ["SparseEmbed"]


class SparsEmbed(torch.nn.Module):
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
    ...     device=device
    ... )

    >>> queries_embeddings = model.encode(
    ...     ["Sports", "Music"],
    ...     k=96
    ... )

    >>> documents_embeddings = model.encode(
    ...    ["Music is great.", "Sports is great."],
    ...     k=256
    ... )

    >>> query_expanded = model.decode(
    ...     sparse_activations=queries_embeddings["sparse_activations"], k=25
    ... )

    >>> print(query_expanded)
    ['sports the games s defaulted and mores of athletics sport hockey a '
     'basketball',
     'music the on s song of a more in and songs musical 2015 to']

    >>> documents_expanded = model.decode(
    ...     sparse_activations=documents_embeddings["sparse_activations"], k=25
    ... )

    >>> print(documents_expanded)
    ['is great music was good wonderful big beautiful huge has are and of fine on '
     'a',
     'is great sports good was big has are and wonderful sport huge nice of games '
     'a']

    >>> queries_embeddings["activations"].shape
    torch.Size([2, 96])

    References
    ----------
    1. [SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval](https://dl.acm.org/doi/pdf/10.1145/3539618.3592065)

    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForMaskedLM,
        k_query: int = 64,
        k_documents: int = 256,
        embedding_size: int = 256,
        device: str = None,
    ) -> None:
        super(SparsEmbed, self).__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.k_query = k_query
        self.k_documents = k_documents
        self.embedding_size = embedding_size

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model.config.output_hidden_states = True

        self.relu = torch.nn.ReLU().to(self.device)
        self.softmax = torch.nn.Softmax(dim=2).to(self.device)

        # Input embedding size:
        with torch.no_grad():
            _, embeddings = self._encode(texts=["test"])
            in_features = embeddings.shape[2]

        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=embedding_size
        ).to(self.device)

    def encode(
        self,
        texts: list[str],
        k: int,
        truncation: bool = True,
        padding: bool = True,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """Encode documents"""
        with torch.no_grad():
            return self(
                texts=texts, k=k, truncation=truncation, padding=padding, **kwargs
            )

    def decode(
        self,
        sparse_activations: torch.Tensor,
        clean_up_tokenization_spaces: bool = False,
        skip_special_tokens: bool = True,
        k: int = 128,
    ) -> list[str]:
        """Decode activated tokens ids where activated value > 0."""
        activations = self._filter_activations(
            sparse_activations=sparse_activations, k=k
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
        k: int,
        truncation: bool = True,
        padding: bool = True,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """Pytorch forward method."""
        kwargs = {"truncation": truncation, "padding": padding, **kwargs}

        logits, embeddings = self._encode(texts=texts, **kwargs)

        activations = self._get_activation(logits=logits, k=k)

        attention = self._get_attention(
            logits=logits,
            activation=activations["activations"],
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

    def _encode(self, texts: list[str], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode sentences."""
        encoded_input = self.tokenizer.batch_encode_plus(
            texts, return_tensors="pt", **kwargs
        )
        if self.device != "cpu":
            encoded_input = {
                key: value.to(self.device) for key, value in encoded_input.items()
            }
        output = self.model(**encoded_input)
        return output.logits, output.hidden_states[-1]

    def _get_activation(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Returns activated tokens."""
        max_pooling = torch.amax(torch.log(1 + self.relu(logits)), dim=1)
        activations = torch.topk(input=max_pooling, k=k, dim=-1).indices

        # Set value of max pooling which are not in top k to 0.
        sparse_activations = max_pooling * torch.zeros(
            (max_pooling.shape[0], max_pooling.shape[1]), dtype=int
        ).to(self.device).scatter_(dim=1, index=activations.long(), value=1)

        return {
            "activations": activations,
            "sparse_activations": sparse_activations,
        }

    def _get_attention(
        self, logits: torch.Tensor, activation: torch.Tensor
    ) -> torch.Tensor:
        """Extract attention scores from MLM logits based on activated tokens."""
        attention = logits.gather(
            dim=2,
            index=torch.stack(
                [
                    torch.stack([token for _ in range(logits.shape[1])])
                    for token in activation
                ]
            ),
        )

        return self.softmax(attention.transpose(1, 2))

    def _filter_activations(
        self, sparse_activations: torch.Tensor, k: int
    ) -> list[torch.Tensor]:
        """Among the set of activations, select the ones with a score > 0."""
        scores, activations = torch.topk(input=sparse_activations, k=k, dim=-1)
        return [
            torch.index_select(
                activation, dim=-1, index=torch.nonzero(score, as_tuple=True)[0]
            )
            for score, activation in zip(scores, activations)
        ]
