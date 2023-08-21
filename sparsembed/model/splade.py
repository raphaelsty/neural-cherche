import string

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

__all__ = ["Splade"]


class Splade(torch.nn.Module):
    """SpladeV1 model.

    Parameters
    ----------
    tokenizer
        HuggingFace Tokenizer.
    model
        HuggingFace AutoModelForMaskedLM.

    Example
    -------
    >>> from transformers import AutoModelForMaskedLM, AutoTokenizer
    >>> from sparsembed import model
    >>> from pprint import pprint as print

    >>> device = "mps"

    >>> model = model.Splade(
    ...     model=AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device),
    ...     tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    ...     device=device
    ... )

    >>> queries_activations = model.encode(
    ...     ["Sports", "Music"],
    ... )

    >>> documents_activations = model.encode(
    ...    ["Music is great.", "Sports is great."],
    ... )

    >>> queries_activations["sparse_activations"].shape
    torch.Size([2, 30522])

    References
    ----------
    1. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)

    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForMaskedLM,
        device: str = None,
    ) -> None:
        super(Splade, self).__init__()
        self.tokenizer = tokenizer
        self.model = model

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model.config.output_hidden_states = True
        self.relu = torch.nn.ReLU().to(self.device)

    def encode(
        self,
        texts: list[str],
        truncation: bool = True,
        padding: bool = True,
        max_length: int = 256,
        k_tokens: int = 256,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """Encode documents"""
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
    ) -> list[str]:
        """Decode activated tokens ids where activated value > 0."""
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
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """Pytorch forward method."""
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
        return output.logits, output.hidden_states[0]

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
