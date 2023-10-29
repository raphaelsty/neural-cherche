import os
from abc import ABC, abstractmethod

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoTokenizer


class Base(ABC, torch.nn.Module):
    """Base class from which all models inherit.

    Parameters
    ----------
    model_name_or_path
        Path to the model or the model name.
    device
        Device to use for the model. CPU or CUDA.
    kwargs
        Additional parameters to the model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = None,
        extra_files_to_load: list[str] = [],
        **kwargs,
    ) -> None:
        """Initialize the model."""
        super(Base, self).__init__()

        if device is not None:
            self.device = device

        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        os.environ["TRANSFORMERS_CACHE"] = "."
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, cache_dir="./", **kwargs
        ).to(self.device)

        # Download linear layer if exists
        for file in extra_files_to_load:
            try:
                _ = hf_hub_download(model_name_or_path, filename=file, cache_dir=".")
            except:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, device=self.device, cache_dir="./", **kwargs
        )

        self.model.config.output_hidden_states = True
        model_folder = os.path.join(
            f"models--{model_name_or_path}".replace("/", "--"), "snapshots"
        )
        snapshot = os.listdir(model_folder)[-1]
        self.model_folder = os.path.join(model_folder, snapshot)

    def _encode(self, texts: list[str], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode sentences.

        Parameters
        ----------
        texts
            List of sentences to encode.
        """
        encoded_input = self.tokenizer.batch_encode_plus(
            texts, return_tensors="pt", **kwargs
        )

        if self.device != "cpu":
            encoded_input = {
                key: value.to(self.device) for key, value in encoded_input.items()
            }

        output = self.model(**encoded_input)
        return output.logits, output.hidden_states[0]

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Pytorch forward method."""
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        """Encode documents."""
        pass

    @abstractmethod
    def scores(self, *args, **kwars):
        """Compute scores."""
        pass

    @abstractmethod
    def save_pretrained(self, path: str):
        """Save model the model."""
        pass
