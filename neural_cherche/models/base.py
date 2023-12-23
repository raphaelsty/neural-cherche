import json
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
    extra_files_to_load
        List of extra files to load.
    accelerate
        Use HuggingFace Accelerate.
    kwargs
        Additional parameters to the model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = None,
        extra_files_to_load: list[str] = [],
        accelerate: bool = False,
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

        self.accelerate = accelerate

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

        if os.path.exists(model_name_or_path):
            # Local checkpoint
            self.model_folder = model_name_or_path
        else:
            # HuggingFace checkpoint
            model_folder = os.path.join(
                f"models--{model_name_or_path}".replace("/", "--"), "snapshots"
            )
            snapshot = os.listdir(model_folder)[-1]
            self.model_folder = os.path.join(model_folder, snapshot)

        self.query_pad_token = self.tokenizer.mask_token
        self.original_pad_token = self.tokenizer.pad_token

    def _encode_accelerate(self, texts: list[str], **kwargs) -> tuple[torch.Tensor]:
        """Encode sentences with multiples gpus.

        Parameters
        ----------
        texts
            List of sentences to encode.

        References
        ----------
        [Accelerate issue.](https://github.com/huggingface/accelerate/issues/97)
        """
        encoded_input = self.tokenizer(texts, return_tensors="pt", **kwargs).to(
            self.device
        )

        position_ids = (
            torch.arange(0, encoded_input["input_ids"].size(1))
            .expand((len(texts), -1))
            .to(self.device)
        )

        output = self.model(**encoded_input, position_ids=position_ids)
        return output.logits, output.hidden_states[-1]

    def _encode(self, texts: list[str], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode sentences.

        Parameters
        ----------
        texts
            List of sentences to encode.
        """
        if self.accelerate:
            return self._encode_accelerate(texts, **kwargs)

        encoded_input = self.tokenizer.batch_encode_plus(
            texts, return_tensors="pt", **kwargs
        )

        if self.device != "cpu":
            encoded_input = {
                key: value.to(self.device) for key, value in encoded_input.items()
            }

        output = self.model(**encoded_input)
        return output.logits, output.hidden_states[-1]

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

    def save_tokenizer_accelerate(self, path: str) -> None:
        """Save tokenizer when using accelerate."""
        tokenizer_config = {
            k: v for k, v in self.tokenizer.__dict__.items() if k != "device"
        }
        tokenizer_config_file = os.path.join(path, "tokenizer_config.json")
        with open(tokenizer_config_file, "w", encoding="utf-8") as file:
            json.dump(tokenizer_config, file, ensure_ascii=False, indent=4)

        # dump vocab
        self.tokenizer.save_vocabulary(path)

        # save special tokens
        special_tokens_file = os.path.join(path, "special_tokens_map.json")
        with open(special_tokens_file, "w", encoding="utf-8") as file:
            json.dump(
                self.tokenizer.special_tokens_map,
                file,
                ensure_ascii=False,
                indent=4,
            )
