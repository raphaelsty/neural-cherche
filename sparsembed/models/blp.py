import json
import os

import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from ..utils import batchify

__all__ = ["BLP"]


class BLP(torch.nn.Module):
    """Inductive entity representation model.

    Parameters
    ----------
    model_name_or_path
        Path to the model or the model name. It should be a SentenceTransformer model.
    relations
        List of relations.
    gamma
        Margin.
    device
        Device to use for the model. CPU or CUDA.
    kwargs
        Additional parameters to the SentenceTransformer model.

    Example
    -------
    >>> from sparsembed import models

    >>> model = models.BLP(
    ...     model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    ...     device="mps",
    ...     relations=["author", "genre"]
    ... )

    >>> embeddings_entities = model.encode_entities(
    ...     ["pop", "rock"],
    ...     batch_size=2,
    ... )

    >>> embeddings_entities.shape
    torch.Size([2, 768])

    >>> embeddings_relations = model.encode_relations(
    ...     relations=["author", "genre"]
    ... )

    >>> embeddings_relations.shape
    torch.Size([2, 768])

    >>> triples = [
    ...     ["pop", "author", "Michael Jackson"],
    ...     ["rock", "author", "The Beatles"],
    ...     ["pop", "genre", "pop"],
    ...     ["rock", "genre", "rock"],
    ... ]

    >>> scores = model.scores_triples(
    ...     triples=triples,
    ...     batch_size=2,
    ... )

    >>> scores.shape
    torch.Size([4])

    >>> model = model.save_pretrained(path="checkpoint")

    >>> model = models.BLP(model_name_or_path="checkpoint")

    >>> scores = model.scores_triples(
    ...     triples=triples,
    ...     batch_size=2,
    ... )

    >>> scores.shape
    torch.Size([4])

    """

    def __init__(
        self,
        model_name_or_path: str,
        relations: list[str] = None,
        margin: float = 9.0,
        device: str = None,
        **kwargs,
    ) -> None:
        """Initialize the model."""
        super(BLP, self).__init__()

        self.model = SentenceTransformer(
            model_name_or_path,
            cache_folder=".",
            device=device,
            **kwargs,
        )

        model_folder = model_name_or_path.replace("/", "_")

        if device is not None:
            self.device = device

        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # SentenceTransformer does import every files.
        if os.path.exists(
            os.path.join(model_folder, "relations.json")
        ) and os.path.exists(os.path.join(model_folder, "relations_embeddings.pt")):
            embeddings = torch.load(
                os.path.join(model_folder, "relations_embeddings.pt"),
                map_location=self.device,
            )

            with open(os.path.join(model_folder, "relations.json"), "r") as f:
                relations = json.load(f)

            with open(os.path.join(model_folder, "margin.json"), "r") as f:
                margin = json.load(f)["margin"]

        if relations is None:
            raise ValueError("Relations should be provided.")

        self.relations = {relation: index for index, relation in enumerate(relations)}

        for relation in self.relations:
            if not isinstance(relation, str):
                raise ValueError(
                    f"Relations should be a list of strings. {relation} is not a string."
                )

        self.embeddings = torch.nn.Embedding(
            len(self.relations), self.model.get_sentence_embedding_dimension()
        ).to(device)

        if os.path.exists(os.path.join(model_folder, "relations_embeddings.pt")):
            self.embeddings.load_state_dict(embeddings)

        self.margin = margin

    def encode_relations(self, relations: list[str]) -> torch.Tensor:
        """Encode relations.

        Parameters
        ----------
        relations
            List of relations to encode.
        """
        if isinstance(relations, str):
            relations = [relations]

        relations = torch.tensor(
            data=[self.relations[relation] for relation in relations],
            device=self.device,
            dtype=torch.long,
        )

        return self.embeddings(relations)

    def encode_entities(
        self, entities: list[str], batch_size: int, **kwargs
    ) -> torch.Tensor:
        """Encode entities.

        Parameters
        ----------
        entities
            List of entities to encode.
        """
        if isinstance(entities, str):
            entities = [entities]

        return self.model.encode(
            sentences=entities,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            **kwargs,
        )

    def forward(
        self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor
    ) -> torch.Tensor:
        """BLP TransE scoring function.

        Parameters
        ----------
        heads
            Heads embeddings.
        relations
            Relations embeddings.
        tails
            Tails embeddings.
        """
        return self.margin - torch.norm((heads + relations) - tails, dim=1, p=2)

    def scores_triples(
        self,
        triples: list[list[str]],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Score triples.

        Parameters
        ----------
        triples
            List of triples to score.
        batch_size
            Batch size.
        tqdm_bar
            Show a progress bar.

        """
        heads, relations, tails = zip(*triples)
        with torch.no_grad():
            scores = self.scores(
                heads=heads,
                relations=relations,
                tails=tails,
                batch_size=batch_size,
                tqdm_bar=tqdm_bar,
                **kwargs,
            )
        return scores

    def scores(
        self,
        heads: list[str],
        relations: list[str],
        tails: list[str],
        batch_size: int = 32,
        tqdm_bar: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Score triples.

        Parameters
        ----------
        triples
            List of triples to score.
        batch_size
            Batch size.
        tqdm_bar
            Show a progress bar.

        """
        scores = []

        for batch_heads, batch_relations, batch_tails in zip(
            batchify(X=heads, batch_size=batch_size, tqdm_bar=tqdm_bar),
            batchify(X=relations, batch_size=batch_size, tqdm_bar=False),
            batchify(
                X=tails,
                batch_size=batch_size,
                tqdm_bar=False,
            ),
        ):
            scores.append(
                self(
                    heads=self.encode_entities(
                        batch_heads, batch_size=batch_size, **kwargs
                    ),
                    relations=self.encode_relations(batch_relations),
                    tails=self.encode_entities(
                        batch_tails, batch_size=batch_size, **kwargs
                    ),
                )
            )

        return torch.cat(scores, dim=-1)

    def save_pretrained(self, path: str) -> "BLP":
        """Save the model.

        Parameters
        ----------
        path
            Path to save the model.
        """
        self.model.save(path)
        torch.save(self.embeddings.state_dict(), path + "/relations_embeddings.pt")
        with open(os.path.join(path, "relations.json"), "w") as file:
            json.dump(self.relations, file)

        with open(os.path.join(path, "margin.json"), "w") as file:
            json.dump({"margin": self.margin}, file)
        return self
