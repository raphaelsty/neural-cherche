# Multi-GPU (Partial)

Neural-Cherche is working towards being fully compatible with multiples GPUs training using [Accelerator](https://huggingface.co/docs/accelerate/package_reference/accelerator). At the moment, there is partial compatibility, and we can train every models of neural-cherche using GPUs in most circumstances, although it's not yet fully supported. Here is a tutorial.

```python
import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader

from neural_cherche import models, train

if __name__ == "__main__":
    # We will need to wrap your training loop in a function to avoid multiprocessing issues.
    accelerator = Accelerator()
    save_each_epoch = True

    model = models.SparseEmbed(
        model_name_or_path="distilbert-base-uncased",
        accelerate=True,
        device=accelerator.device,
    ).to(accelerator.device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # Dataset creation using HuggingFace Datasets library.
    dataset = Dataset.from_dict(
        {
            "anchors": ["anchor 1", "anchor 2", "anchor 3", "anchor 4"],
            "positives": ["positive 1", "positive 2", "positive 3", "positive 4"],
            "negatives": ["negative 1", "negative 2", "negative 3", "negative 4"],
        }
    )

    # Convert your dataset to a DataLoader.
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Wrap model, optimizer, and dataloader in accelerator.
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    for epoch in range(2):
        for batch in enumerate(data_loader):
            # Batch is a triple like (anchors, positives, negatives)
            anchors, positives, negatives = (
                batch["anchors"],
                batch["positives"],
                batch["negatives"],
            )

            loss = train.train_sparse_embed(
                model=model,
                optimizer=optimizer,
                anchor=anchors,
                positive=positives,
                negative=negatives,
                threshold_flops=30,
                accelerator=accelerator,
            )

        if accelerator.is_main_process and save_each_epoch:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                "checkpoint/epoch" + str(epoch),
            )

    # Save at the end of the training loop
    # We check to make sure that only the main process will export the model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained("checkpoint")
```