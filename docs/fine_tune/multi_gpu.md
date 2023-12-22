# Multi-GPU (Accelerator)


Training any of the models on multiple GPU via the accelerator library is simple. You just need to modify the training loop in a few key ways:

```python
from neural_cherche import models, utils, train
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator


# Wrap in main function to avoid multiprocessing issues
if __name__ == "__main__"":
    accelerator = Accelerator()
    device = accelerator.device
    batch_size = 32
    epochs = 2
    save_on_epoch = True

    model = models.SparseEmbed(
        model_name_or_path="distilbert-base-uncased",
        device=device
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # prepare your dataset -- this example uses a huggingface `datasets` object
    ...

    # Convert the data into a PyTorch dataloader for ease of preparation
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Wrap the model, optimizer, and data loader in the accelerator
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    for epoch in range(epochs):
        for batch_data in enumerate(data_loader):
            # Assuming batch_data is a tuple in the form (anchors, positives, negatives)
            anchors, positives, negatives = batch_data

            loss = train_sparse_embed(
                model=model,
                optimizer=optimizer,
                anchor=anchors,
                positive=positives,
                negative=negatives,
                threshold_flops=30,
                accelerator=accelerator,
            )
    
        if accelerator.is_main_process and save_on_epoch:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
            "checkpoint/epoch" + str(epoch),
            )

    # Save at the end of the training loop
    # We check to make sure that only the main process will export the model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained("checkpoint", accelerator=True)
```