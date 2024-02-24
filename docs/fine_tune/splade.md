# Splade

Training the Splade Model with PyTorch and Neural Cherche Library. The model is updated
each time we call `train.train_splade` function. It's higly recommended to use a GPU
and to use a Masked Language Model as the base model.

```python
from neural_cherche import models, utils, train, losses
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
epochs = 2

model = models.Splade(
    model_name_or_path="distilbert-base-uncased",
    device=device
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
flops_scheduler = losses.FlopsScheduler()

X = [
    ("query", "positive document", "negative document"),
    ("query", "positive document", "negative document"),
    ("query", "positive document", "negative document"),
]

for step, (anchor, positive, negative) in enumerate(utils.iter(
        X,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False
    )):

    loss = train.train_splade(
        model=model,
        optimizer=optimizer,
        anchor=anchor,
        positive=positive,
        negative=negative,
        threshold_flops=30,
        flops_loss_weight=flops_scheduler.get(),
        step=step,
        gradient_accumulation_steps=50,
    )

    if (step + 1) % 1000 == 0:
        # Save the model every 1000 steps
        model.save_pretrained("checkpoint")
```

We can load the checkpoint using:

```python
from neural_cherche import models
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.Splade(
    model_name_or_path="checkpoint",
    device=device
)
```