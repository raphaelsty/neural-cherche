# Colbert

Training the ColBERT Model with PyTorch and Neural Cherche Library. The model is updated
each time we call `train.train_colbert` function. It's higly recommended to use a GPU
and to use a Sentence Transformer model as the base model.

```python
from neural_cherche import models, utils, train
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
epochs = 2

model = models.ColBERT(
    model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    device=device
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

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

    loss = train.train_colbert(
        model=model,
        optimizer=optimizer,
        anchor=anchor,
        positive=positive,
        negative=negative,
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

model = models.ColBERT(
    model_name_or_path="checkpoint",
    device=device
)
```