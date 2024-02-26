# Scoring

We can compute similarity between queries and documents without using a retriever using 
the `scores` method.

## ColBERT

```python
import torch

from neural_cherche import models

model = models.ColBERT(
    model_name_or_path="raphaelsty/neural-cherche-colbert",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

model.scores(
    queries=[
        "What is the capital of France?",
        "What is the largest city in Quebec?",
        "Where is Bordeaux?",
    ],
    documents=[
        "Paris is the capital of France.",
        "Montreal is the largest city in Quebec.",
        "Bordeaux in Southwestern France.",
    ],
    batch_size=32,
)
```

```python
tensor([20.6498, 26.2132, 23.7048])
```

## Splade

```python
import torch

from neural_cherche import models

model = models.Splade(
    model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

model.scores(
    queries=[
        "What is the capital of France?",
        "What is the largest city in Quebec?",
        "Where is Bordeaux?",
    ],
    documents=[
        "Paris is the capital of France.",
        "Montreal is the largest city in Quebec.",
        "Bordeaux in Southwestern France.",
    ],
    batch_size=32,
)
```

```python
tensor([517.9335, 526.4659, 395.0022])
```

## SparseEmbed

```python
import torch

from neural_cherche import models

model = models.SparseEmbed(
    model_name_or_path="raphaelsty/neural-cherche-sparse-embed",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

model.scores(
    queries=[
        "What is the capital of France?",
        "What is the largest city in Quebec?",
        "Where is Bordeaux?",
    ],
    documents=[
        "Paris is the capital of France.",
        "Montreal is the largest city in Quebec.",
        "Bordeaux in Southwestern France.",
    ],
    batch_size=32,
)
```

```python
tensor([150.6469, 161.5140, 109.4595])
```