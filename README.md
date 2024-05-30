<div align="center">
  <h1>Neural-Cherche</h1>
  <p>Neural Search</p>
</div>

<p align="center"><img width=500 src="docs/img/logo.png"/></p>

<div align="center">
  <!-- Documentation -->
  <a href="https://raphaelsty.github.io/neural-cherche/"><img src="https://img.shields.io/website?label=Documentation&style=flat-square&url=https%3A%2F%2Fraphaelsty.github.io/neural-cherche/%2F" alt="documentation"></a>
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="license"></a>
</div>

Neural-Cherche is a library designed to fine-tune neural search models such as Splade, ColBERT, and SparseEmbed on a specific dataset. Neural-Cherche also provide classes to run efficient inference on a fine-tuned retriever or ranker. Neural-Cherche aims to offer a straightforward and effective method for fine-tuning and utilizing neural search models in both offline and online settings. It also enables users to save all computed embeddings to prevent redundant computations.

Neural-Cherche is compatible with CPU, GPU and MPS devices. We can fine-tune ColBERT from any
Sentence Transformer pre-trained checkpoint. Splade and SparseEmbed are more tricky to fine-tune and need a MLM pre-trained model.

## Installation

We can install neural-cherche using:

```
pip install neural-cherche
```

If we plan to evaluate our model while training install:

```
pip install "neural-cherche[eval]"
```

## Documentation

The complete documentation is available [here](https://raphaelsty.github.io/neural-cherche/).

## Quick Start

Your training dataset must be made out of triples `(anchor, positive, negative)` where anchor is a query, positive is a document that is directly linked to the anchor and negative is a document that is not relevant for the anchor.

```python
X = [
    ("anchor 1", "positive 1", "negative 1"),
    ("anchor 2", "positive 2", "negative 2"),
    ("anchor 3", "positive 3", "negative 3"),
]
```

And here is how to fine-tune ColBERT from a Sentence Transformer pre-trained checkpoint using neural-cherche:

```python
import torch

from neural_cherche import models, utils, train

model = models.ColBERT(
    model_name_or_path="raphaelsty/neural-cherche-colbert",
    device="cuda" if torch.cuda.is_available() else "cpu" # or mps
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)

X = [
    ("query", "positive document", "negative document"),
    ("query", "positive document", "negative document"),
    ("query", "positive document", "negative document"),
]

for step, (anchor, positive, negative) in enumerate(utils.iter(
        X,
        epochs=1, # number of epochs
        batch_size=8, # number of triples per batch
        shuffle=True
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

## Retrieval

Here is how to use the fine-tuned ColBERT model to re-rank documents:

```python
import torch
from lenlp import sparse

from neural_cherche import models, rank, retrieve

documents = [
    {"id": "doc1", "title": "Paris", "text": "Paris is the capital of France."},
    {"id": "doc2", "title": "Montreal", "text": "Montreal is the largest city in Quebec."},
    {"id": "doc3", "title": "Bordeaux", "text": "Bordeaux in Southwestern France."},
]

retriever = retrieve.BM25(
    key="id",
    on=["title", "text"],
    count_vectorizer=sparse.CountVectorizer(
        normalize=True, ngram_range=(3, 5), analyzer="char_wb", stop_words=[]
    ),
    k1=1.5,
    b=0.75,
    epsilon=0.0,
)

model = models.ColBERT(
    model_name_or_path="raphaelsty/neural-cherche-colbert",
    device="cuda" if torch.cuda.is_available() else "cpu",  # or mps
)

ranker = rank.ColBERT(
    key="id",
    on=["title", "text"],
    model=model,
)

documents_embeddings = retriever.encode_documents(
    documents=documents,
)

retriever.add(
    documents_embeddings=documents_embeddings,
)
```

Now we can retrieve documents using the fine-tuned model:

```python
queries = ["Paris", "Montreal", "Bordeaux"]

queries_embeddings = retriever.encode_queries(
    queries=queries,
)

ranker_queries_embeddings = ranker.encode_queries(
    queries=queries,
)

candidates = retriever(
    queries_embeddings=queries_embeddings,
    batch_size=32,
    k=100,  # number of documents to retrieve
)

# Compute embeddings of the candidates with the ranker model.
# Note, we could also pre-compute all the embeddings.
ranker_documents_embeddings = ranker.encode_candidates_documents(
    candidates=candidates,
    documents=documents,
    batch_size=32,
)

scores = ranker(
    queries_embeddings=ranker_queries_embeddings,
    documents_embeddings=ranker_documents_embeddings,
    documents=candidates,
    batch_size=32,
)

scores
```

```python
[[{'id': 0, 'similarity': 22.825355529785156},
  {'id': 1, 'similarity': 11.201947212219238},
  {'id': 2, 'similarity': 10.748161315917969}],
 [{'id': 1, 'similarity': 23.21628189086914},
  {'id': 0, 'similarity': 9.9658203125},
  {'id': 2, 'similarity': 7.308732509613037}],
 [{'id': 1, 'similarity': 6.4031805992126465},
  {'id': 0, 'similarity': 5.601611137390137},
  {'id': 2, 'similarity': 5.599479675292969}]]
```

Neural-Cherche provides a `SparseEmbed`, a `SPLADE`, a `TFIDF`, a `BM25` retriever and a `ColBERT` ranker which can be used to re-order output of a retriever. For more information, please refer to the [documentation](https://raphaelsty.github.io/neural-cherche/).

### Pre-trained Models

We provide pre-trained checkpoints specifically designed for neural-cherche: [raphaelsty/neural-cherche-sparse-embed](https://huggingface.co/raphaelsty/neural-cherche-sparse-embed) and [raphaelsty/neural-cherche-colbert](https://huggingface.co/raphaelsty/neural-cherche-colbert). Those checkpoints are fine-tuned on a subset of the MS-MARCO dataset and would benefit from being fine-tuned on your specific dataset. You can fine-tune ColBERT from any Sentence Transformer pre-trained checkpoint in order to fit your specific language. You should use a MLM based-checkpoint to fine-tune SparseEmbed.

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0pky"></th>
    <th class="tg-rvyq" colspan="3">scifact dataset</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">model</td>
    <td class="tg-7btt">HuggingFace Checkpoint</td>
    <td class="tg-rvyq">ndcg@10</td>
    <td class="tg-rvyq">hits@10</td>
    <td class="tg-rvyq">hits@1</td>
  </tr>
  <tr>
    <td class="tg-c3ow">TfIdf</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">0,62</td>
    <td class="tg-c3ow">0,86</td>
    <td class="tg-c3ow">0,50</td>
  </tr>
  <tr>
    <td class="tg-c3ow">BM25</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">0,69</td>
    <td class="tg-c3ow">0,92</td>
    <td class="tg-c3ow">0,56</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SparseEmbed</td>
    <td class="tg-c3ow">raphaelsty/neural-cherche-sparse-embed</td>
    <td class="tg-c3ow">0,62</td>
    <td class="tg-c3ow">0,87</td>
    <td class="tg-c3ow">0,48</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Sentence Transformer</td>
    <td class="tg-c3ow">sentence-transformers/all-mpnet-base-v2</td>
    <td class="tg-c3ow">0,66</td>
    <td class="tg-c3ow">0,89</td>
    <td class="tg-c3ow">0,53</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ColBERT</td>
    <td class="tg-c3ow">raphaelsty/neural-cherche-colbert</td>
    <td class="tg-7btt">0,70</td>
    <td class="tg-7btt">0,92</td>
    <td class="tg-7btt">0,58</td>
  </tr>
  <tr>
    <td class="tg-c3ow">TfIDF Retriever + ColBERT Ranker</td>
    <td class="tg-c3ow">raphaelsty/neural-cherche-colbert</td>
    <td class="tg-7btt">0,71</td>
    <td class="tg-7btt">0,94</td>
    <td class="tg-7btt">0,59</td>
  </tr>
  <tr>
    <td class="tg-c3ow">BM25 Retriever + ColBERT Ranker</td>
    <td class="tg-c3ow">raphaelsty/neural-cherche-colbert</td>
    <td class="tg-7btt">0,72</td>
    <td class="tg-7btt">0,95</td>
    <td class="tg-7btt">0,59</td>
  </tr>
</tbody>
</table>

### Neural-Cherche Contributors

- [Benjamin Clavié](https://github.com/bclavie)
- [Arthur Satouf](https://github.com/arthur-75)

## References

- *[SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)* authored by Thibault Formal, Benjamin Piwowarski, Stéphane Clinchant, SIGIR 2021.

- *[SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval](https://arxiv.org/abs/2109.10086)* authored by Thibault Formal, Carlos Lassance, Benjamin Piwowarski, Stéphane Clinchant, SIGIR 2022.

- *[SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval](https://research.google/pubs/pub52289/)* authored by Weize Kong, Jeffrey M. Dudek, Cheng Li, Mingyang Zhang, and Mike Bendersky, SIGIR 2023.

- *[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)* authored by Omar Khattab, Matei Zaharia, SIGIR 2020.

## License

This Python library is licensed under the MIT open-source license, and the splade model is licensed as non-commercial only by the authors. SparseEmbed and ColBERT are fully open-source including commercial usage.
