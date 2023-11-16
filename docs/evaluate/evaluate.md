# Evaluate

Neural-Cherche evaluation is based on [RANX](https://github.com/AmenRa/ranx). We can also download datasets of [BEIR Benchmark](https://github.com/beir-cellar/beir) with the `utils.load_beir` function.


## Installation

```bash
pip install "neural-cherche[eval]"
```

## Usage

Let"s first create a pipeline which output candidates and scores:

```python
from neural_cherche import models, retrieve, utils

model = models.Splade(
    model_name_or_path="distilbert-base-uncased",
    device="cpu",
)

# Input dataset for evaluation
documents, queries_ids, queries, qrels = utils.load_beir(
    "scifact",
    split="test",
)

# Let"s keep only 10 documents for the example
documents = documents[:10]

retriever = retrieve.Splade(key="id", on=["title", "text"], model=model)

documents_embeddings = retriever.encode_documents(
    documents=documents,
    batch_size=1,
)

documents_embeddings = retriever.add(
    documents_embeddings=documents_embeddings,
)

queries_embeddings = retriever.encode_queries(
    queries=queries,
    batch_size=batch_size,
)

scores = retriever(
    queries_embeddings=queries_embeddings,
    k=30,
    batch_size=batch_size,
)

utils.evaluate(
    scores=scores,
    qrels=qrels,
    queries_ids=queries_ids,
    metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
)
```

```python
{
    "map": 0.45,
    "ndcg@10": 0.45,
    "ndcg@100": 0.45,
    "recall@10": 0.45,
    "recall@100": 0.45
}
```

## Evaluation dataset

Here are what documents should looks like (an id with multiples fields):

```python
[
    {
        "id": "document_0",
        "title": "Bayesian measures of model complexity and fit",
        "text": "Summary. We consider the problem of comparing complex hierarchical models in which the number of parameters is not clearly defined. Using an information theoretic argument we derive a measure pD for the effective number of parameters in a model as the difference between the posterior mean of the deviance and the deviance at the posterior means of the parameters of interest. In general pD approximately corresponds to the trace of the product of Fisher's information and the posterior covariance, which in normal models is the trace of the ‘hat’ matrix projecting observations onto fitted values. Its properties in exponential families are explored. The posterior mean deviance is suggested as a Bayesian measure of fit or adequacy, and the contributions of individual observations to the fit and complexity can give rise to a diagnostic plot of deviance residuals against leverages. Adding pD to the posterior mean deviance gives a deviance information criterion for comparing models, which is related to other information criteria and has an approximate decision theoretic justification. The procedure is illustrated in some examples, and comparisons are drawn with alternative Bayesian and classical proposals. Throughout it is emphasized that the quantities required are trivial to compute in a Markov chain Monte Carlo analysis.",
    },
    {
        "id": "document_1",
        "title": "Simplifying likelihood ratios",
        "text": "Likelihood ratios are one of the best measures of diagnostic accuracy, although they are seldom used, because interpreting them requires a calculator to convert back and forth between “probability” and “odds” of disease. This article describes a simpler method of interpreting likelihood ratios, one that avoids calculators, nomograms, and conversions to “odds” of disease. Several examples illustrate how the clinician can use this method to refine diagnostic decisions at the bedside.",
    },
]
```

Queries is a list of strings:

```python
[
    "Varenicline monotherapy is more effective after 12 weeks of treatment compared to combination nicotine replacement therapies with varenicline or bupropion.",
    "Venules have a larger lumen diameter than arterioles.",
    "Venules have a thinner or absent smooth layer compared to arterioles.",
    "Vitamin D deficiency effects the term of delivery.",
    "Vitamin D deficiency is unrelated to birth weight.",
    "Women with a higher birth weight are more likely to develop breast cancer later in life.",
]
```

QueriesIds is a list of ids with respect to the order of queries:

```python
[
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
]
```

Qrels is the mapping between queries ids as key and dict of relevant documents with 1 as value:

```python
{
    "1": {"document_0": 1},
    "3": {"document_10": 1},
    "5": {"document_5": 1},
    "13": {"document_22": 1},
    "36": {"document_23": 1, "document_0": 1},
    "42": {"document_2": 1},
}
```

## Metrics

We can evaluate our model with various metrics detailed [here](https://amenra.github.io/ranx/metrics/).