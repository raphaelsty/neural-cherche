# Pre-trained Models

Neural-Cherche models such as ColBERT and SparseEmbed should be initialized with a pre-trained sentence-transformer model. The pre-trained models names can be found in the [sentence-transformers documentation](https://www.sbert.net/docs/pretrained_models.html) or on HuggingFace hub.

After having selected a pre-trained checkpoint, we should fine-tune it on our dataset. If we don't
wan't to fine-tune the model, we can use the `raphaelsty/neural-cherche-sparse-embed` and `raphaelsty/neural-cherche-colbert` checkpoints.

## Fine-tuning ColBERT on the Scifact Dataset

Here is a sample code to fine-tune ColBERT on the Scifact Dataset. If we plan to run this code, we should install neural-cherche with the following command:

```bash
pip install "neural-cherche[eval]"
```

There are other dataset available from the [BEIR Benchmark](https://github.com/beir-cellar/beir)
which can be used with the `utils.load_beir` function such as `scifact`, `trec-covid`, `cord19`, `fiqa`, `hotpotqa`, `natural-questions`, `msmarco`, `eli5`, `quora`. Of course, we can use our own dataset by providing triples. Then, by building queries, documents and qrels, we can evaluate the model using the `utils.evaluate` function.

```python
import random

import torch
from neural_cherche import models, retrieve, rank, train, utils

documents, queries_ids, queries, qrels = utils.load_beir(
    "scifact",
    split="train",
)

model = models.ColBERT(
    model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Building dataset
mapping_documents = {
    document["id"]: " ".join([document[field] for field in ["title", "text"]])
    for document in documents
}

X = []
for query, (_, query_documents) in zip(queries, qrels.items()):
    for query_document in list(query_documents.keys()):
        # Building triples, query, positive document, random negative document
        X.append(
            (
                query,
                mapping_documents[query_document],
                random.choice(list(mapping_documents.values())),
            )
        )
# End building dataset

# Training loop
batch_size = 1

for step, (anchor, positive, negative) in enumerate(
    utils.iter(X, epochs=1, batch_size=batch_size, shuffle=True)
):
    loss = train.train_colbert(
        model=model,
        optimizer=optimizer,
        anchor=anchor,
        positive=positive,
        negative=negative,
        step=step,
        gradient_accumulation_steps=5,
    )

    # Eval the model every 512 steps
    if (step + 1) % 3 == 0:
        test_documents, queries_ids, queries, qrels = utils.load_beir(
            dataset_name="scifact",
            split="test",
        )

        # Setting up the retriever 
        retriever = retrieve.BM25(
            key="id",
            on=["title", "text"],
        )

        retriever_documents_embeddings = retriever.encode_documents(
            documents=test_documents,
        )

        retriever.add(
            documents_embeddings=retriever_documents_embeddings,
        )

        queries_embeddings = retriever.encode_queries(
            queries=queries,
        )

        candidates = retriever(
            queries_embeddings=queries_embeddings,
            k=100,
        )

        # Setting up the ranker
        ranker = rank.ColBERT(key="id", on=["title", "text"], model=model)

        ranker_documents_embeddings = ranker.encode_documents(
            documents=candidates,
            batch_size=batch_size,
        )

        ranker_queries_embeddings = ranker.encode_queries(
            queries=queries,
            batch_size=batch_size,
        )

        scores = ranker(
            documents=candidates,
            queries_embeddings=ranker_queries_embeddings,
            documents_embeddings=ranker_documents_embeddings,
            k=10,
        )

        # Evaluate the pipeline
        scores = utils.evaluate(
            scores=scores,
            qrels=qrels,
            queries_ids=queries_ids,
            metrics=["ndcg@10"] + [f"hits@{k}" for k in range(1, 10)],
        )

        print(scores)

model.save_pretrained("colbert-scifact")
```