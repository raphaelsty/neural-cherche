__all__ = ["evaluate", "load_beir"]


def load_beir(dataset_name: str, split: str = "test") -> tuple[list, list, dict]:
    """Load BEIR dataset.

    Parameters
    ----------
    dataset_name
        Dataset name: scifact.

    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = util.download_and_unzip(
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
        "./evaluation_datasets/",
    )

    documents, queries, qrels = GenericDataLoader(data_folder=data_path).load(
        split=split
    )

    documents = [
        {
            "id": document_id,
            "title": document["title"],
            "text": document["text"],
        }
        for document_id, document in documents.items()
    ]

    return documents, list(queries.keys()), list(queries.values()), qrels


def evaluate(
    scores: list[list[dict]],
    qrels: dict,
    queries_ids: list[str],
    metrics: list = [],
):
    """Evaluate candidates matchs.

    Parameters
    ----------
    matchs
        Matchs.
    qrels
        Qrels.
    queries
        index of queries of qrels.
    k
        Number of documents to retrieve.
    metrics
        Metrics to compute.

    Examples
    --------
    >>> from neural_cherche import models, retrieve, utils
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.Splade(
    ...     model_name_or_path="distilbert-base-uncased",
    ...     device="cpu",
    ... )

    >>> documents, queries_ids, queries, qrels = utils.load_beir(
    ...     "scifact",
    ...     split="test",
    ... )

    >>> documents = documents[:10]

    >>> retriever = retrieve.Splade(
    ...     key="id",
    ...     on=["title", "text"],
    ...     model=model
    ... )

    >>> documents_embeddings = retriever.encode_documents(
    ...     documents=documents,
    ...     batch_size=1,
    ... )

    >>> documents_embeddings = retriever.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> queries_embeddings = retriever.encode_queries(
    ...     queries=queries,
    ...     batch_size=1,
    ... )

    >>> scores = retriever(
    ...     queries_embeddings=queries_embeddings,
    ...     k=30,
    ...     batch_size=1,
    ... )

    >>> utils.evaluate(
    ...     scores=scores,
    ...     qrels=qrels,
    ...     queries_ids=queries_ids,
    ...     metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]
    ... )
    {'map': 0.0033333333333333335, 'ndcg@10': 0.0033333333333333335, 'ndcg@100': 0.0033333333333333335, 'recall@10': 0.0033333333333333335, 'recall@100': 0.0033333333333333335}

    """
    from ranx import Qrels, Run, evaluate

    qrels = Qrels(qrels)

    run_dict = {
        id_query: {
            match["id"]: 1 - (rank / len(query_matchs))
            for rank, match in enumerate(query_matchs)
        }
        for id_query, query_matchs in zip(queries_ids, scores)
    }

    run = Run(run_dict)

    if not metrics:
        metrics = ["ndcg@10"] + [f"hits@{k}" for k in [1, 2, 3, 4, 5, 10]]

    return evaluate(
        qrels,
        run,
        metrics,
        make_comparable=True,
    )
