# TfIdf

TfIdf retriever based on cosine similarities.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*list[str]*)

    Fields to use to match the query to the documents.

- **tfidf** (*sklearn.feature_extraction.text.TfidfVectorizer*) – defaults to `None`

    TfidfVectorizer class of Sklearn to create a custom TfIdf retriever.

- **fit** (*bool*) – defaults to `True`



## Examples

```python
>>> from neural_cherche import retrieve
>>> from pprint import pprint

>>> documents = [
...     {"id": 0, "document": "Food"},
...     {"id": 1, "document": "Sports"},
...     {"id": 2, "document": "Cinema"},
... ]

>>> queries = ["Food", "Sports", "Cinema"]

>>> retriever = retrieve.TfIdf(
...     key="id",
...     on=["document"],
... )

>>> documents_embeddings = retriever.encode_documents(
...     documents=documents
... )

>>> retriever = retriever.add(
...     documents_embeddings=documents_embeddings,
... )

>>> queries_embeddings = retriever.encode_queries(
...     queries=queries
... )

>>> scores = retriever(
...     queries_embeddings=queries_embeddings,
...     k=4
... )

>>> pprint(scores)
[[{'id': 0, 'similarity': 1.0}],
 [{'id': 1, 'similarity': 0.9999999999999999}],
 [{'id': 2, 'similarity': 0.9999999999999999}]]
```

## Methods

???- note "__call__"

    Retrieve documents from batch of queries.

    **Parameters**

    - **queries_embeddings**     (*dict[str, scipy.sparse._csr.csr_matrix]*)    
    - **k**     (*int*)     – defaults to `None`    
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.
    - **batch_size**     (*int*)     – defaults to `2000`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    
???- note "add"

    Add new documents to the TFIDF retriever. The tfidf won't be refitted.

    **Parameters**

    - **documents_embeddings**     (*dict[str, scipy.sparse._csr.csr_matrix]*)    
    
???- note "encode_documents"

    Encode queries into sparse matrix.

    **Parameters**

    - **documents**     (*list[dict]*)    
        Documents in TFIdf retriever are static. The retriever must be reseted to index new documents.
    
???- note "encode_queries"

    Encode queries into sparse matrix.

    **Parameters**

    - **queries**     (*list[str]*)    
    
???- note "top_k"

    Return the top k documents for each query.

    **Parameters**

    - **similarities**     (*scipy.sparse._csc.csc_matrix*)    
    - **k**     (*int*)    
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.
    
## References

1. [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

