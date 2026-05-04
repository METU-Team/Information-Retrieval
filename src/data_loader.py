"""
Load a reproducible subset of MS MARCO passages and queries using ir-datasets.
"""
import ir_datasets
import pandas as pd
from src.config import CORPUS_DATASET, EVAL_DATASET, CORPUS_SUBSET_SIZE, QUERIES_SUBSET_SIZE

def load_corpus(subset_size: int = CORPUS_SUBSET_SIZE) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ['docno', 'text']
    'docno' must be a string (PyTerrier requirement).
    """
    dataset = ir_datasets.load(CORPUS_DATASET)
    rows = []
    for i, doc in enumerate(dataset.docs_iter()):
        if i >= subset_size:
            break
        rows.append({"docno": str(doc.doc_id), "text": doc.text})
    return pd.DataFrame(rows)


def load_queries_and_qrels(n_queries: int = QUERIES_SUBSET_SIZE):
    """
    Returns:
        queries_df: DataFrame with columns ['qid', 'query']
        qrels_df:   DataFrame with columns ['qid', 'docno', 'label']
    """
    dataset = ir_datasets.load(EVAL_DATASET)
    queries, qrels = [], []

    qrel_set = set()
    for qrel in dataset.qrels_iter():
        qrel_set.add(str(qrel.query_id))
        qrels.append({
            "qid": str(qrel.query_id),
            "docno": str(qrel.doc_id),
            "label": int(qrel.relevance)
        })

    seen = 0
    for q in dataset.queries_iter():
        if str(q.query_id) in qrel_set:
            queries.append({"qid": str(q.query_id), "query": q.text})
            seen += 1
        if seen >= n_queries:
            break

    return pd.DataFrame(queries), pd.DataFrame(qrels)