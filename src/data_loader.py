"""
Load a reproducible subset of MS MARCO passages and queries using ir-datasets.

Qrels-aware corpus loading: guarantees all relevant passages from the
evaluation qrels are present in the indexed corpus, so that MRR/nDCG
scores are not artificially deflated by missing ground-truth documents.
"""
import ir_datasets
import pandas as pd
from src.config import CORPUS_DATASET, EVAL_DATASET, CORPUS_SUBSET_SIZE, QUERIES_SUBSET_SIZE


def load_queries_and_qrels(n_queries: int = QUERIES_SUBSET_SIZE):
    """
    Returns:
        queries_df: DataFrame with columns ['qid', 'query']
        qrels_df:   DataFrame with columns ['qid', 'docno', 'label']
    Only includes qrels for the selected queries.
    """
    dataset = ir_datasets.load(EVAL_DATASET)

    # Collect all qrels
    qrel_set = set()
    all_qrels = []
    for qrel in dataset.qrels_iter():
        qrel_set.add(str(qrel.query_id))
        all_qrels.append({
            "qid": str(qrel.query_id),
            "docno": str(qrel.doc_id),
            "label": int(qrel.relevance),
        })

    # Select n_queries that have qrels
    selected_qids = set()
    queries = []
    for q in dataset.queries_iter():
        if str(q.query_id) in qrel_set:
            queries.append({"qid": str(q.query_id), "query": q.text})
            selected_qids.add(str(q.query_id))
        if len(selected_qids) >= n_queries:
            break

    # Filter qrels to selected queries only
    filtered_qrels = [qr for qr in all_qrels if qr["qid"] in selected_qids]

    return pd.DataFrame(queries), pd.DataFrame(filtered_qrels)


def load_corpus(qrels_df: pd.DataFrame = None, subset_size: int = CORPUS_SUBSET_SIZE) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ['docno', 'text']
    'docno' must be a string (PyTerrier requirement).

    If qrels_df is provided, guarantees all relevant passages are included
    in the corpus. Remaining slots are filled with sequential passages.
    """
    dataset = ir_datasets.load(CORPUS_DATASET)

    if qrels_df is not None:
        relevant_docnos = set(qrels_df[qrels_df["label"] > 0]["docno"].unique())
        print(f"Qrels reference {len(relevant_docnos)} unique relevant passages — guaranteeing inclusion.")

        # Fetch relevant docs via random access (fast)
        store = dataset.docs_store()
        relevant_rows = []
        for docno in relevant_docnos:
            try:
                doc = store.get(docno)
                relevant_rows.append({"docno": str(doc.doc_id), "text": doc.text})
            except (KeyError, TypeError):
                print(f"  Warning: doc {docno} not found in corpus, skipping.")

        # Fill remaining slots with sequential (non-relevant) passages
        filler_limit = max(0, subset_size - len(relevant_rows))
        filler_rows = []
        for doc in dataset.docs_iter():
            if len(filler_rows) >= filler_limit:
                break
            docno = str(doc.doc_id)
            if docno not in relevant_docnos:
                filler_rows.append({"docno": docno, "text": doc.text})

        all_rows = relevant_rows + filler_rows
        print(f"Corpus loaded: {len(relevant_rows)} relevant + {len(filler_rows)} filler = {len(all_rows)} total")
        return pd.DataFrame(all_rows)
    else:
        # Original behavior: sequential subset
        rows = []
        for i, doc in enumerate(dataset.docs_iter()):
            if i >= subset_size:
                break
            rows.append({"docno": str(doc.doc_id), "text": doc.text})
        return pd.DataFrame(rows)