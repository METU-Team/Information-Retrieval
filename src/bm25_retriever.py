"""
Build a PyTerrier inverted index and run BM25 retrieval.

Preprocessing steps applied:
  - lowercasing
  - stopword removal (Terrier built-in English stopword list)
  - Porter stemming
  - positional index (supports phrase queries in the future)
"""
import os
import pyterrier as pt
import pandas as pd
from src.config import INDEX_DIR, BM25_TOP_K

if not pt.started():
    pt.init()


def build_index(corpus_df: pd.DataFrame, overwrite: bool = False) -> pt.IndexRef:
    """
    corpus_df must have columns: ['docno', 'text']
    Returns a PyTerrier IndexRef.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    indexer = pt.DFIndexer(
        INDEX_DIR,
        overwrite=overwrite,
        blocks=True,          # positional index
        stemmer="porter",
        stopwords="terrier",
        tokeniser="english",
    )
    index_ref = indexer.index(corpus_df["text"], corpus_df["docno"])
    return index_ref


def load_index() -> pt.IndexRef:
    return pt.IndexRef.of(INDEX_DIR)


def get_bm25_retriever(index_ref: pt.IndexRef, top_k: int = BM25_TOP_K) -> pt.BatchRetrieve:
    return pt.BatchRetrieve(
        index_ref,
        wmodel="BM25",
        num_results=top_k,
        metadata=["docno"],
    )


def retrieve(retriever: pt.BatchRetrieve, queries_df: pd.DataFrame) -> pd.DataFrame:
    """
    queries_df must have columns: ['qid', 'query']
    Returns a results DataFrame with columns: ['qid', 'docno', 'score', 'rank']
    """
    return retriever.transform(queries_df)