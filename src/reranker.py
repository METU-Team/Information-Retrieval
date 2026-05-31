"""
Cross-encoder re-ranking stage.

Takes the merged candidate pool (e.g. from RRF) and re-scores every
query–passage pair with a cross-encoder that reads both texts jointly.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO with relevance labels
  - Input: (query, passage) → scalar relevance score
  - Much more accurate than bi-encoder but too expensive for first-stage
    retrieval — hence used only for re-ranking a small candidate set.
"""
import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from src.config import CROSSENCODER_MODEL, FINAL_TOP_K, DEVICE


def load_crossencoder() -> CrossEncoder:
    """Load the cross-encoder model."""
    model = CrossEncoder(CROSSENCODER_MODEL, max_length=512, device=DEVICE)
    return model


def rerank(
    cross_model: CrossEncoder,
    candidate_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    top_k: int = FINAL_TOP_K,
) -> pd.DataFrame:
    """
    Re-rank candidates using cross-encoder scores.

    Args:
        cross_model: loaded CrossEncoder instance
        candidate_df: DataFrame with ['qid', 'docno', 'score', 'rank']
                      (output of RRF or any first-stage retrieval)
        queries_df: DataFrame with ['qid', 'query']
        corpus_df: DataFrame with ['docno', 'text']
        top_k: number of final results to return per query

    Returns:
        DataFrame with ['qid', 'docno', 'score', 'rank'] re-ranked by
        cross-encoder relevance.
    """
    # Build lookup dictionaries
    query_lookup = dict(zip(queries_df["qid"].astype(str), queries_df["query"]))
    doc_lookup = dict(zip(corpus_df["docno"].astype(str), corpus_df["text"]))

    reranked_rows = []

    for qid, group in tqdm(candidate_df.groupby("qid"), desc="Cross-encoder re-ranking"):
        qid_str = str(qid)
        query_text = query_lookup.get(qid_str, "")
        if not query_text:
            continue

        # Build (query, passage) pairs for this query
        pairs = []
        docnos = []
        for _, row in group.iterrows():
            docno = str(row["docno"])
            passage_text = doc_lookup.get(docno, "")
            if passage_text:
                pairs.append((query_text, passage_text))
                docnos.append(docno)

        if not pairs:
            continue

        # Score all pairs at once (batch prediction)
        scores = cross_model.predict(pairs, show_progress_bar=False)

        # Sort by cross-encoder score (descending) and keep top_k
        scored = sorted(zip(docnos, scores), key=lambda x: x[1], reverse=True)[:top_k]

        for rank, (docno, score) in enumerate(scored):
            reranked_rows.append({
                "qid": qid_str,
                "docno": docno,
                "score": float(score),
                "rank": rank,
            })

    return pd.DataFrame(reranked_rows)
