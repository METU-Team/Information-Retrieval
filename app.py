"""
Streamlit-based interactive UI for the hybrid passage retrieval system.

Run with:
    streamlit run app.py

Features:
  - Natural language query input
  - Pipeline selector (BM25 / Dense / Hybrid RRF / Hybrid+ReRank)
  - Ranked passage display with relevance scores
  - Query-term highlighting in passage snippets
  - Latency reporting per stage
"""
import os
import re
import time
import streamlit as st
import pandas as pd
import pyterrier as pt

# Must init PyTerrier before importing project modules
if not pt.started():
    pt.init()

from src.config import INDEX_DIR, FAISS_INDEX_PATH, FINAL_TOP_K
from src.bm25_retriever import load_index, get_bm25_retriever, retrieve
from src.dense_retriever import load_biencoder, load_faiss_index, dense_retrieve
from src.hybrid_retriever import convex_combination
from src.reranker import load_crossencoder, rerank
from src.data_loader import load_corpus, load_queries_and_qrels, load_trec_dl_queries_and_qrels


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Verifiers — Hybrid Passage Retrieval",
    page_icon="🔍",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 1100px; margin: 0 auto; }
    .result-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid #3d3d5c;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        color: #e0e0e0;
    }
    .result-card:hover {
        border-color: #6c63ff;
        box-shadow: 0 0 15px rgba(108, 99, 255, 0.15);
    }
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6c63ff, #48c6ef);
        color: white;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        margin-right: 8px;
    }
    .score-badge {
        display: inline-block;
        background: rgba(108, 99, 255, 0.15);
        color: #a8a4ff;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82em;
    }
    .docno-tag {
        color: #888;
        font-size: 0.78em;
        font-family: monospace;
    }
    mark { background-color: #ffe066; color: #1e1e2e; padding: 1px 3px; border-radius: 3px; }
    .latency-bar {
        background: rgba(108, 99, 255, 0.08);
        border-left: 3px solid #6c63ff;
        padding: 8px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 0.85em;
        color: #aaa;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached Model Loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BM25 index…")
def _load_bm25():
    index_ref = load_index()
    retriever = get_bm25_retriever(index_ref)
    return retriever


@st.cache_resource(show_spinner="Loading bi-encoder + FAISS…")
def _load_dense():
    model = load_biencoder()
    faiss_idx, passage_ids = load_faiss_index()
    return model, faiss_idx, passage_ids


@st.cache_resource(show_spinner="Loading cross-encoder…")
def _load_crossencoder():
    return load_crossencoder()


@st.cache_resource(show_spinner="Loading corpus…")
def _load_corpus():
    queries_df, qrels_df = load_queries_and_qrels()
    trec_dl_queries, trec_dl_qrels = load_trec_dl_queries_and_qrels()
    combined_qrels = pd.concat([qrels_df, trec_dl_qrels], ignore_index=True).drop_duplicates(
        subset=["qid", "docno"], keep="first"
    )
    corpus_df = load_corpus(qrels_df=combined_qrels)
    return corpus_df


def highlight_terms(text: str, query: str) -> str:
    """Highlight query terms in passage text."""
    terms = set(query.lower().split())
    # Escape HTML
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for term in terms:
        if len(term) < 2:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", text)
    return text


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")
    pipeline = st.selectbox(
        "Retrieval pipeline",
        ["BM25 Only", "Dense Only", "Hybrid (CC)", "Hybrid (CC) + Cross-Encoder"],
        index=3,
    )
    top_k = st.slider("Results to display", 5, 50, FINAL_TOP_K)
    st.markdown("---")
    st.markdown("""
    **Pipeline stages:**
    1. 🔤 **BM25** — lexical matching
    2. 🧠 **Dense** — semantic bi-encoder
    3. 🔀 **Convex Combination** — score fusion
    4. 🎯 **Cross-Encoder** — neural re-ranking
    """)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Verifiers — Hybrid Passage Retrieval")
st.markdown("*Lexical, Dense, and Neural Re-ranking Paradigms*")

# ─── Query Input ─────────────────────────────────────────────────────────────
query = st.text_input(
    "Enter your query",
    placeholder="e.g., what is the speed of light",
    key="query_input",
)

if query:
    query_df = pd.DataFrame([{"qid": "interactive", "query": query}])

    timings = {}

    # BM25
    if pipeline in ["BM25 Only", "Hybrid (CC)", "Hybrid (CC) + Cross-Encoder"]:
        bm25_retriever = _load_bm25()
        t0 = time.time()
        bm25_results = retrieve(bm25_retriever, query_df)
        timings["BM25"] = time.time() - t0

    # Dense
    if pipeline in ["Dense Only", "Hybrid (CC)", "Hybrid (CC) + Cross-Encoder"]:
        bi_model, faiss_idx, passage_ids = _load_dense()
        t0 = time.time()
        dense_results = dense_retrieve(bi_model, faiss_idx, passage_ids, query_df)
        timings["Dense"] = time.time() - t0

    # Select results based on pipeline
    if pipeline == "BM25 Only":
        final_results = bm25_results.head(top_k)
    elif pipeline == "Dense Only":
        final_results = dense_results.head(top_k)
    elif pipeline == "Hybrid (CC)":
        t0 = time.time()
        final_results = convex_combination(bm25_results, dense_results, top_n=top_k)
        timings["CC Fusion"] = time.time() - t0
    elif pipeline == "Hybrid (CC) + Cross-Encoder":
        t0 = time.time()
        hybrid_results = convex_combination(bm25_results, dense_results, top_n=100)
        timings["CC Fusion"] = time.time() - t0

        corpus_df = _load_corpus()
        cross_model = _load_crossencoder()
        t0 = time.time()
        final_results = rerank(cross_model, hybrid_results, query_df, corpus_df, top_k=top_k)
        timings["Cross-Encoder"] = time.time() - t0

    # ─── Latency Bar ─────────────────────────────────────────────────────
    timing_str = " → ".join(f"**{k}**: {v*1000:.0f}ms" for k, v in timings.items())
    total = sum(timings.values())
    st.markdown(
        f'<div class="latency-bar">⏱ {timing_str} &nbsp;|&nbsp; Total: <b>{total*1000:.0f}ms</b></div>',
        unsafe_allow_html=True,
    )

    # ─── Results Display ─────────────────────────────────────────────────
    corpus_df = _load_corpus()
    doc_lookup = dict(zip(corpus_df["docno"].astype(str), corpus_df["text"]))

    st.markdown(f"### 📋 Top {len(final_results)} results — *{pipeline}*")

    for _, row in final_results.iterrows():
        docno = str(row["docno"])
        score = float(row["score"])
        rank = int(row.get("rank", 0)) + 1
        text = doc_lookup.get(docno, "*[passage text not in loaded corpus]*")
        highlighted = highlight_terms(text, query)

        st.markdown(
            f"""<div class="result-card">
                <span class="rank-badge">#{rank}</span>
                <span class="score-badge">score: {score:.4f}</span>
                <span class="docno-tag">&nbsp;docno: {docno}</span>
                <p style="margin-top:10px; line-height:1.6;">{highlighted}</p>
            </div>""",
            unsafe_allow_html=True,
        )
else:
    st.info("👆 Type a query above to search across 200,000 MS MARCO passages.")
