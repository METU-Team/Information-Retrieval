"""
Bi-encoder dense retrieval using sentence-transformers + FAISS.

Model: msmarco-distilbert-base-v3
  - Trained on MS MARCO with dot-product similarity
  - Use dot_score / inner product (IndexFlatIP), NOT cosine after L2-norm
"""
import os
import numpy as np
import faiss
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.config import (
    BIENCODER_MODEL, FAISS_INDEX_PATH, PASSAGE_IDS_PATH,
    EMBEDDING_DIM, FAISS_BATCH_SIZE, DEVICE, DENSE_TOP_K
)


def load_biencoder() -> SentenceTransformer:
    model = SentenceTransformer(BIENCODER_MODEL, device=DEVICE)
    return model


def encode_corpus(
    model: SentenceTransformer,
    corpus_df: pd.DataFrame,
    batch_size: int = FAISS_BATCH_SIZE,
    save: bool = True
) -> tuple[np.ndarray, list]:
    """
    Encodes all passages. Returns (embeddings_matrix, passage_ids_list).
    Saves to disk if save=True.
    """
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    texts = corpus_df["text"].tolist()
    passage_ids = corpus_df["docno"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # msmarco-distilbert uses dot product, not cosine
    )

    if save:
        np.save(PASSAGE_IDS_PATH, np.array(passage_ids))
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"FAISS index saved: {index.ntotal} vectors")

    return embeddings, passage_ids


def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    passage_ids = np.load(PASSAGE_IDS_PATH, allow_pickle=True).tolist()
    return index, passage_ids


def dense_retrieve(
    model: SentenceTransformer,
    faiss_index,
    passage_ids: list,
    queries_df: pd.DataFrame,
    top_k: int = DENSE_TOP_K,
) -> pd.DataFrame:
    """
    Returns a results DataFrame compatible with PyTerrier: ['qid', 'docno', 'score', 'rank']
    """
    query_embeddings = model.encode(
        queries_df["query"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    scores, indices = faiss_index.search(query_embeddings, top_k)

    rows = []
    for i, qid in enumerate(queries_df["qid"].tolist()):
        for rank, (idx, score) in enumerate(zip(indices[i], scores[i])):
            rows.append({
                "qid": qid,
                "docno": passage_ids[idx],
                "score": float(score),
                "rank": rank,
            })
    return pd.DataFrame(rows)