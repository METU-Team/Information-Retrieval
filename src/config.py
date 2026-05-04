import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(ROOT_DIR, "index", "bm25_index")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss.index")
PASSAGE_IDS_PATH = os.path.join(EMBEDDINGS_DIR, "passage_ids.npy")

# Dataset
DATASET_NAME = "msmarco-passage/train/triples-small"   # small split for dev
CORPUS_SUBSET_SIZE = 500_000   # number of passages to index (increase if you have resources)
QUERIES_SUBSET_SIZE = 500      # number of queries for evaluation

# Models
BIENCODER_MODEL = "sentence-transformers/msmarco-distilbert-base-v3"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval parameters
BM25_TOP_K = 10         # candidates from BM25
DENSE_TOP_K = 10         # candidates from dense retrieval
RERANK_TOP_K = 10        # candidates to send to re-ranker
FINAL_TOP_K = 10          # final results to return

# Evaluation
METRICS = ["MRR@10", "nDCG@10"]

# FAISS
EMBEDDING_DIM = 768        # msmarco-distilbert-base-v3 output dim
FAISS_BATCH_SIZE = 64

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"