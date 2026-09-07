"""
Central configuration for the RAG pipeline.
Edit paths/models here instead of hunting through every file.
"""

import os

# --- Storage paths (point these at Google Drive in Colab to persist across sessions) ---
BASE_DIR = os.environ.get("RAG_BASE_DIR", "./data")

CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npy")
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", os.path.join(BASE_DIR, "hf_cache"))

# --- Dataset ---
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# --- Models ---
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

LLM_MODEL_NAME = "Qwen/Qwen3-1.7B"   # swap to "Qwen/Qwen3-0.6B" for CPU-only fallback

# --- Qdrant ---
COLLECTION_NAME = "support_kb"

# --- Retrieval / generation ---
TOP_K = 5
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.3
CHUNK_MAX_WORDS = 120
