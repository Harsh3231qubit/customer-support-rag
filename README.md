# Customer-support-rag

A minimal Retrieval-Augmented Generation pipeline:

```
Documents → Chunking → MiniLM embeddings → Qdrant → Top-k retrieval → Qwen3 → Answer
```

Built on the [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

## Project structure

```
rag-project/
├── src/
│   ├── config.py         # paths, model names, hyperparameters
│   ├── data_loader.py     # load + clean the dataset
│   ├── chunking.py        # split documents into chunks
│   ├── embedder.py        # MiniLM embedding wrapper
│   ├── vector_store.py    # Qdrant build + search
│   ├── generator.py       # Qwen3 loading + generation
│   └── rag_pipeline.py    # retrieve() + generate_answer(), the public API
├── Notebook/
|   └── RAG_customer_Dataset.ipynb  # Experimentation 
├── scripts/
│   ├── build_index.py     # run once to build the index
│   └── query.py           # run to ask a question
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Build the index (run once):**

```bash
python -m scripts.build_index
```

This loads the dataset, cleans and chunks it, embeds it with MiniLM, and
builds a local Qdrant collection under `data/qdrant_db`. It also saves
`chunks.pkl` and `embeddings.npy` to `data/` so you don't have to
re-embed if you rebuild the collection later.

**2. Ask a question:**

```bash
python -m scripts.query "How do I cancel my order?"
```

## Using it from Python directly

```python
from src.rag_pipeline import generate_answer

print(generate_answer("How do I cancel my order?"))
```

## Notes

- Default LLM is `Qwen/Qwen3-1.7B` loaded in 4-bit (needs a GPU + `bitsandbytes`).
  For CPU-only environments, edit `src/generator.py`'s `load_llm(use_4bit=False)`
  call and switch `LLM_MODEL_NAME` in `src/config.py` to `Qwen/Qwen3-0.6B`.

