"""
Wraps Qwen3 loading and answer generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import LLM_MODEL_NAME, HF_CACHE_DIR, MAX_NEW_TOKENS, TEMPERATURE

_tokenizer = None
_model = None


def load_llm(use_4bit: bool = True):
    """
    Loads the Qwen3 tokenizer + model once per process.
    Set use_4bit=False for CPU-only environments (slower, no bitsandbytes needed).
    """
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, cache_dir=HF_CACHE_DIR)

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=HF_CACHE_DIR,
        )
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float32,
            cache_dir=HF_CACHE_DIR,
        )

    return _tokenizer, _model


def generate_from_context(query: str, context_chunks: list[str]) -> str:
    """Generates a grounded answer given a query and retrieved context chunks."""
    tokenizer, model = load_llm()

    context = "\n".join(f"- {c}" for c in context_chunks)

    prompt = f"""Answer the user's question using only the context below.
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {query}
Answer:"""

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
