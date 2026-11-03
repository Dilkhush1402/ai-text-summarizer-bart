"""
Core neural summarization module using facebook/bart-large-cnn with Hugging Face Transformers.

Implements model caching, length presets, deterministic beam search, token counting,
and safe chunking for inputs exceeding BART's 1024-token context limit.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from src.utils import count_words, split_text_into_chunks, validate_parameters

# Default Hugging Face Model Identifier
MODEL_NAME = "facebook/bart-large-cnn"
MAX_CONTEXT_TOKENS = 1024

# Preset generation configurations
SUMMARY_PRESETS: Dict[str, Dict[str, Any]] = {
    "Short": {
        "min_length": 30,
        "max_length": 60,
        "description": "Concise key takeaways (1-2 sentences)",
        "num_beams": 4,
        "length_penalty": 2.0,
    },
    "Medium": {
        "min_length": 60,
        "max_length": 130,
        "description": "Standard balanced summary (3-4 sentences)",
        "num_beams": 4,
        "length_penalty": 2.0,
    },
    "Long": {
        "min_length": 130,
        "max_length": 250,
        "description": "Detailed multi-paragraph executive overview",
        "num_beams": 4,
        "length_penalty": 2.0,
    },
}


class ModelLoadingError(Exception):
    """Raised when the BART summarization pipeline fails to initialize."""
    pass


def get_torch_device() -> Tuple[str, int]:
    """
    Detect available PyTorch acceleration device.
    
    Returns:
        Tuple of (device_name, device_index_for_transformers_pipeline).
    """
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})", 0
    return "CPU", -1


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    Load tokenizer, seq2seq model, and summarization pipeline.
    
    Attempts offline cached load first to avoid network timeouts in restricted environments;
    falls back to online download if weights are not yet cached.
    """
    device_label, pipeline_device = get_torch_device()
    
    # 1. Try loading from local Hugging Face cache first
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        tokenizer.model_max_length = MAX_CONTEXT_TOKENS
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        summarizer_pipeline = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=pipeline_device,
        )
        return tokenizer, model, summarizer_pipeline, device_label
    except Exception:
        pass

    # 2. Try standard Hub loading
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = MAX_CONTEXT_TOKENS
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer_pipeline = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=pipeline_device,
        )
        return tokenizer, model, summarizer_pipeline, device_label
    except Exception as e:
        raise ModelLoadingError(
            f"Failed to load '{model_name}'. Ensure weights are downloaded or network access is enabled. "
            f"Details: {str(e)}"
        ) from e


def count_bpe_tokens(text: str, tokenizer: Any = None) -> int:
    """
    Calculate the exact number of subword BPE tokens for given text.
    Falls back to word-based subword estimation if tokenizer is unavailable.
    """
    if not text or not text.strip():
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, truncation=False))
        except Exception:
            pass
    # Fallback approximation: 1 word ~ 1.33 subword tokens in English
    return max(1, int(count_words(text) * 1.33))


def generate_summary(
    pipeline_obj: Any,
    tokenizer: Any,
    text: str,
    min_length: int = 60,
    max_length: int = 130,
    num_beams: int = 4,
    length_penalty: float = 2.0,
    no_repeat_ngram_size: int = 3,
    chunk_strategy: str = "Smart Chunking",
) -> Dict[str, Any]:
    """
    Generate an abstractive summary using the BART pipeline with defensive bounds checking.
    
    Args:
        pipeline_obj: Initialized Hugging Face summarization pipeline.
        tokenizer: Initialized AutoTokenizer instance.
        text: Input source text.
        min_length: Minimum summary tokens.
        max_length: Maximum summary tokens.
        num_beams: Number of beams for deterministic beam search.
        length_penalty: Exponential penalty applied to sequence length.
        no_repeat_ngram_size: Size of n-grams that cannot occur repeatedly.
        chunk_strategy: Handling strategy for texts > 1024 tokens ('Smart Chunking' or 'Safe Truncation').
        
    Returns:
        Dict with status, summary_text, token_counts, and chunking metadata.
    """
    is_valid, err_msg = validate_parameters(min_length, max_length)
    if not is_valid:
        return {
            "success": False,
            "summary_text": "",
            "error": err_msg,
            "was_chunked": False,
            "tokens_in": 0,
            "tokens_out": 0,
        }

    tokens_in = count_bpe_tokens(text, tokenizer)
    
    # Common generation kwargs for deterministic beam search
    gen_kwargs = {
        "min_length": min_length,
        "max_length": max_length,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "do_sample": False,
        "early_stopping": True,
    }

    try:
        # Scenario 1: Input text is safely within single-pass context (< 800 tokens)
        if tokens_in <= 800:
            result = pipeline_obj(
                text,
                truncation=True,
                **gen_kwargs,
            )
            summary_text = result[0]["summary_text"].strip()
            tokens_out = count_bpe_tokens(summary_text, tokenizer)
            return {
                "success": True,
                "summary_text": summary_text,
                "error": None,
                "was_chunked": False,
                "chunks_count": 1,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            }

        # Scenario 2: Text exceeds comfortable context limit
        if chunk_strategy == "Safe Truncation":
            result = pipeline_obj(
                text,
                truncation=True,
                max_length=max_length,
                min_length=min_length,
                **{k: v for k, v in gen_kwargs.items() if k not in ["min_length", "max_length"]},
            )
            summary_text = result[0]["summary_text"].strip()
            tokens_out = count_bpe_tokens(summary_text, tokenizer)
            return {
                "success": True,
                "summary_text": summary_text,
                "error": None,
                "was_chunked": False,
                "was_truncated": True,
                "chunks_count": 1,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "note": f"Input ({tokens_in} tokens) was truncated to BART's 1024-token context limit.",
            }

        # Scenario 3: Smart Chunking & Hierarchical Synthesis
        # Split text into chunks (~350 words / ~450-500 tokens)
        chunks = split_text_into_chunks(text, max_chunk_words=350)
        if not chunks:
            chunks = [text]

        chunk_summaries: List[str] = []
        # Target per-chunk length proportional to number of chunks
        per_chunk_max = max(40, min(100, max_length // max(1, len(chunks)) + 25))
        per_chunk_min = max(15, min_length // max(1, len(chunks)))

        for chunk in chunks:
            chunk_res = pipeline_obj(
                chunk,
                truncation=True,
                min_length=per_chunk_min,
                max_length=per_chunk_max,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=False,
            )
            chunk_summaries.append(chunk_res[0]["summary_text"].strip())

        combined_chunk_summary = " ".join(chunk_summaries)
        combined_tokens = count_bpe_tokens(combined_chunk_summary, tokenizer)

        # If combined chunk summaries exceed max_length, perform a second synthesis pass
        if combined_tokens > max_length + 20 and len(chunks) > 1:
            second_pass = pipeline_obj(
                combined_chunk_summary,
                truncation=True,
                **gen_kwargs,
            )
            final_summary = second_pass[0]["summary_text"].strip()
        else:
            final_summary = combined_chunk_summary

        tokens_out = count_bpe_tokens(final_summary, tokenizer)
        return {
            "success": True,
            "summary_text": final_summary,
            "error": None,
            "was_chunked": True,
            "chunks_count": len(chunks),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

    except Exception as e:
        return {
            "success": False,
            "summary_text": "",
            "error": f"Summarization generation error: {str(e)}",
            "was_chunked": False,
            "tokens_in": tokens_in,
            "tokens_out": 0,
        }
