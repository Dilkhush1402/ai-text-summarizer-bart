"""
Utility functions for text statistics, input validation, and text chunking.
"""

from __future__ import annotations
import re
from typing import List, Tuple


def count_words(text: str) -> int:
    """
    Count the number of words in a text string.
    
    Args:
        text: Input string.
        
    Returns:
        Number of whitespace-separated non-empty words.
    """
    if not text or not isinstance(text, str):
        return 0
    return len(text.strip().split())


def count_characters(text: str, include_spaces: bool = True) -> int:
    """
    Count the number of characters in a text string.
    
    Args:
        text: Input string.
        include_spaces: Whether to include whitespace characters.
        
    Returns:
        Character count.
    """
    if not text or not isinstance(text, str):
        return 0
    if include_spaces:
        return len(text)
    return len("".join(text.split()))


def calculate_compression_ratio(input_word_count: int, summary_word_count: int) -> float:
    """
    Calculate the compression ratio as:
        (summary_word_count / input_word_count) * 100
        
    Note: Compression ratio reflects the volumetric reduction of text,
    not its informational quality, factual consistency, or readability.
    
    Args:
        input_word_count: Word count of the original source text.
        summary_word_count: Word count of the generated summary.
        
    Returns:
        Compression percentage rounded to 1 decimal place. Returns 0.0 if input is 0.
    """
    if input_word_count <= 0 or summary_word_count < 0:
        return 0.0
    ratio = (summary_word_count / input_word_count) * 100.0
    return round(ratio, 1)


def validate_parameters(min_length: int, max_length: int) -> Tuple[bool, str]:
    """
    Validate sequence generation length constraints.
    
    Args:
        min_length: Minimum token length for the summary.
        max_length: Maximum token length for the summary.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if min_length < 5:
        return False, "Minimum length must be at least 5 tokens."
    if max_length < 10:
        return False, "Maximum length must be at least 10 tokens."
    if min_length > max_length:
        return False, f"Minimum length ({min_length}) cannot exceed maximum length ({max_length})."
    if max_length > 1024:
        return False, "Maximum length cannot exceed model context limit (1024 tokens)."
    return True, ""


def validate_input_text(text: str, min_words: int = 10) -> Tuple[bool, str]:
    """
    Validate input text before passing to the neural summarizer.
    
    Args:
        text: The source text to summarize.
        min_words: Minimum recommended words for meaningful summarization.
        
    Returns:
        Tuple of (is_valid, message).
    """
    if not text or not text.strip():
        return False, "Input text cannot be empty. Please enter or paste some text."
    
    words = count_words(text)
    if words < min_words:
        return False, (
            f"Input text is too short ({words} words). "
            f"Please provide at least {min_words} words for meaningful summarization."
        )
    
    return True, ""


def split_text_into_chunks(text: str, max_chunk_words: int = 400) -> List[str]:
    """
    Split long text into sentence-aware chunks that fit comfortably within
    BART's 1024-token position embedding boundary.
    
    Args:
        text: Raw source text.
        max_chunk_words: Target word limit per chunk (default 400 words ~ 550-600 BPE tokens).
        
    Returns:
        List of text chunks.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []
    
    # Split text into sentences using common punctuation boundaries
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
    if not sentences:
        return [cleaned_text]
        
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = count_words(sentence)
        
        # If single sentence is exceptionally long, split it directly
        if sentence_words > max_chunk_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            # Split giant sentence by word windows
            words = sentence.split()
            for i in range(0, len(words), max_chunk_words):
                chunks.append(" ".join(words[i : i + max_chunk_words]))
            continue
            
        if current_word_count + sentence_words > max_chunk_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

