"""
Evaluation module providing ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L).

Implements standard ROUGE scoring (Lin, 2004) without external dependencies,
calculating Precision, Recall, and F1 for unigram, bigram, and longest common subsequence.
"""

from __future__ import annotations
import re
from collections import Counter
from typing import Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase alphanumeric tokens.
    """
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Generate n-gram counts from token sequence.
    """
    if len(tokens) < n or n < 1:
        return Counter()
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


def _compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute length of Longest Common Subsequence (LCS) using space-optimized DP.
    """
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
        
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def calculate_rouge_n(
    candidate: str,
    reference: str,
    n: int = 1
) -> Dict[str, float]:
    """
    Calculate ROUGE-N (n-gram overlap) Precision, Recall, and F1.
    
    Args:
        candidate: The model-generated summary.
        reference: The ground-truth reference summary.
        n: N-gram order (1 for unigrams, 2 for bigrams).
        
    Returns:
        Dict with 'precision', 'recall', and 'f1' scores (between 0.0 and 1.0).
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    
    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    cand_ngrams = _get_ngrams(cand_tokens, n)
    ref_ngrams = _get_ngrams(ref_tokens, n)
    
    total_cand_ngrams = sum(cand_ngrams.values())
    total_ref_ngrams = sum(ref_ngrams.values())
    
    if total_cand_ngrams == 0 or total_ref_ngrams == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    # Clipped overlap count
    overlap = sum(
        min(cand_ngrams[ngram], ref_ngrams[ngram])
        for ngram in cand_ngrams
        if ngram in ref_ngrams
    )
    
    precision = overlap / total_cand_ngrams
    recall = overlap / total_ref_ngrams
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def calculate_rouge_l(
    candidate: str,
    reference: str
) -> Dict[str, float]:
    """
    Calculate ROUGE-L (Longest Common Subsequence) Precision, Recall, and F1.
    
    Args:
        candidate: The model-generated summary.
        reference: The ground-truth reference summary.
        
    Returns:
        Dict with 'precision', 'recall', and 'f1' scores (between 0.0 and 1.0).
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    
    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    lcs_len = _compute_lcs_length(cand_tokens, ref_tokens)
    
    precision = lcs_len / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
    recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_all_rouge_scores(
    candidate: str,
    reference: str
) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores between candidate and reference.
    
    Args:
        candidate: The model-generated summary.
        reference: The ground-truth reference summary.
        
    Returns:
        Nested dict with ROUGE-1, ROUGE-2, ROUGE-L Precision, Recall, and F1.
    """
    return {
        "rouge1": calculate_rouge_n(candidate, reference, n=1),
        "rouge2": calculate_rouge_n(candidate, reference, n=2),
        "rougeL": calculate_rouge_l(candidate, reference),
    }

