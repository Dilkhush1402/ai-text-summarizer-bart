"""
Unit tests for ROUGE evaluation metrics (ROUGE-1, ROUGE-2, ROUGE-L).
"""

import unittest
from src.evaluator import (
    calculate_rouge_n,
    calculate_rouge_l,
    compute_all_rouge_scores,
    _compute_lcs_length,
)


class TestEvaluator(unittest.TestCase):

    def test_rouge_identical_texts(self):
        text = "The quick brown fox jumps over the lazy dog."
        res = compute_all_rouge_scores(candidate=text, reference=text)
        
        # Perfect identity must yield 1.0 for ROUGE-1, ROUGE-2, ROUGE-L
        self.assertAlmostEqual(res["rouge1"]["f1"], 1.0, places=3)
        self.assertAlmostEqual(res["rouge2"]["f1"], 1.0, places=3)
        self.assertAlmostEqual(res["rougeL"]["f1"], 1.0, places=3)

    def test_rouge_completely_disjoint(self):
        cand = "Apples oranges bananas pears."
        ref = "Quantum physics gravitational wave astronomy."
        res = compute_all_rouge_scores(candidate=cand, reference=ref)
        
        self.assertEqual(res["rouge1"]["f1"], 0.0)
        self.assertEqual(res["rouge2"]["f1"], 0.0)
        self.assertEqual(res["rougeL"]["f1"], 0.0)

    def test_rouge_empty_inputs(self):
        res1 = compute_all_rouge_scores(candidate="", reference="Some text")
        self.assertEqual(res1["rouge1"]["f1"], 0.0)
        self.assertEqual(res1["rougeL"]["f1"], 0.0)

        res2 = compute_all_rouge_scores(candidate="Some text", reference="")
        self.assertEqual(res2["rouge1"]["f1"], 0.0)
        self.assertEqual(res2["rougeL"]["f1"], 0.0)

        res3 = compute_all_rouge_scores(candidate="", reference="")
        self.assertEqual(res3["rouge1"]["f1"], 0.0)

    def test_rouge1_partial_overlap(self):
        # cand: 4 words ("the", "cat", "sat", "mat")
        # ref: 4 words ("the", "cat", "slept", "bed")
        # common: "the", "cat" (2 matches)
        cand = "the cat sat mat"
        ref = "the cat slept bed"
        scores = calculate_rouge_n(candidate=cand, reference=ref, n=1)
        self.assertEqual(scores["precision"], 0.5)
        self.assertEqual(scores["recall"], 0.5)
        self.assertEqual(scores["f1"], 0.5)

    def test_rouge2_bigram_overlap(self):
        # cand bigrams: [("machine", "learning"), ("learning", "is"), ("is", "great")] -> 3 bigrams
        # ref bigrams: [("machine", "learning"), ("learning", "models"), ("models", "advance")] -> 3 bigrams
        # overlap: ("machine", "learning") -> 1 match
        cand = "machine learning is great"
        ref = "machine learning models advance"
        scores = calculate_rouge_n(candidate=cand, reference=ref, n=2)
        expected_overlap = 1 / 3
        self.assertAlmostEqual(scores["precision"], expected_overlap, places=3)
        self.assertAlmostEqual(scores["recall"], expected_overlap, places=3)

    def test_lcs_computation(self):
        seq1 = ["a", "b", "c", "d", "e"]
        seq2 = ["a", "c", "e"]
        self.assertEqual(_compute_lcs_length(seq1, seq2), 3)

        seq3 = ["x", "y", "z"]
        self.assertEqual(_compute_lcs_length(seq1, seq3), 0)


if __name__ == "__main__":
    unittest.main()

