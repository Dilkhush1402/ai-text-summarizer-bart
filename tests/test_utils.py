"""
Unit tests for text processing utilities and validation logic.
"""

import unittest
from src.utils import (
    count_words,
    count_characters,
    calculate_compression_ratio,
    validate_parameters,
    validate_input_text,
    split_text_into_chunks,
)


class TestUtils(unittest.TestCase):

    def test_count_words_standard(self):
        text = "Natural language processing powers modern conversational intelligence."
        self.assertEqual(count_words(text), 7)

    def test_count_words_extra_whitespace_and_newlines(self):
        text = "  Word1 \t Word2   \n\n  Word3  "
        self.assertEqual(count_words(text), 3)

    def test_count_words_empty_and_edge_cases(self):
        self.assertEqual(count_words(""), 0)
        self.assertEqual(count_words("   "), 0)
        self.assertEqual(count_words(None), 0)

    def test_count_characters(self):
        text = "Hello World!"
        self.assertEqual(count_characters(text, include_spaces=True), 12)
        self.assertEqual(count_characters(text, include_spaces=False), 11)
        self.assertEqual(count_characters(""), 0)
        self.assertEqual(count_characters(None), 0)

    def test_compression_ratio_standard(self):
        # 30 summary words from 100 input words = 30.0%
        ratio = calculate_compression_ratio(input_word_count=100, summary_word_count=30)
        self.assertEqual(ratio, 30.0)

    def test_compression_ratio_zero_and_edge_cases(self):
        # Zero input words should safely return 0.0 without ZeroDivisionError
        self.assertEqual(calculate_compression_ratio(input_word_count=0, summary_word_count=10), 0.0)
        self.assertEqual(calculate_compression_ratio(input_word_count=100, summary_word_count=0), 0.0)
        self.assertEqual(calculate_compression_ratio(input_word_count=-5, summary_word_count=10), 0.0)
        self.assertEqual(calculate_compression_ratio(input_word_count=100, summary_word_count=-2), 0.0)

    def test_validate_parameters_valid(self):
        is_valid, msg = validate_parameters(min_length=30, max_length=130)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_validate_parameters_min_exceeds_max(self):
        is_valid, msg = validate_parameters(min_length=100, max_length=50)
        self.assertFalse(is_valid)
        self.assertIn("cannot exceed", msg)

    def test_validate_parameters_out_of_bounds(self):
        # min too small
        is_valid, _ = validate_parameters(min_length=2, max_length=50)
        self.assertFalse(is_valid)
        # max too large
        is_valid, _ = validate_parameters(min_length=10, max_length=2000)
        self.assertFalse(is_valid)

    def test_validate_input_text_empty(self):
        is_valid, msg = validate_input_text("")
        self.assertFalse(is_valid)
        self.assertIn("cannot be empty", msg)

        is_valid, msg = validate_input_text("   \n\t  ")
        self.assertFalse(is_valid)

    def test_validate_input_text_too_short(self):
        is_valid, msg = validate_input_text("Only three words.", min_words=10)
        self.assertFalse(is_valid)
        self.assertIn("too short", msg)

    def test_validate_input_text_valid(self):
        text = "This is a sufficiently long sentence designed to meet the minimum word validation threshold."
        is_valid, msg = validate_input_text(text, min_words=10)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_split_text_into_chunks(self):
        sentences = [
            f"Sentence number {i} describes an aspect of the broader problem."
            for i in range(1, 20)
        ]
        text = " ".join(sentences)
        # Using a small chunk word limit to trigger multi-chunk splitting
        chunks = split_text_into_chunks(text, max_chunk_words=30)
        self.assertGreater(len(chunks), 1)
        # Check that rejoined chunks preserve total word content
        total_chunk_words = sum(count_words(c) for c in chunks)
        self.assertEqual(total_chunk_words, count_words(text))


if __name__ == "__main__":
    unittest.main()

