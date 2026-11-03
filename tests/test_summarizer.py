"""
Unit tests for summarizer presets and configurations.
"""

import unittest
from src.summarizer import SUMMARY_PRESETS, get_torch_device, count_bpe_tokens
from src.utils import validate_parameters


class TestSummarizerConfig(unittest.TestCase):

    def test_presets_completeness_and_validity(self):
        required_presets = ["Short", "Medium", "Long"]
        for preset_name in required_presets:
            self.assertIn(preset_name, SUMMARY_PRESETS)
            cfg = SUMMARY_PRESETS[preset_name]
            self.assertIn("min_length", cfg)
            self.assertIn("max_length", cfg)
            self.assertIn("num_beams", cfg)
            
            # Validate parameters using our validator
            is_valid, msg = validate_parameters(cfg["min_length"], cfg["max_length"])
            self.assertTrue(is_valid, f"Preset '{preset_name}' invalid: {msg}")
            self.assertLess(cfg["min_length"], cfg["max_length"])

    def test_torch_device_detection(self):
        label, dev_idx = get_torch_device()
        self.assertIsInstance(label, str)
        self.assertIn(dev_idx, [0, -1])

    def test_count_bpe_tokens_fallback(self):
        # When tokenizer is None, falls back to word count approximation safely
        tokens = count_bpe_tokens("This is an exemplary sentence for testing fallback.", tokenizer=None)
        self.assertGreater(tokens, 0)
        self.assertEqual(count_bpe_tokens("", tokenizer=None), 0)


if __name__ == "__main__":
    unittest.main()

