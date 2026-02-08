"""Tests for the SplitLlama model and SplitLlamaConfig."""

import unittest
from dataclasses import fields
from unittest.mock import patch

import torch.nn as nn

from models.split_llama import SplitLlama, SplitLlamaConfig


class TestSplitLlamaConfig(unittest.TestCase):
    """Test SplitLlamaConfig dataclass creation and defaults."""

    def test_default_values(self):
        cfg = SplitLlamaConfig()
        self.assertEqual(cfg.path8b, "")
        self.assertEqual(cfg.path70b, "")
        self.assertEqual(cfg.num_layers_8, 4)
        self.assertEqual(cfg.num_layers_70, 8)
        self.assertEqual(cfg.vocab_size, 128256)
        self.assertEqual(cfg.hidden_size, 4096)
        self.assertEqual(cfg.num_attention_heads, 32)
        self.assertFalse(cfg.mlp)
        self.assertFalse(cfg.use_flash)

    def test_custom_values(self):
        cfg = SplitLlamaConfig(
            path8b="/tmp/8b",
            path70b="/tmp/70b",
            num_layers_8=2,
            num_layers_70=4,
            vocab_size=32000,
            hidden_size=2048,
            num_attention_heads=16,
            mlp=True,
            use_flash=True,
        )
        self.assertEqual(cfg.path8b, "/tmp/8b")
        self.assertEqual(cfg.path70b, "/tmp/70b")
        self.assertEqual(cfg.num_layers_8, 2)
        self.assertEqual(cfg.num_layers_70, 4)
        self.assertEqual(cfg.vocab_size, 32000)
        self.assertEqual(cfg.hidden_size, 2048)
        self.assertEqual(cfg.num_attention_heads, 16)
        self.assertTrue(cfg.mlp)
        self.assertTrue(cfg.use_flash)

    def test_is_dataclass(self):
        cfg = SplitLlamaConfig()
        field_names = {f.name for f in fields(cfg)}
        expected = {
            "path8b", "path70b", "num_layers_8", "num_layers_70",
            "vocab_size", "hidden_size", "num_attention_heads",
            "mlp", "use_flash",
        }
        self.assertEqual(field_names, expected)


class TestSplitLlamaImport(unittest.TestCase):
    """Test that SplitLlama can be imported from the models package."""

    def test_import_from_package(self):
        from models import SplitLlama, SplitLlamaConfig
        self.assertIsNotNone(SplitLlama)
        self.assertIsNotNone(SplitLlamaConfig)

    def test_class_exists(self):
        self.assertTrue(callable(SplitLlama))
        self.assertTrue(callable(SplitLlamaConfig))


class TestSplitLlamaInit(unittest.TestCase):
    """Test SplitLlama initialization behavior."""

    def test_missing_paths_raises(self):
        cfg = SplitLlamaConfig()
        with self.assertRaises(ValueError):
            SplitLlama(cfg)

    def test_missing_path8b_raises(self):
        cfg = SplitLlamaConfig(path70b="/tmp/70b")
        with self.assertRaises(ValueError):
            SplitLlama(cfg)

    def test_missing_path70b_raises(self):
        cfg = SplitLlamaConfig(path8b="/tmp/8b")
        with self.assertRaises(ValueError):
            SplitLlama(cfg)


class TestSplitLlamaModelName(unittest.TestCase):
    """Test the model_name property using a mocked SplitLlama instance."""

    @patch.object(SplitLlama, "__init__", lambda self, config: None)
    def test_model_name_property(self):
        model = SplitLlama.__new__(SplitLlama)
        self.assertEqual(model.model_name, "split-llama")


class TestSplitLlamaGetNumParams(unittest.TestCase):
    """Test get_num_params using a mocked SplitLlama instance."""

    @patch.object(SplitLlama, "__init__", lambda self, config: nn.Module.__init__(self))
    def test_get_num_params(self):
        model = SplitLlama(config=None)

        # Set up minimal modules so parameters() and get_num_params work
        model.embed_tokens = nn.Embedding(100, 16)
        model.lm_head = nn.Linear(16, 100, bias=False)
        model.adapter = nn.Linear(16, 16, bias=False)

        total, embd = model.get_num_params()
        expected_embed = 100 * 16  # embed_tokens
        expected_lm = 100 * 16    # lm_head
        expected_adapter = 16 * 16  # adapter

        self.assertEqual(total, expected_embed + expected_lm + expected_adapter)
        self.assertEqual(embd, expected_embed + expected_lm)


if __name__ == "__main__":
    unittest.main()
