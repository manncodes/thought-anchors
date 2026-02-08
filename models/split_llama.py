"""
Split LLaMA architecture for scalable curvature measurement.

This module implements a hybrid architecture that combines early layers from a
smaller LLaMA model (e.g., 8B) with late layers from a larger LLaMA model
(e.g., 70B), connected by a trainable adapter layer.

Architecture overview:
    Token Embedding (8B vocab) -> Early Layers (8B config) ->
    Adapter Bridge (Linear or MLP, 8B hidden -> 70B hidden) ->
    Late Layers (70B config) -> RMSNorm -> LM Head

The adapter bridge is a learned projection that maps the hidden dimension of
the smaller model to the hidden dimension of the larger model. When mlp=True,
this is a two-layer MLP with ReLU activation; otherwise a single linear layer.

Usage:
    config = SplitLlamaConfig(
        path8b="/path/to/8b",
        path70b="/path/to/70b",
        num_layers_8=4,
        num_layers_70=8,
    )
    model = SplitLlama(config)
    logits = model(input_ids)
"""

import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SplitLlamaConfig:
    """Configuration for the SplitLlama model.

    Attributes:
        path8b: Path to the smaller (8B) model directory with HF LlamaConfig.
        path70b: Path to the larger (70B) model directory with HF LlamaConfig.
        num_layers_8: Number of early layers to use from the 8B model.
        num_layers_70: Number of late layers to use from the 70B model.
        vocab_size: Vocabulary size for token embeddings and LM head.
        hidden_size: Hidden dimension size (updated from model configs at init).
        num_attention_heads: Number of attention heads (updated from model configs at init).
        mlp: If True, use a two-layer MLP adapter instead of a single linear layer.
        use_flash: If True, use flash attention 2; otherwise use SDPA.
    """

    path8b: str = ""
    path70b: str = ""
    num_layers_8: int = 4
    num_layers_70: int = 8
    vocab_size: int = 128256
    hidden_size: int = 4096
    num_attention_heads: int = 32
    mlp: bool = False
    use_flash: bool = False


class SplitLlama(nn.Module):
    """Split LLaMA model combining early layers from a small model with late
    layers from a large model, connected via a learned adapter bridge.

    The forward interface takes token IDs and returns logits:
        forward(idx) -> logits of shape (batch_size, seq_len, vocab_size)
    """

    def __init__(self, config: SplitLlamaConfig):
        super().__init__()
        self.config = config

        if not config.path8b or not config.path70b:
            raise ValueError(
                "path8b and path70b must be provided, pointing to directories "
                "containing HuggingFace LlamaConfig format config files."
            )

        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaRMSNorm,
            LlamaRotaryEmbedding,
        )

        # Load HF configs for the two model sizes
        config_8b = LlamaConfig.from_pretrained(config.path8b)
        config_70b = LlamaConfig.from_pretrained(config.path70b)

        attn_impl = "flash_attention_2" if config.use_flash else "sdpa"
        config_8b._attn_implementation = attn_impl
        config_70b._attn_implementation = attn_impl

        self._config_8b = config_8b
        self._config_70b = config_70b
        self._use_mlp_adapter = config.mlp

        # Update config fields from the loaded HF configs
        config.hidden_size = config_8b.hidden_size
        config.num_attention_heads = config_8b.num_attention_heads

        # Token embeddings (8B hidden size as the input dimension)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config_8b.hidden_size,
            padding_idx=getattr(config_8b, "pad_token_id", None),
        )

        # Early layers (from 8B config)
        self.layers_first = nn.ModuleList(
            [
                LlamaDecoderLayer(config_8b, layer_idx=i)
                for i in range(config.num_layers_8)
            ]
        )

        # Adapter bridge: 8B hidden dim -> 70B hidden dim
        if config.mlp:
            self.adapter_linear_1 = nn.Linear(
                config_8b.hidden_size, config_70b.hidden_size, bias=False
            )
            self.adapter_linear_2 = nn.Linear(
                config_70b.hidden_size, config_70b.hidden_size, bias=False
            )
        else:
            self.adapter = nn.Linear(
                config_8b.hidden_size, config_70b.hidden_size, bias=False
            )

        # Late layers (from 70B config, using the final N layers)
        start_idx_70b = config_70b.num_hidden_layers - config.num_layers_70
        self.layers_last = nn.ModuleList(
            [
                LlamaDecoderLayer(config_70b, layer_idx=start_idx_70b + i)
                for i in range(config.num_layers_70)
            ]
        )

        # Final normalization and language model head
        self.norm = LlamaRMSNorm(
            config_70b.hidden_size, eps=config_70b.rms_norm_eps
        )
        self.lm_head = nn.Linear(
            config_70b.hidden_size, config.vocab_size, bias=False
        )

        # Rotary embeddings for each layer group
        self.rotary_emb_8b = LlamaRotaryEmbedding(config_8b)
        self.rotary_emb_70b = LlamaRotaryEmbedding(config_70b)

    @property
    def model_name(self) -> str:
        """Return the canonical name for this model architecture."""
        return "split-llama"

    def get_num_params(self):
        """Return (total_params, embedding_params) for logging."""
        n_params = sum(p.numel() for p in self.parameters())
        embd_params = (
            self.embed_tokens.weight.numel() + self.lm_head.weight.numel()
        )
        return n_params, embd_params

    def forward(self, idx):
        """Run the split LLaMA forward pass.

        Args:
            idx: Input token IDs, shape (batch_size, seq_len).

        Returns:
            logits: Output logits, shape (batch_size, seq_len, vocab_size).
        """
        device = idx.device
        batch_size, seq_len = idx.shape

        hidden_states = self.embed_tokens(idx)

        position_ids = (
            torch.arange(seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # 8B rotary embeddings
        position_embeddings_8b = self.rotary_emb_8b(hidden_states, position_ids)

        # Process early (8B) layers
        for layer in self.layers_first:
            layer_output = layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings_8b,
            )
            hidden_states = layer_output[0]

        # Adapter bridge: 8B hidden dim -> 70B hidden dim
        if self._use_mlp_adapter:
            hidden_states = torch.relu(self.adapter_linear_1(hidden_states))
            hidden_states = self.adapter_linear_2(hidden_states)
        else:
            hidden_states = self.adapter(hidden_states)

        # 70B rotary embeddings
        position_embeddings_70b = self.rotary_emb_70b(
            hidden_states, position_ids
        )

        # Process late (70B) layers
        for layer in self.layers_last:
            layer_output = layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings_70b,
            )
            hidden_states = layer_output[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
