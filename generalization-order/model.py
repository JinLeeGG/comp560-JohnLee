"""
From-scratch micro-transformer for the generalization experiments.

A standard decoder-style transformer with causal masking (matching the nanoGPT baseline
we reproduce in the verification gate), but with the positional encoding factored out
into a swappable module (see pos_encoding.py). The PE is the only thing that varies
across the Phase-2/3 comparison; everything in this file is shared by all PE modes.

The engine is deliberately TASK-AGNOSTIC: it takes token ids (B, T) and returns logits
(B, T, vocab_size). It makes no assumption that the task is Y/N detection, that the
vocab is 15, or what counts as the "answer token" -- those live in the data, config, and
training/eval code. That is what lets the same engine later serve the relative-order task
unchanged.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from pos_encoding import build_positional_encoding


@dataclass
class MicroTransformerConfig:
    vocab_size: int                 # from data/meta.pkl, never hardcoded
    block_size: int                 # max sequence length the model can attend over
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True               # bias in Linear/LayerNorm layers
    pos_type: str = 'learned'       # 'none' | 'learned' | 'sinusoidal' | 'rope' | 't5'
    causal: bool = True             # True = decoder mask; False = bidirectional attention


class CausalSelfAttention(nn.Module):
    def __init__(self, config, pe):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.causal = config.causal
        # The PE module is owned and registered by the parent MicroTransformer. We keep a
        # NON-registered reference here (wrapped in a tuple so nn.Module doesn't register
        # it as a submodule) -- otherwise a parametric PE like learned-absolute would be
        # duplicated in the state_dict once per layer.
        self._pe = (pe,)
        # causal mask (lower-triangular); sliced to the actual sequence length at runtime
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    @property
    def pe(self):
        return self._pe[0]

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = self.pe.rotate_qk(q, k)                                  # Branch B (rope)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)      # (B, n_head, T, T)
        bias = self.pe.attention_bias(T, x.device)                     # Branch C (t5)
        if bias is not None:
            att = att + bias
        if self.causal:
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                                                    # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LN transformer block."""
    def __init__(self, config, pe):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, pe)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MicroTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # one shared PE instance: it is consulted at Branch A here and handed to every
        # attention layer for Branches B/C. Registered once (as self.pe) so any PE params
        # appear exactly once in the state_dict.
        self.pe = build_positional_encoding(
            config.pos_type, block_size=config.block_size,
            n_embd=config.n_embd, n_head=config.n_head, causal=config.causal)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [Block(config, self.pe) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying (nanoGPT-style): the input embedding and output projection share weights
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        x = self.tok_emb(idx)                # (B, T, n_embd)
        x = self.pe.add_to_embedding(x)      # Branch A (learned, sinusoidal)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)             # (B, T, vocab_size)
        return logits

    def num_params(self):
        """Trainable parameter count (the tied embedding/head is counted once)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
