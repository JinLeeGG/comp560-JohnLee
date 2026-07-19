"""
Swappable positional encodings, all behind one interface.

The PE is the single independent variable of the Phase-2/3 comparison: every other
knob (d_model, n_layer, n_head, block_size, data, seed, optimizer) is held fixed and
only `pos_type` changes. To make that clean, each variant is plugged into the model at
exactly one of three branch points, and is a no-op at the other two:

    Branch A  add_to_embedding(x)         absolute PEs add a position vector to x
    Branch B  rotate_qk(q, k)             rotary PE rotates q/k inside attention
    Branch C  attention_bias(T, device)   relative-bias PE adds to the attention logits

Every variant implements all three methods (inheriting the no-op defaults for the ones
it does not use), so model.py can call all three unconditionally and never branch on
`pos_type` itself. Which branch is "live" for a given variant is decided here, in one
place, by which method that subclass overrides.

       pos_type     A: add_to_embedding   B: rotate_qk   C: attention_bias    family
       none         -                     -              -                    (none)
       learned      learned vector        -              -                    absolute
       sinusoidal   sin/cos vector        -              -                    absolute
       rope         -                     rotate q,k     -                    relative
       t5           -                     -              relative bias        relative

All five are implemented. The hypothesis under test (Phase 3): the *relative* family
(rope, t5) and `none` generalize across held-out positions, while the *absolute* family
(learned, sinusoidal) ties itself to training positions and fails to generalize.
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Base interface. The three hooks default to no-ops; a subclass overrides the
    single hook for its branch. This is what lets model.py stay branch-free."""

    def add_to_embedding(self, x):
        """Branch A. x: (B, T, n_embd) -> (B, T, n_embd)."""
        return x

    def rotate_qk(self, q, k):
        """Branch B. q, k: (B, n_head, T, head_dim) -> rotated (q, k)."""
        return q, k

    def attention_bias(self, T, device):
        """Branch C. Return a bias broadcastable to (B, n_head, T, T) to add to the
        attention logits, or None for no bias."""
        return None


class NoPE(PositionalEncoding):
    """No positional information at all: every hook is a no-op (inherited)."""
    pass


class LearnedAbsolute(PositionalEncoding):
    """Learned absolute PE (nanoGPT default): a trainable vector per position, added to
    the token embeddings at Branch A. This is the variant the verification gate uses."""

    def __init__(self, block_size, n_embd):
        super().__init__()
        self.pos_emb = nn.Embedding(block_size, n_embd)

    def add_to_embedding(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)          # (T,)
        return x + self.pos_emb(pos).unsqueeze(0)        # (1, T, n_embd) broadcast over batch


class Sinusoidal(PositionalEncoding):
    """Sinusoidal absolute PE (Vaswani et al. 2017), added at Branch A.

    A fixed (non-learned) table:
        PE[pos, 2i]   = sin(pos / 10000**(2i / n_embd))
        PE[pos, 2i+1] = cos(pos / 10000**(2i / n_embd))
    precomputed once into a (block_size, n_embd) buffer and added to x.
    """

    def __init__(self, block_size, n_embd):
        super().__init__()
        pe = torch.zeros(block_size, n_embd)
        pos = torch.arange(block_size, dtype=torch.float).unsqueeze(1)              # (block_size, 1)
        div = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float)
                        * (-math.log(10000.0) / n_embd))                            # (n_embd/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe, persistent=False)                           # fixed table

    def add_to_embedding(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)                                         # (1, T, n_embd)


def _rotate_half(x):
    """(..., d) -> (..., d) with the two halves rotated: [x1, x2] -> [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(PositionalEncoding):
    """Rotary PE (Su et al. 2021): rotate q and k by position-dependent angles at Branch B.

    Each position t rotates feature pair i by angle t * theta_i, theta_i = base**(-2i/head_dim).
    Implemented with the standard rotate-by-halves trick (q*cos + rotate_half(q)*sin), so the
    dot product q.k depends only on the *relative* offset between the two positions.
    """

    def __init__(self, n_head, n_embd, block_size, base=10000.0):
        super().__init__()
        self.head_dim = n_embd // n_head
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.block_size = block_size
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float)
                                   / self.head_dim))                                # (head_dim/2,)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def _cos_sin(self, T, device):
        t = torch.arange(T, device=device, dtype=torch.float)                       # (T,)
        freqs = torch.outer(t, self.inv_freq.to(device))                           # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)                                     # (T, head_dim)
        return emb.cos()[None, None], emb.sin()[None, None]                         # (1,1,T,head_dim)

    def rotate_qk(self, q, k):
        cos, sin = self._cos_sin(q.size(-2), q.device)
        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot


class T5RelativeBias(PositionalEncoding):
    """T5 relative position bias (Raffel et al. 2020): a learned scalar bias per
    (head, bucketed relative distance), added to the attention logits at Branch C.

    Bucketing uses the standard T5 scheme. In causal/decoder mode, only keys at or before
    the query receive distinct buckets. In bidirectional mode, half of the buckets are
    assigned to keys at/before the query and half to keys after the query. Nearby offsets
    get exact buckets; far offsets share log-spaced buckets. An nn.Embedding(num_buckets,
    n_head) maps each bucket to a per-head scalar, producing an (n_head, T, T) bias.
    """

    def __init__(self, n_head, block_size, num_buckets=32, max_distance=128, causal=True):
        super().__init__()
        if not causal:
            assert num_buckets % 2 == 0, "bidirectional T5 bias needs an even number of buckets"
        self.n_head = n_head
        self.block_size = block_size
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.causal = causal
        self.rel_bias = nn.Embedding(num_buckets, n_head)                           # learned (bucket -> per-head bias)

    @staticmethod
    def _bucket(relative_position, num_buckets, max_distance, causal):
        """relative_position = key_index - query_index.

        Causal mode keeps the original decoder-style T5 scheme: positive relative
        positions are clamped to zero because future keys are masked out anyway.
        Bidirectional mode splits the bucket range by direction, so +d and -d can learn
        different scalar biases.
        """
        if causal:
            rp = -torch.clamp(relative_position, max=0)                             # >= 0 distance back to the key
        else:
            half = num_buckets // 2
            direction = (relative_position > 0).long() * half                      # right/future keys use upper half
            rp = relative_position.abs()
            num_buckets = half

        max_exact = num_buckets // 2
        is_small = rp < max_exact                                                   # exact buckets for near offsets
        large = max_exact + (
            torch.log(rp.float() / max_exact + 1e-6)
            / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        large = torch.minimum(large, torch.full_like(large, num_buckets - 1))       # clamp far offsets
        buckets = torch.where(is_small, rp, large)
        if not causal:
            buckets = buckets + direction
        return buckets

    def attention_bias(self, T, device):
        q_pos = torch.arange(T, device=device)[:, None]                            # (T, 1) query index
        k_pos = torch.arange(T, device=device)[None, :]                            # (1, T) key index
        rel = k_pos - q_pos                                                         # (T, T) = key - query
        buckets = self._bucket(rel, self.num_buckets, self.max_distance, self.causal)  # (T, T)
        bias = self.rel_bias(buckets)                                               # (T, T, n_head)
        return bias.permute(2, 0, 1).unsqueeze(0)                                   # (1, n_head, T, T)


def build_positional_encoding(pos_type, *, block_size, n_embd, n_head, causal=True):
    """Single dispatch point: map a pos_type string to its PE module."""
    if pos_type == 'none':
        return NoPE()
    if pos_type == 'learned':
        return LearnedAbsolute(block_size, n_embd)
    if pos_type == 'sinusoidal':
        return Sinusoidal(block_size, n_embd)
    if pos_type == 'rope':
        return RoPE(n_head, n_embd, block_size)
    if pos_type == 't5':
        return T5RelativeBias(n_head, block_size, causal=causal)
    raise ValueError(
        f"unknown pos_type {pos_type!r}; expected one of: none, learned, sinusoidal, rope, t5")
