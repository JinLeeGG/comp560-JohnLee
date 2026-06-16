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

       pos_type     A: add_to_embedding   B: rotate_qk   C: attention_bias
       none         -                     -              -
       learned      learned vector        -              -
       sinusoidal   sin/cos vector        -              -
       rope         -                     rotate q,k     -
       t5           -                     -              relative bias

This milestone fully implements `none` and `learned`; `sinusoidal`, `rope`, and `t5`
are wired in but raise NotImplementedError so the next milestone is fill-in-the-blank.
"""
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

    STUB (next milestone). Wiring is in place; the fill-in is the fixed table
        PE[pos, 2i]   = sin(pos / 10000**(2i / n_embd))
        PE[pos, 2i+1] = cos(pos / 10000**(2i / n_embd))
    precomputed into a (block_size, n_embd) buffer and added to x in add_to_embedding.
    """

    def __init__(self, block_size, n_embd):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd

    def add_to_embedding(self, x):
        raise NotImplementedError("Sinusoidal PE: implement the Vaswani 2017 table (next milestone).")


class RoPE(PositionalEncoding):
    """Rotary PE (Su et al. 2021): rotate q and k by position-dependent angles at Branch B.

    STUB (next milestone). Wiring is in place; the fill-in is the standard rotation of
    each (even, odd) feature pair of q/k by angle theta_i * position.
    """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.head_dim = n_embd // n_head
        self.block_size = block_size

    def rotate_qk(self, q, k):
        raise NotImplementedError("RoPE: implement the standard q/k rotation (next milestone).")


class T5RelativeBias(PositionalEncoding):
    """T5 relative position bias (Raffel et al. 2020): a learned scalar bias per
    (head, bucketed relative distance), added to the attention logits at Branch C.

    STUB (next milestone). Wiring is in place; the fill-in is the relative-position
    bucketing and an nn.Embedding(num_buckets, n_head) producing an (n_head, T, T) bias.
    """

    def __init__(self, n_head, block_size, num_buckets=32):
        super().__init__()
        self.n_head = n_head
        self.block_size = block_size
        self.num_buckets = num_buckets

    def attention_bias(self, T, device):
        raise NotImplementedError("T5 relative bias: implement bucketed relative bias (next milestone).")


def build_positional_encoding(pos_type, *, block_size, n_embd, n_head):
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
        return T5RelativeBias(n_head, block_size)
    raise ValueError(
        f"unknown pos_type {pos_type!r}; expected one of: none, learned, sinusoidal, rope, t5")
