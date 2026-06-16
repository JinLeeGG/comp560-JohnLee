"""
Stage-1 cheap sanity checks for the from-scratch engine (no training needed).

Run from generalization-detect/:
    ../venv/bin/python test_engine.py

Checks (all must pass before the Stage-2 training gate):
  1. PE interface contract   -- every pos_type exposes the 3 hooks + builder; inactive
                                branches are no-ops; stub modes raise on their active hook.
  2. Forward shape + no NaN  -- learned: (B,T) ids -> (B,T,vocab) logits, finite.
  3. Causal masking holds    -- changing the last token leaves all earlier logits identical.
  4. No duplicated PE params -- learned pos embedding appears exactly once in state_dict.
  5. Param count sanity      -- micro range, well under 1M, near the ~0.79M baseline.
Exits nonzero if any check fails.
"""
import os
import pickle
import sys

import torch

from model import MicroTransformer, MicroTransformerConfig
from pos_encoding import build_positional_encoding

BLOCK_SIZE = 64
N_EMBD = 128
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD

# vocab_size from the data if present, else the known detection value
try:
    with open(os.path.join('data', 'detect', 'meta.pkl'), 'rb') as f:
        VOCAB = pickle.load(f)['vocab_size']
except FileNotFoundError:
    VOCAB = 15

torch.manual_seed(0)
failures = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ''))
    if not ok:
        failures.append(name)


# ----------------------------------------------------------------------------
# 1. PE interface contract
# ----------------------------------------------------------------------------
# For each pos_type, what each branch should do:
#   A (add_to_embedding): 'noop' returns x unchanged | 'change' alters x | 'raise'
#   B (rotate_qk):        'noop' returns q,k unchanged | 'raise'
#   C (attention_bias):   'none' returns None | 'raise'
print("=== 1. PE interface contract ===")
contract = {
    'none':       ('noop',   'noop',  'none'),
    'learned':    ('change', 'noop',  'none'),
    'sinusoidal': ('raise',  'noop',  'none'),
    'rope':       ('noop',   'raise', 'none'),
    't5':         ('noop',   'noop',  'raise'),
}

for pos_type, (a, b, c) in contract.items():
    pe = build_positional_encoding(pos_type, block_size=BLOCK_SIZE, n_embd=N_EMBD, n_head=N_HEAD)
    has_hooks = all(callable(getattr(pe, m, None))
                    for m in ('add_to_embedding', 'rotate_qk', 'attention_bias'))
    check(f"{pos_type}: builds + exposes 3 hooks", has_hooks)

    x = torch.randn(2, 22, N_EMBD)
    q = torch.randn(2, N_HEAD, 22, HEAD_DIM)
    k = torch.randn(2, N_HEAD, 22, HEAD_DIM)

    # Branch A
    if a == 'raise':
        raised = False
        try:
            pe.add_to_embedding(x)
        except NotImplementedError:
            raised = True
        check(f"{pos_type}: add_to_embedding raises NotImplementedError", raised)
    else:
        out = pe.add_to_embedding(x)
        same = torch.equal(out, x)
        check(f"{pos_type}: add_to_embedding {'no-op' if a == 'noop' else 'changes x'}",
              same if a == 'noop' else not same)

    # Branch B
    if b == 'raise':
        raised = False
        try:
            pe.rotate_qk(q, k)
        except NotImplementedError:
            raised = True
        check(f"{pos_type}: rotate_qk raises NotImplementedError", raised)
    else:
        rq, rk = pe.rotate_qk(q, k)
        check(f"{pos_type}: rotate_qk no-op", torch.equal(rq, q) and torch.equal(rk, k))

    # Branch C
    if c == 'raise':
        raised = False
        try:
            pe.attention_bias(22, torch.device('cpu'))
        except NotImplementedError:
            raised = True
        check(f"{pos_type}: attention_bias raises NotImplementedError", raised)
    else:
        check(f"{pos_type}: attention_bias returns None",
              pe.attention_bias(22, torch.device('cpu')) is None)

# none + learned must build AND run a full forward
for pos_type in ('none', 'learned'):
    m = MicroTransformer(MicroTransformerConfig(vocab_size=VOCAB, block_size=BLOCK_SIZE, pos_type=pos_type))
    m.eval()
    with torch.no_grad():
        _ = m(torch.randint(0, VOCAB, (2, 22)))
    check(f"{pos_type}: full model forward runs", True)

# ----------------------------------------------------------------------------
# 2. Forward shape + no NaN  (pos_type='learned')
# ----------------------------------------------------------------------------
print("\n=== 2. Forward shape + no NaN (learned) ===")
model = MicroTransformer(MicroTransformerConfig(vocab_size=VOCAB, block_size=BLOCK_SIZE, pos_type='learned'))
model.eval()
B, T = 8, 22
ids = torch.randint(0, VOCAB, (B, T))
with torch.no_grad():
    logits = model(ids)
check("output shape == (B, T, vocab_size)", tuple(logits.shape) == (B, T, VOCAB),
      f"got {tuple(logits.shape)}, expected {(B, T, VOCAB)}")
check("logits are finite (no NaN/Inf)", bool(torch.isfinite(logits).all()))

# ----------------------------------------------------------------------------
# 3. Causal masking holds (no future-token leakage)
# ----------------------------------------------------------------------------
print("\n=== 3. Causal masking holds ===")
ids2 = ids.clone()
# change the LAST token of every row to a definitely-different id
ids2[:, -1] = (ids[:, -1] + 1) % VOCAB
with torch.no_grad():
    logits2 = model(ids2)
earlier_same = torch.allclose(logits[:, :-1, :], logits2[:, :-1, :], atol=1e-6)
last_changed = not torch.allclose(logits[:, -1, :], logits2[:, -1, :], atol=1e-6)
check("logits at positions 0..T-2 unchanged when last token altered", earlier_same)
check("logits at the last position DO change (sanity: edit had an effect)", last_changed)

# ----------------------------------------------------------------------------
# 4. No duplicated PE params in state_dict (learned)
# ----------------------------------------------------------------------------
print("\n=== 4. No duplicated PE params (learned) ===")
sd = model.state_dict()
pe_keys = [k for k in sd if 'pos_emb' in k]
per_layer = [k for k in sd if 'attn' in k and 'pos_emb' in k]
check("learned pos embedding appears exactly once", len(pe_keys) == 1, f"keys={pe_keys}")
check("no per-layer PE duplication under attn.*", len(per_layer) == 0)

# ----------------------------------------------------------------------------
# 5. Param count sanity
# ----------------------------------------------------------------------------
print("\n=== 5. Param count sanity ===")
n = model.num_params()
check(f"micro range (<1M, near ~0.79M baseline)", 0.5e6 < n < 1e6, f"{n/1e6:.3f}M ({n:,})")

# ----------------------------------------------------------------------------
print("\n" + ("ALL STAGE-1 CHECKS PASSED" if not failures
              else f"STAGE-1 FAILURES ({len(failures)}): {failures}"))
sys.exit(1 if failures else 0)
