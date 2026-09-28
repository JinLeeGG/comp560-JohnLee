"""Load the verified task-agnostic transformer from generalization-distd.

The scratchpad experiment changes data, objectives, and inference only.  Loading the
existing engine keeps the model and NoPE implementation byte-for-byte identical to
the validated baseline without maintaining a second copy.
"""
import sys
from pathlib import Path


ENGINE_DIR = Path(__file__).resolve().parents[1] / 'generalization-distd'
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from model import MicroTransformer, MicroTransformerConfig  # noqa: E402,F401


__all__ = ['MicroTransformer', 'MicroTransformerConfig']
