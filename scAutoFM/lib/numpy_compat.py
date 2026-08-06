"""NumPy 2.x shims for older deps (e.g. wandb<0.15 pulled in via timm).

Import this module before any ``timm`` / ``wandb`` import when using the
``hologenomnics`` conda env (numpy>=2 with an older wandb).
"""
from __future__ import annotations

import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "int_"):
    np.int_ = np.int64  # type: ignore[attr-defined]
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128  # type: ignore[attr-defined]
if not hasattr(np, "bool_"):
    np.bool_ = bool  # type: ignore[attr-defined,misc]
