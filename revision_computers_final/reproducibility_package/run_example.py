from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent
REVISION = PACKAGE.parent
sys.path.insert(0, str(REVISION / "scripts"))

from baselines import constant_half  # noqa: E402
from geometry_metrics import compute_geometry, summarize_interior  # noqa: E402


example = json.loads((PACKAGE / "example_trajectory.json").read_text(encoding="utf-8"))
ratios = np.asarray(example["target_ratios"], dtype=float)
result = compute_geometry(np.asarray(example["vectors"], dtype=float), ratios)
summary = summarize_interior(result)
legacy = np.asarray([0.1, 0.25, 0.5, 0.75, 0.9])
sixths = ratios[(ratios > 0) & (ratios < 1)]
summary["constant_half_legacy_mae"] = float(np.abs(constant_half(len(legacy)) - legacy).mean())
summary["constant_half_sixths_mae"] = float(np.abs(constant_half(len(sixths)) - sixths).mean())
print(json.dumps(summary, indent=2))
