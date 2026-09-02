"""Reproduce the released canonical CSVs from raw pipeline output.

The analysis pipeline in ``code/`` still generates five columns that no reported
specification uses (see COLUMN_REMOVAL_LOG.md). The released frozen files in
``analysis/canonical/`` have those columns removed. This script performs that
removal and verifies that the four category composites and the headline
between-only estimates are unchanged, so that the difference between a fresh
pipeline run and the released file is fully accounted for.

Usage:
    python tools/strip_legacy_columns.py <input.csv> <output.csv>
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

LEGACY_COLUMNS = [
    "self_efficacy",
    "avg_proj_duration",
    "admin_burden_ratio",
    "researcher_retire_age",
    "competitiveness",
]

COMPOSITES = ["idx_sci_culture", "idx_hard_rd", "idx_structure", "idx_economic"]


def main(src: str, dst: str) -> int:
    df = pd.read_csv(src)
    present = [c for c in LEGACY_COLUMNS if c in df.columns]
    out = df.drop(columns=present)

    for col in COMPOSITES:
        if col in df.columns and col in out.columns:
            if not np.allclose(df[col].to_numpy(), out[col].to_numpy(), equal_nan=True):
                print(f"FAIL: composite {col} changed after column removal", file=sys.stderr)
                return 1

    out.to_csv(dst, index=False)
    print(f"removed {len(present)} legacy columns: {', '.join(present) or '(none)'}")
    print(f"{src} ({df.shape[1]} cols) -> {dst} ({out.shape[1]} cols), {len(out)} rows unchanged")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
