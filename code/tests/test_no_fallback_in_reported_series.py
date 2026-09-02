"""Guard test: no fallback-generated values in any series a reported result uses.

``code/data_collector.py`` contains a fallback generator that produces
placeholder series for constructs with no harmonized international source. The
production path overwrites these with observed values for every country the
corresponding source covers. Two attitudinal columns retain placeholder values
for nine countries; this is disclosed in Section 3.2 of the manuscript.

This test asserts that no *other* series carries placeholder values, so that a
reader can verify the containment claim mechanically rather than taking it on
trust. Placeholder retention has an unambiguous signature in the
pre-interpolation file: the fallback generator emits a value for all twelve
panel years, whereas the real sources are periodic (PISA: four assessment
cycles) or wave-based (WVS: one or two waves).

Run with:  pytest code/tests/test_no_fallback_in_reported_series.py -v
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

RAW = Path(__file__).resolve().parents[2] / "analysis" / "canonical" / "raw_dataset.csv"

# Columns whose source is periodic, so full 12-year coverage would indicate a
# retained placeholder series rather than observed data.
PERIODIC = ["sci_literacy", "pisa_math", "pisa_reading"]

# Columns where placeholder retention is known, documented, and confined.
KNOWN_RETENTION = {
    "sci_trust": 9,
    "tech_acceptance": 9,
}

# Columns removed from the released files entirely.
REMOVED = [
    "self_efficacy",
    "avg_proj_duration",
    "admin_burden_ratio",
    "researcher_retire_age",
    "competitiveness",
]


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    if not RAW.is_file():
        pytest.skip(f"frozen pre-interpolation file not found at {RAW}")
    return pd.read_csv(RAW)


def _full_coverage_countries(df: pd.DataFrame, column: str) -> list[str]:
    counts = df.groupby("country")[column].apply(lambda s: s.notna().sum())
    return sorted(counts[counts == 12].index)


def test_pisa_columns_carry_no_placeholder_values(raw: pd.DataFrame) -> None:
    for column in PERIODIC:
        offenders = _full_coverage_countries(raw, column)
        assert offenders == [], (
            f"{column} shows full 12-year coverage for {offenders}, which PISA's "
            "four-cycle schedule cannot produce; this indicates retained placeholder values."
        )


def test_pisa_observed_anchor_share_matches_manuscript(raw: pd.DataFrame) -> None:
    share = raw["sci_literacy"].notna().sum() / len(raw)
    assert 0.305 <= share <= 0.310, f"expected the 30.7% reported in Section 3.2, got {share:.4f}"


def test_attitudinal_retention_is_exactly_as_disclosed(raw: pd.DataFrame) -> None:
    for column, expected in KNOWN_RETENTION.items():
        offenders = _full_coverage_countries(raw, column)
        assert len(offenders) == expected, (
            f"{column}: manuscript Section 3.2 discloses {expected} countries with retained "
            f"placeholder values, found {len(offenders)}: {offenders}"
        )


def test_attitudinal_observed_anchor_share_matches_manuscript(raw: pd.DataFrame) -> None:
    counts = raw.groupby("country")["sci_trust"].apply(lambda s: s.notna().sum())
    observed = int(counts[counts < 12].sum())
    share = observed / len(raw)
    assert observed == 33, f"expected 33 observed WVS anchor cells, got {observed}"
    assert abs(share - 0.0625) < 1e-4, f"expected the 6.25% reported in Section 3.2, got {share:.4f}"


def test_removed_columns_are_absent_from_released_file(raw: pd.DataFrame) -> None:
    still_present = [c for c in REMOVED if c in raw.columns]
    assert still_present == [], (
        f"columns {still_present} should have been removed before release; "
        "see COLUMN_REMOVAL_LOG.md"
    )
