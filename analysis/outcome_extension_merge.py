"""외부 outcome merge + 변환 — PCT/OpenAlex 를 동결 processed_dataset 에 병합.

read-only 입력: analysis/canonical/processed_dataset.csv (동결) + data/external/*.csv.
출력: data/external/outcome_extension_merged.csv (processed output; raw 미수정).
변환(분모: population·researchers_per_m): raw / per_million / log1p_per_million / per_researcher / log1p_raw.
재현: cd <package-root>; python analysis/outcome_extension_merge.py
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

PROC = "analysis/canonical/processed_dataset.csv"
PCT = "data/external/pct_applications.csv"
OPENALEX = "data/external/openalex_top_cited.csv"
MERGED_OUT = "data/external/outcome_extension_merged.csv"

# (outcome_key, count_col, csv_path, label)
OUTCOMES = [
    ("pct", "pct_applications_origin", PCT, "WIPO PCT applications by origin (OECD)"),
    ("openalex_top10", "top10_cited_publications", OPENALEX, "OpenAlex top-10% cited publications"),
    (
        "openalex_total",
        "total_publications",
        OPENALEX,
        "OpenAlex total publications (article|review)",
    ),
]
TRANSFORMS = ["raw", "per_million", "log1p_per_million", "per_researcher", "log1p_raw"]


def file_hash(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def add_transforms(df: pd.DataFrame, key: str, count_col: str) -> pd.DataFrame:
    """outcome 의 5개 변환 컬럼(접두사 {key}__)을 df 에 부착."""
    pop_m = df["population"] / 1e6
    researchers = (df["researchers_per_m"] * pop_m).replace(0, np.nan)
    c = df[count_col]
    df[f"{key}__raw"] = c
    df[f"{key}__per_million"] = c / pop_m
    df[f"{key}__log1p_per_million"] = np.log1p((c / pop_m).clip(lower=0))
    df[f"{key}__per_researcher"] = c / researchers
    df[f"{key}__log1p_raw"] = np.log1p(c.clip(lower=0))
    # 분모분리(denominator separation)용 보조 컬럼
    df[f"{key}__log_raw"] = np.log(c.clip(lower=1))
    df["log_pop"] = np.log(df["population"].clip(lower=1))
    return df


def build_merged() -> tuple[pd.DataFrame, dict[str, str]]:
    proc = pd.read_csv(PROC)
    hashes = {"processed_dataset.csv": file_hash(PROC)}
    pct = pd.read_csv(PCT).rename(columns={"country_iso3": "country"})
    oa = pd.read_csv(OPENALEX).rename(columns={"country_iso3": "country"})
    hashes["pct_applications.csv"] = file_hash(PCT)
    hashes["openalex_top_cited.csv"] = file_hash(OPENALEX)

    merged = proc.merge(
        pct[["country", "year", "pct_applications_origin"]], on=["country", "year"], how="left"
    ).merge(
        oa[["country", "year", "total_publications", "top10_cited_publications"]],
        on=["country", "year"],
        how="left",
    )
    for key, ccol, _path, _lbl in OUTCOMES:
        merged = add_transforms(merged, key, ccol)
    return merged, hashes


def main() -> int:
    merged, hashes = build_merged()
    merged.to_csv(MERGED_OUT, index=False)
    print(f"[OK] merged 저장: {MERGED_OUT} (rows={len(merged)}, cols={merged.shape[1]})")
    for _key, ccol, _p, lbl in OUTCOMES:
        nn = int(merged[ccol].notna().sum())
        ncov = int(merged.loc[merged[ccol].notna(), "country"].nunique())
        print(f"  {lbl:<48} nonnull={nn}/{len(merged)} countries={ncov}")
    print("  input hashes:", {k: v[:12] for k, v in hashes.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
