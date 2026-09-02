# -*- coding: utf-8 -*-
"""유효표본·보간비중 감사 — 44국(pisa-gii) 패널의 '실측 앵커 vs 보간/채움' 분해.

read-only 탐색: 정본·산출물 미변경, stdout만.

핵심 질문(이 스크립트가 답하는 것):
  (1) pisa-gii 44국 ISO3 목록.
  (2) sci_literacy·sci_trust·tech_acceptance·relative_salary 4개 변수에 대해
      국가×연도 셀 중 '실측 앵커'(WVS/PISA/ILOSTAT 원천 직접관측) 대 '보간/합성채움'의 비율.
  (3) 이 44국의 OECD/고소득 편향 (149국 gii 셋과 대비).

방법 — 정본 merge 의미를 그대로 사용:
  · build_full_dataset(use_worldbank=True) 의 merge 패턴(data_collector L901-971):
      WVS/PISA/ILOSTAT '보유국'은 해당 컬럼을 전 연도 NaN 으로 비운 뒤 원천 연도만 실 앵커로 채움.
      '미보유국'은 합성(generate_synthetic_data)값을 그대로 유지.
    → 따라서 raw(전처리 전) 의 non-null 셀 = {보유국의 실 앵커} ∪ {미보유국의 합성 잔존}.
  · 진짜 실측 앵커만 분리하려면 각 로더(load_pisa_scores/load_wvs_indicators/load_ilostat_salary)를
    44국×YEARS 로 직접 호출 → (country,year) 앵커 집합을 ground-truth 로 잡고,
    그 집합에 속한 셀만 '실측'으로 카운트한다.
  · preprocessor 의 국가내 interpolate(limit_direction="both")+열중앙값 impute 는 이 앵커들
    '사이/밖' 을 채우므로, 전처리 후 X(528행 전부 채워짐)와 대비하면 보간비중이 드러난다.

산출물 미변경: 어떤 png/json/csv 도 쓰지 않음. stdout 만.
"""
import warnings

warnings.filterwarnings("ignore")
import sys

import os as _os
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, _os.environ.get("SC_CODE_DIR") or str(_Path(__file__).resolve().parents[1] / "code"))
import numpy as np
import pandas as pd

import data_collector as d
import preprocessor as p

VARS = ["sci_literacy", "sci_trust", "tech_acceptance", "relative_salary"]
SRC = {  # 변수 → (원천 라벨, 로더)
    "sci_literacy": ("PISA", d.load_pisa_scores),
    "sci_trust": ("WVS", d.load_wvs_indicators),
    "tech_acceptance": ("WVS", d.load_wvs_indicators),
    "relative_salary": ("ILOSTAT", d.load_ilostat_salary),
}

# ── OECD 회원국 (38개, 2024 기준; OECD.org 회원목록) ──────────────────────────
OECD = {
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST", "FIN",
    "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN", "KOR", "LVA",
    "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN", "ESP",
    "SWE", "CHE", "TUR", "GBR", "USA",
}
# ── World Bank 고소득(HIC) 분류 (FY24, GNI/capita≥$13,846; data.worldbank.org) ──
#    44/149 셋에 등장 가능한 코드 위주의 참조 집합(보수적·널리 합의된 HIC만).
WB_HIGH_INCOME = {
    "AUS", "AUT", "BEL", "CAN", "CHL", "HRV", "CYP", "CZE", "DNK", "EST", "FIN",
    "FRA", "DEU", "GRC", "HKG", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN", "KOR",
    "KWT", "LVA", "LTU", "LUX", "MLT", "NLD", "NZL", "NOR", "OMN", "PAN", "POL",
    "PRT", "QAT", "ROU", "SAU", "SGP", "SVK", "SVN", "ESP", "SWE", "CHE", "TWN",
    "ARE", "GBR", "USA", "URY", "BHR", "BRN", "GUY", "MAC", "NCL", "PLW", "SYC",
    "TTO", "BHS", "BRB", "ANT",
}

print("=" * 92)
print("■ 유효표본·보간비중 감사 — pisa-gii 44국 패널 (read-only)")
print("=" * 92)

# ── (1) 국가셋 해석 ───────────────────────────────────────────────────────────
CC = d.resolve_country_set("pisa-gii")
GII = d.resolve_country_set("gii")
YEARS = d.YEARS
n_years = len(YEARS)
total_cells = len(CC) * n_years

print(f"\n[1] pisa-gii 국가셋: {len(CC)}개국 · 연도 {YEARS[0]}–{YEARS[-1]} ({n_years}년) "
      f"→ 격자 {len(CC)}×{n_years} = {total_cells} 셀/변수")
print(f"    gii 국가셋(대비군): {len(GII)}개국")
print("\n    44국 ISO3:")
for i in range(0, len(CC), 11):
    print("      " + "  ".join(CC[i:i + 11]))

# ── 데이터 빌드 (실 WB) — raw = 전처리 전 merge 결과 ──────────────────────────
print("\n[빌드] build_full_dataset(use_worldbank=True) … (실 WB 로드 — 수 분 소요)")
raw = d.build_full_dataset(countries=CC, years=YEARS, use_worldbank=True,
                           use_fred=False, seed=42)
raw = raw.sort_values(["country", "year"]).reset_index(drop=True)
print(f"    raw shape = {raw.shape}  (행 = {raw.shape[0]} ; 44×12={total_cells} 기대)")

# 격자 무결성 체크
grid_ok = (raw.shape[0] == total_cells) and (raw["country"].nunique() == len(CC))
print(f"    격자 정합: {'OK' if grid_ok else 'MISMATCH'}")

# ── 각 원천 로더를 44국×YEARS 로 직접 호출 → 진짜 실측 앵커 (country,year) 집합 ─
anchor_keys = {}  # 변수 → set[(country, year)]  (원천이 직접 관측한 셀)
loader_cache = {}
for var, (lbl, loader) in SRC.items():
    if lbl not in loader_cache:
        loader_cache[lbl] = loader(CC, YEARS)
    src_df = loader_cache[lbl]
    if src_df.empty or var not in src_df.columns:
        anchor_keys[var] = set()
        continue
    sub = src_df.dropna(subset=[var])[["country", "year"]]
    # 44국·YEARS 범위로 한정(로더가 이미 필터하지만 방어적으로 재확인)
    sub = sub[sub["country"].isin(set(CC)) & sub["year"].isin(set(YEARS))]
    anchor_keys[var] = set(map(tuple, sub.itertuples(index=False, name=None)))

# 원천 보유국(=실 앵커를 가진 국가) 집합
covered_countries = {var: {c for (c, _y) in keys} for var, keys in anchor_keys.items()}

# ── (2) 셀 분해: 실측앵커 / 합성잔존(non-null but not anchor) / NaN(보간대상) ───
raw_keyset = set(map(tuple, raw[["country", "year"]].itertuples(index=False, name=None)))
raw_idx = raw.set_index(["country", "year"])

print("\n" + "=" * 92)
print("[2] 변수별 셀 분해 (528 셀/변수 기준)")
print("=" * 92)
hdr = (f"{'변수':18}{'원천':8}{'실측앵커':>9}{'합성잔존':>9}{'NaN(보간)':>10}"
       f"{'앵커국':>7}{'실측%':>9}{'채워진셀%':>10}")
print(hdr)
print("-" * 92)

summary = {}
for var in VARS:
    lbl = SRC[var][0]
    keys = anchor_keys[var]
    # 실측 앵커 셀 수 = 앵커키 ∩ 격자
    n_anchor = len(keys & raw_keyset)
    # raw 에서 non-null 인 셀 (격자 전체 기준)
    col = raw_idx[var]
    nonnull_keys = set(col.dropna().index)
    n_nonnull = len(nonnull_keys)
    # 합성 잔존 = non-null 인데 실측 앵커가 아닌 셀 (미보유국의 합성값)
    n_synth_retained = len(nonnull_keys - keys)
    # NaN(전처리 interpolate/impute 대상) = 격자 - non-null
    n_nan = total_cells - n_nonnull
    n_cov = len(covered_countries[var])
    pct_anchor = 100.0 * n_anchor / total_cells
    pct_filled = 100.0 * n_nonnull / total_cells
    summary[var] = dict(anchor=n_anchor, synth=n_synth_retained, nan=n_nan,
                        cov=n_cov, pct_anchor=pct_anchor, pct_filled=pct_filled,
                        nonnull=n_nonnull)
    print(f"{var:18}{lbl:8}{n_anchor:>9}{n_synth_retained:>9}{n_nan:>10}"
          f"{n_cov:>7}{pct_anchor:>8.1f}%{pct_filled:>9.1f}%")

print("-" * 92)
print("해설: '실측앵커'=원천이 직접 관측한 (국가,연도) 셀 · '합성잔존'=원천 미보유국의 합성값(raw에 잔존)")
print("      'NaN(보간)'=raw에서 비어 전처리 interpolate(국가내, both)+열중앙값이 채울 셀")
print("      실측%=실측앵커/528 · 채워진셀%=raw non-null/528 (나머지는 전처리가 보간)")

# ── 실측 앵커의 '커버국 한정' 비율도 — 보유국 기준 보간밀도 ────────────────────
print("\n[2b] 원천 보유국 한정 — 보유국 내부에서도 대부분이 보간임을 보이기")
print(f"{'변수':18}{'보유국×12':>11}{'실측앵커':>9}{'보유국내 실측%':>15}{'보유국내 보간%':>15}")
print("-" * 70)
for var in VARS:
    s = summary[var]
    cov_cells = s["cov"] * n_years
    if cov_cells == 0:
        print(f"{var:18}{'0':>11}{'0':>9}{'—':>15}{'—':>15}")
        continue
    pct_real_in_cov = 100.0 * s["anchor"] / cov_cells
    print(f"{var:18}{cov_cells:>11}{s['anchor']:>9}"
          f"{pct_real_in_cov:>14.1f}%{100 - pct_real_in_cov:>14.1f}%")
print("해설: 보유국조차 12년 중 PISA 4주기·WVS 1~2웨이브·ILOSTAT 소수연도만 실측 →")
print("      나머지 연도는 국가내 interpolate(both)로 채워짐(앵커 사이/이후 carry).")

# ── 전처리 후 X 와 대조: 528 전부 채워짐(보간 100% 메움) 확인 ──────────────────
print("\n[2c] 전처리 후 X — 보간/impute 가 결측을 0으로 만드는지 대조")
processed, X, y, feats, pipe, X_raw = p.run_preprocessing(raw, target="innovation_score")
in_X = [v for v in VARS if v in (feats if isinstance(feats, list) else list(feats))]
print(f"    X shape = {X.shape} · y(innovation_score) 유효행 = {len(y)} (GII dropna 후)")
print(f"    감사 4변수 중 모델 피처로 사용: {in_X}")
Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=list(feats))
for var in in_X:
    if var in Xdf.columns:
        nn = int(Xdf[var].notna().sum())
        print(f"      {var:18} 전처리후 non-null = {nn}/{len(Xdf)} "
              f"(결측 0 → 100% 보간/impute로 메움)")

# ── (3) OECD / 고소득 편향: 44국 vs 149국 ────────────────────────────────────
print("\n" + "=" * 92)
print("[3] OECD / 고소득 편향 — pisa-gii(44) vs gii(149)")
print("=" * 92)


def classify(codes):
    s = set(codes)
    n = len(s)
    n_oecd = len(s & OECD)
    n_hic = len(s & WB_HIGH_INCOME)
    return n, n_oecd, n_hic, sorted(s - OECD), sorted(s - WB_HIGH_INCOME)


n44, oecd44, hic44, non_oecd44, non_hic44 = classify(CC)
n149, oecd149, hic149, non_oecd149, non_hic149 = classify(GII)

print(f"{'국가셋':14}{'N':>5}{'OECD수':>8}{'OECD%':>8}{'고소득수':>9}{'고소득%':>9}")
print("-" * 56)
print(f"{'pisa-gii (44)':14}{n44:>5}{oecd44:>8}{100*oecd44/n44:>7.0f}%"
      f"{hic44:>9}{100*hic44/n44:>8.0f}%")
print(f"{'gii (149)':14}{n149:>5}{oecd149:>8}{100*oecd149/n149:>7.0f}%"
      f"{hic149:>9}{100*hic149/n149:>8.0f}%")

print(f"\n  pisa-gii 44 중 비-OECD ({len(non_oecd44)}국): {'  '.join(non_oecd44) or '없음'}")
print(f"  pisa-gii 44 중 비-고소득 ({len(non_hic44)}국): {'  '.join(non_hic44) or '없음'}")
print(f"\n  gii 149 중 비-OECD: {len(non_oecd149)}국 (= {100*len(non_oecd149)/n149:.0f}%) "
      f"— 저·중소득 다수 포함")
print(f"  gii 149 중 비-고소득: {len(non_hic149)}국 (= {100*len(non_hic149)/n149:.0f}%)")
print("\n  → PISA 응시 = OECD/협력국 위주 ⇒ pisa-gii 교집합은 구조적으로 고소득·OECD 편향.")
print("    149국 gii 셋은 저·중소득을 대거 포함 → 두 셋의 CV R² 격차는 표본 구성 차이와 교란.")

print("\n### 감사 완료 (read-only — 산출물 미변경)")
