"""§7.7 특허 within-FE 재검 — 공유 인구 분모(denominator) 교란 제거 후 sci_literacy within β 재추정.

read-only 탐색: 정본·산출물(xai_results.json/*.png/*.csv) 미변경, stdout 만.

문제의식 (findings §7.7 caveat (a)):
  원본 타깃 patents_per_m_log = log1p(patent_apps / population) 은 researchers_per_m 과
  **인구(population)를 분모로 공유**한다 → ratio correlation 으로 within β 가 인공적으로
  부풀려졌을 수 있다. §7.7 은 "within FE 는 국가 더미가 인구 *수준*을 흡수하므로 강건"이라
  주장한다. 본 스크립트는 그 주장을 **정면 검증**한다:

  타깃을 RAW count 로그 log1p(patent_apps) 로 바꾸고 (분모 공유 제거),
  log(population) [WB SP.POP.TOTL] 를 우변 통제로 명시적으로 넣어 인구 규모를
  *연속 통제*한 뒤, 국가 더미 + 클러스터 SE 로 sci_literacy within β 를 재추정한다.
  변이형으로 researchers_per_m 까지 추가 통제한 추정도 보고한다.

원본 대조치: 정본 KNN 경로(pisa-gii 44국, --target patents_per_m_log, n=427)
            특허 within β = +0.349*** (p=0.0006).  [findings §7.7 line 250]

핵심 질문: 분모를 공유하지 않을 때 within 효과가 살아남는가?

방법론 정합:
  · 표본·전처리(국가내 보간 + fold-밖 동등 KNN/median/StandardScaler)는 **정본 run_preprocessing
    (target=patents_per_m_log)** 을 그대로 사용 → +0.349 가 나온 그 n=427 표본·표준화 footing 과 동일.
  · sci_literacy / researchers_per_m 는 정본 X(StandardScaler 적용)에서 가져온다.
  · 타깃·log(population) 은 분석표본에서 z-score 표준화(원본이 sci_literacy·타깃을 표준화한 것과 정합 —
    표준화 β 끼리 직접 비교 가능).
  · 국가 더미 FE + 국가 클러스터 강건 SE (원본 within 추정과 동일 사양).
"""
import warnings

warnings.filterwarnings("ignore")
import sys

import os as _os
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, _os.environ.get("SC_CODE_DIR") or str(_Path(__file__).resolve().parents[1] / "code"))
import data_collector as d
import numpy as np
import pandas as pd
import preprocessor as p
import statsmodels.api as sm

TARGET_ORIG = "patents_per_m_log"


def star(pv: float) -> str:
    return "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else "n.s."


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


# ── 1. 정본 데이터 + 전처리 (target=patents_per_m_log → +0.349 가 나온 표본·footing) ──
CC = d.resolve_country_set("pisa-gii")
print(f"### 국가셋 pisa-gii: {len(CC)}국")
raw = d.build_full_dataset(countries=CC, years=d.YEARS, use_worldbank=True, use_fred=False, seed=42)
assert "patents_per_m_log" in raw.columns, "탈순환 타깃 부재 — 실 WB 모드 필요"
assert "patent_apps" in raw.columns and "population" in raw.columns, "raw count/인구 부재"

processed, X, y, feats, pipe, X_raw = p.run_preprocessing(raw, target=TARGET_ORIG)
groups = processed.loc[y.index, "country"].values
print(f"### 정본 전처리(target={TARGET_ORIG}): X={X.shape}, n={len(y)}, "
      f"{pd.Series(groups).nunique()}국  (← +0.349 가 나온 표본)")
assert "sci_literacy" in X.columns and "researchers_per_m" in X.columns

# X.index 는 raw.index 의 부분집합 (dropna(subset=[target]) 후). 같은 index 로 raw 의
# 원시 count·인구를 정렬해 끌어온다. patents_per_m_log 비결측 행은 정의상 둘 다 비결측.
idx = X.index
patent_apps = raw.loc[idx, "patent_apps"].astype(float)
population = raw.loc[idx, "population"].astype(float)

# 안전망: 잔여 결측/비양수 제거 (log 정의역)
valid = patent_apps.notna() & population.notna() & (population > 0) & (patent_apps >= 0)
n_drop = int((~valid).sum())
if n_drop:
    print(f"  ⚠ 잔여 결측/비양수 {n_drop}행 제외")
idx = idx[valid.values]
X = X.loc[idx]
groups = processed.loc[idx, "country"].values
patent_apps = patent_apps.loc[idx]
population = population.loc[idx]
y_origlog = y.loc[idx]  # patents_per_m_log (원본 타깃)

# ── 2. 변수 구성 ──
sci = zscore(X["sci_literacy"])                     # 정본 표준화 → 분석표본서 재중심
rsr = zscore(X["researchers_per_m"])                # 연구원 밀도 (분모 공유원)
logpop = zscore(np.log(population))                  # log(인구) 연속 통제
y_raw = zscore(np.log1p(patent_apps))               # ★ RAW count 로그 — 분모 공유 제거
y_orig_z = zscore(y_origlog)                         # 원본 per-capita-log (재현용, 표준화)

# 원본 §7.7 within-FE 사양의 통제 집합(decircle_experiment 패턴: sci_literacy + RND_INPUT + DEV).
# +0.349 를 충실히 재현하려면 이 통제와 함께 추정해야 한다.
rd_gdp = zscore(X["rd_gdp_pct"])
gdp_pc = zscore(X["gdp_per_capita"])
RND_DEV = {"rd_gdp_pct": rd_gdp, "researchers_per_m": rsr, "gdp_per_capita": gdp_pc}

g = groups
Dc = pd.get_dummies(pd.Series(g), prefix="c", drop_first=True).astype(float).reset_index(drop=True)

print(f"### 분석 표본: n={len(idx)}, {pd.Series(g).nunique()}국")
print(f"### sci_literacy ↔ log(population) 상관(pooled): r={np.corrcoef(sci, logpop)[0,1]:+.3f}")
# within(국가평균 제거) 상관 — FE 가 보는 실제 공선성
sci_w = sci.values - pd.Series(sci.values).groupby(g).transform("mean").values
lp_w = logpop.values - pd.Series(logpop.values).groupby(g).transform("mean").values
print(f"### sci_literacy ↔ log(population) 상관(within, 국가평균 제거): "
      f"r={np.corrcoef(sci_w, lp_w)[0,1]:+.3f}")
print()


def within_fe(yvar: pd.Series, controls: dict, label: str) -> tuple[float, float]:
    """국가 더미 FE + 클러스터 SE 로 sci_literacy within β 추정."""
    parts = {"sci_literacy": sci.reset_index(drop=True)}
    parts.update({k: v.reset_index(drop=True) if isinstance(v, pd.Series)
                  else pd.Series(np.asarray(v)) for k, v in controls.items()})
    Xfe = pd.concat([pd.DataFrame(parts), Dc], axis=1)
    m = sm.OLS(pd.Series(np.asarray(yvar)).reset_index(drop=True),
               sm.add_constant(Xfe)).fit(cov_type="cluster", cov_kwds={"groups": g})
    b = m.params["sci_literacy"]
    pv = m.pvalues["sci_literacy"]
    se = m.bse["sci_literacy"]
    extra = "  [통제: " + ", ".join(controls.keys()) + "]" if controls else "  [통제: 없음]"
    print(f"  {label:54} sci β = {b:+.3f} {star(pv):>4}  (SE={se:.3f}, p={pv:.4f}){extra}")
    return b, pv


print("=" * 100)
print(f"■ A. 재현 (sanity) — 원본 타깃 patents_per_m_log (분모 공유 O), 정본 표본 n={len(idx)}")
print("=" * 100)
print("   목표: findings §7.7 의 +0.349*** 를 이 표본·footing 에서 복원하는지 확인")
b_orig_pc, p_orig_pc = within_fe(y_orig_z, {}, "per-capita-log(z), 통제 없음 (단순)")
b_orig_full, p_orig_full = within_fe(
    y_orig_z, RND_DEV, "per-capita-log(z) + RND_INPUT+GDP (★원본 §7.7 within 사양)")
# 원본 +0.349 는 타깃 비표준화(원단위) 추정으로 보임 → 비표준화 타깃으로도 1회 추정해
# 문서값과 정합 확인(표준화 β 끼리의 본 비교와는 무관, sanity 용).
b_orig_unz, p_orig_unz = within_fe(
    pd.Series(y_origlog.values), RND_DEV,
    "per-capita-log(원단위 비표준화) + RND+GDP (문서 +0.349 정합확인)")
print()

print("=" * 100)
print("■ B. ★ RAW count 타깃 log1p(patent_apps) — 인구 분모 공유 제거")
print("=" * 100)
print("   B0: 인구 통제 없음 (raw count 를 그냥 — within 이 인구 *수준*을 더미로 흡수하는지)")
b_raw0, p_raw0 = within_fe(y_raw, {}, "raw-count-log, 통제 없음")
print("   B1: + log(population) 연속 통제  ← 핵심 검정 (분모를 우변에서 명시 통제)")
b_raw1, p_raw1 = within_fe(y_raw, {"log_population": logpop}, "raw-count-log + log(pop)")
print("   B2: + log(population) + researchers_per_m  ← 분모 공유원까지 추가 통제")
b_raw2, p_raw2 = within_fe(y_raw, {"log_population": logpop, "researchers_per_m": rsr},
                           "raw-count-log + log(pop) + researchers_per_m")
print("   B3: + log(population) + 원본 RND_INPUT+GDP 통제 (원본 §7.7 통제와 정합)")
b_raw3, p_raw3 = within_fe(y_raw, {"log_population": logpop, **RND_DEV},
                           "raw-count-log + log(pop) + RND_INPUT+GDP")
# 문서 +0.349(비표준화 타깃)와 같은 단위로 직접 비교하기 위한 비표준화 raw 타깃 추정.
y_raw_unz = pd.Series(np.log1p(patent_apps.values))
b_raw3u, p_raw3u = within_fe(
    y_raw_unz, {"log_population": pd.Series(np.log(population.values)), **RND_DEV},
    "raw-count-log(원단위) + log(pop) + RND+GDP (문서값과 동일단위)")
print()

# ── 3. 종합 ──
print("=" * 100)
print("■ 종합 — 분모 비공유 시 within 효과 생존 여부")
print("=" * 100)
print("  원본 §7.7 (정본 KNN, 문서값)              : +0.349*** (p=0.0006)")
print("  [표준화 β — 본 비교의 기준 단위]")
print(f"  A.  per-capita-log(z) + RND+GDP (원본 사양)  : {b_orig_full:+.3f} {star(p_orig_full)} (p={p_orig_full:.4f})")
print(f"      (참고: per-capita-log(z) 단순 통제無)     : {b_orig_pc:+.3f} {star(p_orig_pc)} (p={p_orig_pc:.4f})")
print("  [비표준화 β — 문서 +0.349 와 동일 단위 정합]")
print(f"  A'. per-capita-log(원단위) + RND+GDP         : {b_orig_unz:+.3f} {star(p_orig_unz)} (p={p_orig_unz:.4f})  ← ≈문서 +0.349")
print(f"  B3'.raw-count-log(원단위)+log(pop)+RND+GDP    : {b_raw3u:+.3f} {star(p_raw3u)} (p={p_raw3u:.4f})  ← 분모비공유 대조")
print("  [표준화 β — raw 타깃]")
print(f"  B0. raw-count-log, 통제없음               : {b_raw0:+.3f} {star(p_raw0)} (p={p_raw0:.4f})")
print(f"  B1. raw-count-log + log(pop)   ★핵심      : {b_raw1:+.3f} {star(p_raw1)} (p={p_raw1:.4f})")
print(f"  B2. raw-count-log + log(pop) + 연구원밀도  : {b_raw2:+.3f} {star(p_raw2)} (p={p_raw2:.4f})")
print(f"  B3. raw-count-log + log(pop) + RND+GDP    : {b_raw3:+.3f} {star(p_raw3)} (p={p_raw3:.4f})")
print()
survive_b1 = p_raw1 < 0.05 and b_raw1 > 0
survive_b2 = p_raw2 < 0.05 and b_raw2 > 0
survive_b3 = p_raw3 < 0.05 and b_raw3 > 0
print(f"  → B1 (분모 비공유, 인구 통제) within 유의 생존?     "
      f"{'예 (β>0, p<0.05)' if survive_b1 else '아니오'}")
print(f"  → B2 (+연구원밀도 통제) within 유의 생존?           "
      f"{'예 (β>0, p<0.05)' if survive_b2 else '아니오'}")
print(f"  → B3 (원본과 동일 통제집합) within 유의 생존?       "
      f"{'예 (β>0, p<0.05)' if survive_b3 else '아니오'}")
# 동일 통제집합·동일 단위 대조: A'(per-capita 원단위) vs B3'(raw count 원단위) — 분모 공유만 차이
ratio = b_raw3 / b_orig_full if abs(b_orig_full) > 1e-9 else float("nan")
ratio_u = b_raw3u / b_orig_unz if abs(b_orig_unz) > 1e-9 else float("nan")
print(f"  → 크기 변화 B3/A  (표준화 β, 동일 통제)       = {ratio:+.2f}배")
print(f"  → 크기 변화 B3'/A' (원단위 β, 동일 통제·단위)  = {ratio_u:+.2f}배  "
      f"(1.0 ≈ 분모 무영향, <1 ≈ 공유분모가 부풀렸음)")
print("\n### §7.7 특허 within 분모 재검 완료 (read-only)")
