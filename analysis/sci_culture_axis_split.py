# -*- coding: utf-8 -*-
"""과학문화 SHAP 기여 — 인지(COGNITIVE) 축 vs 태도(ATTITUDE) 축 분해.

검토 논점 검증: "과학문화 신호는 대부분 PISA 인지(cognitive)이고 문화(태도)가 아니다."

설계
----
- 데이터: pisa-gii 44국 패널, 실 World Bank 모드 (정본 build_full_dataset).
- 전처리: 정본 preprocessor.run_preprocessing (X 는 이미 StandardScaler 표준화).
- 모델: 정본 XGBoost (mt.DEFAULT_XGB_PARAMS 에서 early_stopping/n_estimators/random_state 제외,
        n_estimators=300, random_state=42) — full X 학습.
- SHAP: 정본 sh.compute_shap_values(model, X) → TreeExplainer.

축 정의 (정본 config.CATEGORY_MAP '과학문화' 7피처를 빠짐없이 둘로 분할)
  COGNITIVE = sci_literacy, pisa_math, pisa_reading, tertiary_enroll, tertiary_attainment
  ATTITUDE  = sci_trust, tech_acceptance

보고: 각 축의 sum(mean(|SHAP|)) 를
  (a) 모델 전체 SHAP 대비 %,
  (b) 과학문화 카테고리 내 % 로 환산.

read-only: 정본·산출물 미변경, stdout 만.
"""
import warnings

warnings.filterwarnings("ignore")
import sys

import os as _os
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, _os.environ.get("SC_CODE_DIR") or str(_Path(__file__).resolve().parents[1] / "code"))

import numpy as np
import xgboost as xgb

import config
import data_collector as d
import model_trainer as mt
import preprocessor as p
import shap_analyzer as sh

# ── 1. 정본 데이터 적재 (pisa-gii, 실 WB) ──────────────────────────────
CC = d.resolve_country_set("pisa-gii")
raw = d.build_full_dataset(
    countries=CC, years=d.YEARS, use_worldbank=True, use_fred=False, seed=42
)
processed, X, y, feats, pipe, X_raw = p.run_preprocessing(raw, target="innovation_score")
groups = processed.loc[y.index, "country"].values

print("\n" + "=" * 78)
print("■ 과학문화 SHAP 축 분해 — COGNITIVE vs ATTITUDE")
print("=" * 78)
print(f"국가 {processed['country'].nunique()}국 · 표본 n={len(X)} · 피처 {X.shape[1]}개 · 타깃 innovation_score")

# ── 2. 정본 XGBoost (early_stopping/n_estimators/random_state 제외, 명시 주입) ──
params = {
    k: v
    for k, v in mt.DEFAULT_XGB_PARAMS.items()
    if k not in ("n_estimators", "random_state")
}
params["n_estimators"] = 300
params["random_state"] = 42
model = xgb.XGBRegressor(**params)
model.fit(X, y)

# ── 3. 정본 SHAP ──────────────────────────────────────────────────────
shap_values, _ = sh.compute_shap_values(model, X)
mean_abs = np.abs(shap_values).mean(axis=0)  # (n_features,)
cols = list(X.columns)
imp = dict(zip(cols, mean_abs))
total = float(mean_abs.sum())

# ── 4. 축 정의 — 정본 '과학문화' 7피처 분할 ───────────────────────────
COGNITIVE = ["sci_literacy", "pisa_math", "pisa_reading", "tertiary_enroll", "tertiary_attainment"]
ATTITUDE = ["sci_trust", "tech_acceptance"]

# 정본 카테고리에서 과학문화 = 어떤 피처인가 (X 에 실제 존재하는 것만)
sci_culture_cols = [c for c in cols if config.CATEGORY_MAP.get(c.split("_lag")[0], config.CATEGORY_MAP.get(c, "기타")) == "과학문화"]
cog_present = [c for c in COGNITIVE if c in cols]
att_present = [c for c in ATTITUDE if c in cols]

# 정합성 점검: 축 합집합이 정본 과학문화 카테고리와 일치하는가
axis_union = set(cog_present) | set(att_present)
sc_set = set(sci_culture_cols)
print("\n[정합성 점검] 정본 '과학문화' 카테고리 피처 :", sorted(sc_set))
print("              인지+태도 축 합집합           :", sorted(axis_union))
if axis_union == sc_set:
    print("              → 일치 ✅ (축 분할이 과학문화 카테고리를 빠짐없이 덮음)")
else:
    print("              → 불일치 ⚠")
    print("                축에 없는 과학문화 피처:", sorted(sc_set - axis_union))
    print("                과학문화 아닌 축 피처  :", sorted(axis_union - sc_set))

# ── 5. 집계 ───────────────────────────────────────────────────────────
cog_sum = float(sum(imp[c] for c in cog_present))
att_sum = float(sum(imp[c] for c in att_present))
sc_sum = float(sum(imp[c] for c in sci_culture_cols))

print("\n" + "-" * 78)
print("[피처별 mean(|SHAP|)] (과학문화 카테고리)")
print("-" * 78)
print(f"{'feature':24}{'axis':10}{'mean|SHAP|':>14}{'% of total':>14}{'% of sci-cult':>16}")
for c in sorted(sci_culture_cols, key=lambda x: -imp[x]):
    ax = "COGNITIVE" if c in cog_present else ("ATTITUDE" if c in att_present else "?")
    print(f"{c:24}{ax:10}{imp[c]:>14.5f}{100*imp[c]/total:>13.2f}%{100*imp[c]/sc_sum:>15.2f}%")

print("\n" + "-" * 78)
print("[축별 합계]")
print("-" * 78)
print(f"{'axis':14}{'sum mean|SHAP|':>16}{'% of MODEL total':>20}{'% within sci-cult':>20}")
print(f"{'COGNITIVE':14}{cog_sum:>16.5f}{100*cog_sum/total:>19.2f}%{100*cog_sum/sc_sum:>19.2f}%")
print(f"{'ATTITUDE':14}{att_sum:>16.5f}{100*att_sum/total:>19.2f}%{100*att_sum/sc_sum:>19.2f}%")
print(f"{'─ 과학문화 합':14}{sc_sum:>16.5f}{100*sc_sum/total:>19.2f}%{'100.00%':>20}")
print(f"{'모델 전체':14}{total:>16.5f}{'100.00%':>20}")

ratio = cog_sum / att_sum if att_sum else float("inf")
print("\n" + "=" * 78)
print("[결론 — 기본: base 7피처만 (과제 사양 그대로)]")
print("=" * 78)
print(f"과학문화 카테고리(lag·복합 포함) = 모델 전체 SHAP 의 {100*sc_sum/total:.1f}%")
print(f"  base 7피처 중 인지(COGNITIVE)={100*cog_sum/sc_sum:.1f}% · 태도(ATTITUDE)={100*att_sum/sc_sum:.1f}%")
print(f"  (나머지 {100*(sc_sum-cog_sum-att_sum)/sc_sum:.1f}% = lag·복합 — 아래 robustness 에서 귀속)")
print(f"  모델 전체 대비: 인지 {100*cog_sum/total:.1f}% vs 태도 {100*att_sum/total:.1f}% · 인지/태도 비율 = {ratio:.2f}x")

# ── 6. Robustness — 과학문화 전체(lag·복합 포함) 완전 귀속 ──────────────
# preprocessing 이 만든 _lag3 와 idx_sci_culture 까지 축에 배정해 카테고리를 100% 덮는다.
#   · _lag3 피처: base 피처의 시차판 → base 와 같은 축.
#   · idx_sci_culture: 복합지수(인지+태도 혼합) → 어느 축에도 깨끗이 안 들어감 → 별도 버킷.
def base_of(col):
    return col.split("_lag")[0]

cog_full, att_full, composite = 0.0, 0.0, 0.0
for c in sci_culture_cols:
    b = base_of(c)
    if c == "idx_sci_culture":
        composite += imp[c]
    elif b in COGNITIVE:
        cog_full += imp[c]
    elif b in ATTITUDE:
        att_full += imp[c]
    else:
        composite += imp[c]

ratio_full = cog_full / att_full if att_full else float("inf")
# 복합지수 제외 기준(인지+태도만)으로도 비율 산출
denom_ex = cog_full + att_full
print("\n" + "=" * 78)
print("[Robustness — 과학문화 전체(lag·복합 포함) 완전 귀속]")
print("=" * 78)
print(f"{'bucket':22}{'sum mean|SHAP|':>16}{'% of MODEL':>14}{'% within sci-cult':>20}")
print(f"{'COGNITIVE(+lag)':22}{cog_full:>16.5f}{100*cog_full/total:>13.2f}%{100*cog_full/sc_sum:>19.2f}%")
print(f"{'ATTITUDE':22}{att_full:>16.5f}{100*att_full/total:>13.2f}%{100*att_full/sc_sum:>19.2f}%")
print(f"{'idx_sci_culture(복합)':22}{composite:>16.5f}{100*composite/total:>13.2f}%{100*composite/sc_sum:>19.2f}%")
print(f"  인지/태도 비율(lag 포함) = {ratio_full:.2f}x")
print(f"  복합지수 제외 시 — 인지 {100*cog_full/denom_ex:.1f}% vs 태도 {100*att_full/denom_ex:.1f}% (둘의 합 기준)")

print("\n" + "=" * 78)
print("[검토 논점 주장 판정]")
print("=" * 78)
verdict = "일치 — 과학문화 신호는 인지(PISA·교육) 축이 압도적으로 우세" if cog_full > att_full else "불일치 — 태도 축이 우세"
print(f"  {verdict}")
print(f"  base만:        인지 {100*cog_sum/sc_sum:.0f}% vs 태도 {100*att_sum/sc_sum:.0f}% (비율 {ratio:.1f}x)")
print(f"  lag 포함:      인지 {100*cog_full/sc_sum:.0f}% vs 태도 {100*att_full/sc_sum:.0f}% (비율 {ratio_full:.1f}x)")
print("\n### 축 분해 완료 (read-only)")
