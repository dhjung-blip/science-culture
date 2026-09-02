# -*- coding: utf-8 -*-
"""단일국가(한국) SHAP local 분해 + 정책 What-if 시나리오 — 44국 실측 표본 위 재계산.
파이프라인 main.py 패턴(run_shap_analysis → run_simulation) 그대로 사용, 표본은 pisa-gii(44국).
read-only: 파이프라인 산출물 미변경, stdout 만.

※ 본 표본은 within FE 가 무의미하므로(§6) 시나리오 delta 는 '횡단 모델 기반 예시'로만 해석한다."""
import warnings
warnings.filterwarnings("ignore")
import sys
import os as _os
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, _os.environ.get("SC_CODE_DIR") or str(_Path(__file__).resolve().parents[1] / "code"))
import xgboost as xgb

import data_collector as d
import preprocessor as p
import model_trainer as mt
import shap_analyzer as sh
import scenario_simulator as sc

CC = d.resolve_country_set("pisa-gii")
raw = d.build_full_dataset(countries=CC, years=d.YEARS, use_worldbank=True, use_fred=False, seed=42)
processed, X, y, feats, pipe, X_raw = p.run_preprocessing(raw, target="innovation_score")
print(f"\n### 본 분석 표본: {processed.loc[y.index, 'country'].nunique()}국 · N={len(y)} · 피처 {len(feats)}\n")

# 최종 모델 (전체 fit — paper_tables 패턴, early stopping 제거)
xgb_params = dict(mt.DEFAULT_XGB_PARAMS)
for _k in ("n_estimators", "random_state", "verbosity", "early_stopping_rounds", "n_jobs"):
    xgb_params.pop(_k, None)
model = xgb.XGBRegressor(n_estimators=300, random_state=42, verbosity=0, **xgb_params)
model.fit(X, y)

# ── 4.5 한국 SHAP Local 병목/강점 ──
shap_results = sh.run_shap_analysis(model, X, processed, focus_country="KOR")
prof = shap_results["country_profile"]
print("\n" + "=" * 74)
print("■ 한국(KOR) SHAP Local — 44국 실측 모델")
print("=" * 74)
mask = processed.loc[X.index, "country"] == "KOR"
kor_idx = processed.loc[X.index[mask], "year"].idxmax()
kor_year = int(processed.loc[kor_idx, "year"])
kor_base = float(model.predict(X.loc[[kor_idx]])[0])
print(f"한국 기준연도 {kor_year} · 예측 혁신점수 {kor_base:.2f}\n")
print("피처별 SHAP (절대값 상위 12 — ▼병목 / ▲강점):")
print(prof.head(12)[["feature", "category", "shap_value", "direction"]].to_string(index=False))

# ── 4.6 정책 시나리오 (4범주 +20%) + 한국 갭 ──
print("\n" + "=" * 74)
print("■ 정책 시나리오 What-if (delta %, 횡단 모델 기반 예시)")
print("=" * 74)
sim = sc.run_simulation(model, X, processed, pipe, shap_results)
print("\n[시나리오 행렬: country × 4범주 시나리오 → delta %]")
print(sim["scenario_matrix"].to_string())
print("\n[한국 vs 선진국(FIN·USA·DEU) 갭 분석]")
print(sim["kor_gap_analysis"].to_string(index=False))
print("\n### 단일국가·시나리오 재계산 완료 (read-only)")
