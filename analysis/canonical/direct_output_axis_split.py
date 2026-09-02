# -*- coding: utf-8 -*-
"""직접 산출(특허·논문 per-capita) 타깃 위에서 과학문화 SHAP 축 분해 — 인지 vs 태도.

검토 논점(가장 중요한 내부 불일치) 해소:
  "정체성 주장('신호는 태도가 아니라 인지 인적자본')이 GII 종합(순환) 모델의 SHAP에서 도출된다.
   인지축 피처(PISA·고등교육)는 GII 구성요소이고 태도축 피처(WVS)는 아니므로, 인지축 우세가
   부분적으로 by-construction(기계적)일 수 있다. 비순환 직접 산출 타깃에서 재산출하라."

설계 (sci_culture_axis_split.py 와 동일 축 정의·동일 XGBoost·동일 SHAP, 단 동결 데이터·비순환 타깃)
------------------------------------------------------------------------------------------
- 데이터: 동결 analysis/canonical/processed_dataset.csv (44국 패널, WB-live drift 차단).
- 타깃: patents_per_m_log·articles_per_m_log (GII 산출이지만 재정규화 무관·비합성 직접 산출).
- ★순환 차단: GII 산출 피처(patent_apps·journal_articles·hi_tech_export_pct[+lag])를 X 에서 제외
  → 특허/논문 타깃의 자기예측(patents_per_m_log≈log(patent_apps/pop)) 방지. 이것이 '탈순환'의 정신.
- 모델: 정본 XGBoost(mt.DEFAULT_XGB_PARAMS, n_estimators=300, random_state=42). 트리 SHAP 는 단조변환
  불변이라 표준화 불요(분할 기준만 사용).
- SHAP: 정본 sh.compute_shap_values → TreeExplainer.

축 정의 (sci_culture_axis_split.py 와 동일)
  COGNITIVE = sci_literacy, pisa_math, pisa_reading, tertiary_enroll, tertiary_attainment
  ATTITUDE  = sci_trust, tech_acceptance

read-only: 정본·산출물 미변경, stdout + JSON 동결용 요약만.
"""
import warnings

warnings.filterwarnings("ignore")
import json
import sys

import os as _os
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, _os.environ.get("SC_CODE_DIR") or str(_Path(__file__).resolve().parents[2] / "code"))

import numpy as np
import pandas as pd
import xgboost as xgb

import config
import model_trainer as mt
import shap_analyzer as sh

DF = pd.read_csv("analysis/canonical/processed_dataset.csv")

# ── 순환 차단: GII 산출 피처 제외 ─────────────────────────────────────
OUTPUT_FEATS = {
    "patent_apps", "journal_articles", "hi_tech_export_pct",
    "hi_tech_export_pct_lag3", "hi_tech_export_pct_lag5", "hi_tech_export_pct_lag7",
}
# X 후보 = base(ALL_FEATURES) + lag + idx 복합, 산출·타깃 제외
BASE = [c for c in config.ALL_FEATURES if c not in OUTPUT_FEATS]
LAGS = [c for c in DF.columns if "_lag" in c and c not in OUTPUT_FEATS]
IDXS = [c for c in DF.columns if c.startswith("idx_")]
FEAT = BASE + LAGS + IDXS

COGNITIVE = ["sci_literacy", "pisa_math", "pisa_reading", "tertiary_enroll", "tertiary_attainment"]
ATTITUDE = ["sci_trust", "tech_acceptance"]


def base_of(col: str) -> str:
    return col.split("_lag")[0]


PARAMS = {k: v for k, v in mt.DEFAULT_XGB_PARAMS.items() if k not in ("n_estimators", "random_state")}
PARAMS["n_estimators"] = 300
PARAMS["random_state"] = 42


def run(target: str) -> dict:
    sub = DF.dropna(subset=[target]).copy()
    cols = [c for c in FEAT if c in sub.columns]
    X = sub[cols]
    y = sub[target]
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X, y)
    shap_values, _ = sh.compute_shap_values(model, X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = dict(zip(list(X.columns), mean_abs))
    total = float(mean_abs.sum())

    # base 7피처만 (과제 사양)
    cog_b = float(sum(imp[c] for c in COGNITIVE if c in imp))
    att_b = float(sum(imp[c] for c in ATTITUDE if c in imp))
    # lag 포함 완전 귀속
    cog_f = float(sum(v for c, v in imp.items() if base_of(c) in COGNITIVE))
    att_f = float(sum(v for c, v in imp.items() if base_of(c) in ATTITUDE))
    composite = float(imp.get("idx_sci_culture", 0.0))
    sc_sum = cog_f + att_f + composite
    pisa_math = float(sum(v for c, v in imp.items() if base_of(c) == "pisa_math"))

    ratio_b = cog_b / att_b if att_b else float("inf")
    ratio_f = cog_f / att_f if att_f else float("inf")

    print("\n" + "=" * 80)
    print(f"■ 직접 산출 타깃 SHAP 축 분해 — {target}")
    print("=" * 80)
    print(f"  n={len(sub)} · {sub.country.nunique()}국 · 피처 {X.shape[1]}개 (GII 산출 제외)")
    print(f"  과학문화 합 = 모델 전체 SHAP 의 {100*sc_sum/total:.1f}%")
    print(f"  [base 7피처] 인지={100*cog_b/(cog_b+att_b):.1f}% vs 태도={100*att_b/(cog_b+att_b):.1f}% · 비율 {ratio_b:.2f}x")
    print(f"  [lag 포함]   인지={100*cog_f/(cog_f+att_f):.1f}% vs 태도={100*att_f/(cog_f+att_f):.1f}% · 비율 {ratio_f:.2f}x")
    print(f"  모델 전체 대비: 인지 {100*cog_f/total:.2f}% · 태도 {100*att_f/total:.2f}% · PISA수학 {100*pisa_math/total:.2f}%")
    verdict = "인지 우세 유지" if cog_f > att_f else "인지 우세 미유지(태도 우세)"
    print(f"  → {verdict}")

    return dict(
        target=target, n=int(len(sub)), n_features=int(X.shape[1]),
        sci_culture_pct_of_model=round(100 * sc_sum / total, 2),
        cog_pct_within_base=round(100 * cog_b / (cog_b + att_b), 1),
        att_pct_within_base=round(100 * att_b / (cog_b + att_b), 1),
        cog_attitude_ratio_base=round(ratio_b, 2),
        cog_pct_within_lag=round(100 * cog_f / (cog_f + att_f), 1),
        att_pct_within_lag=round(100 * att_f / (cog_f + att_f), 1),
        cog_attitude_ratio_lag=round(ratio_f, 2),
        cog_pct_of_model=round(100 * cog_f / total, 2),
        att_pct_of_model=round(100 * att_f / total, 2),
        pisa_math_pct_of_model=round(100 * pisa_math / total, 2),
        cognitive_dominates=bool(cog_f > att_f),
    )


print("=" * 80)
print("직접 산출 타깃 위 과학문화 SHAP 인지/태도 축 재분해 (비순환 검증)")
print("=" * 80)
print(f"X 피처({len(FEAT)}개 후보): GII 산출(patent_apps·journal_articles·hi_tech) 제외")
res = [run("patents_per_m_log"), run("articles_per_m_log")]

print("\n" + "=" * 80)
print("[JSON 동결용 요약]")
print("=" * 80)
print(json.dumps(res, ensure_ascii=False))
print("\n### 직접 산출 축 분해 완료 (read-only)")
