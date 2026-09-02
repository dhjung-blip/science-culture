# -*- coding: utf-8 -*-
"""A. 보조 지표 일괄 — 시차·VIF·SHAP Top10·국가별예측·패널FE·한계기여·순환상관·within분해.
정본 함수 재사용. read-only, /tmp. 44국 실 WB (결정적, seed=42)."""
import warnings
warnings.filterwarnings("ignore")
import sys
import os as _os
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, _os.environ.get("SC_CODE_DIR") or str(_Path(__file__).resolve().parents[1] / "code"))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

import data_collector as d
import preprocessor as p
import model_trainer as mt
import shap_analyzer as sh

CC = d.resolve_country_set("pisa-gii")
raw = d.build_full_dataset(countries=CC, years=d.YEARS, use_worldbank=True, use_fred=False, seed=42)
processed, X, y, feats, pipe, X_raw = p.run_preprocessing(raw, target="innovation_score")
groups = processed.loc[y.index, "country"].values
src = processed.loc[y.index]
xp = dict(mt.DEFAULT_XGB_PARAMS) if hasattr(mt, "DEFAULT_XGB_PARAMS") else {}
for _k in ("n_estimators", "random_state", "verbosity", "early_stopping_rounds", "n_jobs"):
    xp.pop(_k, None)
def newxgb(): return xgb.XGBRegressor(n_estimators=300, random_state=42, verbosity=0, **xp)
def cv_r2(Xin):
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(5).split(Xin, y, groups):
        m = newxgb(); m.fit(Xin.iloc[tr], y.iloc[tr]); oof[va] = m.predict(Xin.iloc[va])
    return r2_score(y, oof)
print(f"### A 보조 지표 — 44국 N={len(y)} 피처 {len(feats)}\n", flush=True)

# A3. SHAP 피처 Top10
print("=" * 64); print("■ A3. SHAP 개별 피처 Top10"); print("=" * 64)
m = newxgb(); m.fit(X, y)
sv, _ = sh.compute_shap_values(m, X)
imp = sh.global_feature_importance(sv, list(X.columns))
imp["pct"] = imp["importance"] / imp["importance"].sum() * 100
for i, r in imp.head(10).iterrows():
    print(f"  {r['feature']:26}{r['pct']:6.2f}%  [{r['category']}]")

# A7. 카테고리 한계기여 (전체 − 그 카테고리 제외)
print("\n" + "=" * 64); print("■ A7. 카테고리 한계기여 (CV R² 증분)"); print("=" * 64)
full = cv_r2(X)
print(f"  전체 모델 CV R² = {full:.3f}")
for cat in ["과학문화", "물적규모", "운영구조", "보상수준"]:
    drop = [f for f in X.columns if sh._get_category(f) == cat]
    marg = full - cv_r2(X.drop(columns=drop))
    print(f"  {cat:8} 한계기여 = {marg:+.3f}  ({len(drop)}피처 제외)")

# A2. VIF (base 피처)
print("\n" + "=" * 64); print("■ A2. 다중공선성 VIF (base 피처, 표준화)"); print("=" * 64)
from config import ALL_FEATURES
base = [c for c in ALL_FEATURES if c in X.columns]
Xb = X[base].dropna()
vif = pd.DataFrame({"피처": base,
                    "VIF": [variance_inflation_factor(Xb.values, i) for i in range(len(base))]})
for _, r in vif.sort_values("VIF", ascending=False).head(12).iterrows():
    flag = "🔴" if r["VIF"] > 10 else ("🟡" if r["VIF"] > 5 else "")
    print(f"  {r['피처']:24}{r['VIF']:7.2f}  {flag}")

# A5. 패널 FE 전체 (과학문화 3종 × Pooled/국가FE/국가+연도FE)
print("\n" + "=" * 64); print("■ A5. 패널 회귀 — 과학문화 3종 β (R&D 통제, 클러스터 SE)"); print("=" * 64)
def star(pv): return "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else "n.s."
def zser(s): sd = s.std(); return (s - s.mean()) / (sd if sd else 1)
reg = src[["country", "year", "sci_literacy", "sci_trust", "tech_acceptance",
           "rd_gdp_pct", "researchers_per_m", "gdp_per_capita", "innovation_score"]].dropna().copy()
y_z = zser(reg["innovation_score"])
ctrl = pd.DataFrame({c: zser(reg[c]) for c in ["rd_gdp_pct", "researchers_per_m", "gdp_per_capita"]})
cdum = pd.get_dummies(reg["country"], prefix="c", drop_first=True).astype(float).reset_index(drop=True)
ydum = pd.get_dummies(reg["year"], prefix="y", drop_first=True).astype(float).reset_index(drop=True)
g2 = reg["country"].values
print(f"{'피처':16}{'Pooled':>14}{'국가FE':>14}{'국가+연도FE':>16}")
for f in ["sci_literacy", "sci_trust", "tech_acceptance"]:
    xz = zser(reg[f]).reset_index(drop=True)
    base_X = pd.concat([xz.rename(f), ctrl.reset_index(drop=True)], axis=1)
    out = []
    for spec, extra in [("pool", None), ("cfe", cdum), ("cyfe", pd.concat([cdum, ydum], axis=1))]:
        XX = base_X if extra is None else pd.concat([base_X, extra], axis=1)
        fit = sm.OLS(y_z.reset_index(drop=True), sm.add_constant(XX)).fit(
            cov_type="cluster", cov_kwds={"groups": g2})
        out.append(f"{fit.params[f]:+.2f}{star(fit.pvalues[f])}")
    print(f"{f:16}{out[0]:>14}{out[1]:>14}{out[2]:>16}")

# A6. 순환성 상관 (44국) + findings 표본별 인용
print("\n" + "=" * 64); print("■ A6. 피처-타깃 상관 (순환성, 44국)"); print("=" * 64)
for c in ["researchers_per_m", "rd_gdp_pct", "patent_apps", "journal_articles", "sci_literacy"]:
    if c in src.columns:
        print(f"  {c:22} r(GII) = {src[c].corr(src['innovation_score']):+.3f}")
print("  ※ 표본 확장 시 급등(findings): researchers 0.45→0.84, rd_gdp 0.40→0.80 (20→149국)")

# A8. within 분해 (between/within 분산)
print("\n" + "=" * 64); print("■ A8. GII 분산 분해 (between/within)"); print("=" * 64)
gii = src.groupby("country")["innovation_score"]
overall_var = src["innovation_score"].var()
between_var = gii.mean().var()
within = src["innovation_score"] - gii.transform("mean")
within_var = within.var()
print(f"  전체 분산 {overall_var:.2f} = between {between_var:.2f} ({100*between_var/(between_var+within_var):.0f}%)"
      f" + within {within_var:.2f} ({100*within_var/(between_var+within_var):.0f}%)")
print(f"  국가 내 GII 변동폭(평균 절대편차): ±{within.abs().mean():.2f}점")

# A4. 국가별 OOF 예측 vs 실제 (주요국)
print("\n" + "=" * 64); print("■ A4. 국가별 예측 vs 실제 (OOF, 주요국)"); print("=" * 64)
oof = np.full(len(y), np.nan)
for tr, va in GroupKFold(5).split(X, y, groups):
    mm = newxgb(); mm.fit(X.iloc[tr], y.iloc[tr]); oof[va] = mm.predict(X.iloc[va])
pred_df = pd.DataFrame({"country": groups, "actual": y.values, "pred": oof})
print(f"  {'국가':6}{'실제 평균':>10}{'예측 평균':>10}{'오차':>8}")
for c in ["KOR", "USA", "JPN", "DEU", "CHN", "FIN", "SWE", "GBR"]:
    sub = pred_df[pred_df.country == c]
    if len(sub):
        a, pr = sub.actual.mean(), sub.pred.mean()
        print(f"  {c:6}{a:>10.1f}{pr:>10.1f}{pr-a:>+8.1f}")

# A1. 시차효과 (sci_literacy 현재 vs lag → GII 상관)
print("\n" + "=" * 64); print("■ A1. 시차효과 — sci_literacy 선행 상관"); print("=" * 64)
for lag in [0, 3, 5, 7]:
    col = "sci_literacy" if lag == 0 else f"sci_literacy_lag{lag}"
    if col in src.columns:
        print(f"  lag{lag}: r(현재GII) = {src[col].corr(src['innovation_score']):+.3f}")
print("\n### A 완료 (read-only)")
