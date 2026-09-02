# -*- coding: utf-8 -*-
"""B. 안정성 — SHAP 카테고리 다중시드 평균±SD + 매개비율 클러스터 부트스트랩 CI + CV 다중시드.
정본 shap_analyzer 재사용(매핑 일치). read-only, /tmp. 44국 실 WB."""
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
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

import data_collector as d
import preprocessor as p
import model_trainer as mt
import shap_analyzer as sh

N_SEED_SHAP = 20      # SHAP 카테고리 다중시드
B_BOOT = 2000         # 매개 클러스터 부트스트랩
N_SEED_CV = 10        # CV R² 다중시드

CC = d.resolve_country_set("pisa-gii")
raw = d.build_full_dataset(countries=CC, years=d.YEARS, use_worldbank=True, use_fred=False, seed=42)
processed, X, y, feats, pipe, X_raw = p.run_preprocessing(raw, target="innovation_score")
groups = processed.loc[y.index, "country"].values
print(f"### 안정성 분석: 44국 N={len(y)} 피처 {len(feats)}  (SHAP {N_SEED_SHAP}시드·부트 {B_BOOT}·CV {N_SEED_CV}시드)\n", flush=True)

xgb_params = dict(mt.DEFAULT_XGB_PARAMS) if hasattr(mt, "DEFAULT_XGB_PARAMS") else {}
for _k in ("n_estimators", "random_state", "verbosity", "early_stopping_rounds", "n_jobs"):
    xgb_params.pop(_k, None)

def shap_categories(Xin, seeds):
    """각 시드 모델→SHAP→정본 카테고리 weight_pct. 카테고리별 시드 분포 반환."""
    acc = {}
    for s in seeds:
        m = xgb.XGBRegressor(n_estimators=300, random_state=s, verbosity=0, **xgb_params)
        m.fit(Xin, y)
        sv, _ = sh.compute_shap_values(m, Xin)
        imp = sh.global_feature_importance(sv, list(Xin.columns))
        cat = sh.category_contribution(imp)
        for _, r in cat.iterrows():
            acc.setdefault(r["category"], []).append(r["weight_pct"])
    return {k: (np.mean(v), np.std(v)) for k, v in acc.items()}

# ── T5a. SHAP 카테고리 안정성 — raw (전체 피처) ──
print("=" * 70)
print(f"■ T5(raw) SHAP 카테고리 기여 — {N_SEED_SHAP}시드 평균±SD (전체 피처)")
print("=" * 70)
raw_stab = shap_categories(X, range(N_SEED_SHAP))
for c in ["물적규모", "과학문화", "보상수준", "운영구조"]:
    if c in raw_stab:
        mu, sd = raw_stab[c]; print(f"  {c:8} {mu:5.1f} ± {sd:.1f}%")

# ── T5b. SHAP 카테고리 안정성 — 통제 (발전수준+순환 산출물 제거) ──
CONTROL_DROP = ["gdp_per_capita", "patent_apps", "journal_articles", "hi_tech_export_pct", "idx_hard_rd"]
CONTROL_DROP += [c for c in X.columns if c.startswith("hi_tech_export_pct_lag")]
X_ctrl = X.drop(columns=[c for c in CONTROL_DROP if c in X.columns])
print(f"\n■ T5(통제) — 발전수준(gdp)+순환 산출물(특허·논문·첨단수출+복합) 제거 후 ({X_ctrl.shape[1]}피처)")
ctrl_stab = shap_categories(X_ctrl, range(N_SEED_SHAP))
for c in ["과학문화", "물적규모", "보상수준", "운영구조"]:
    if c in ctrl_stab:
        mu, sd = ctrl_stab[c]; print(f"  {c:8} {mu:5.1f} ± {sd:.1f}%")

# ── T7. 매개비율 클러스터(국가) 부트스트랩 95% CI ──
print("\n" + "=" * 70)
print(f"■ T7 매개비율 — 국가 클러스터 부트스트랩 {B_BOOT}회 (소양→연구원밀도→혁신)")
print("=" * 70)
med = processed.loc[y.index, ["country", "sci_literacy", "researchers_per_m", "innovation_score"]].dropna()
grp = {c: g for c, g in med.groupby("country")}
ctry = list(grp)
rng = np.random.default_rng(42)
def zcoef(df, ycol, xcols):
    z = df.copy()
    for col in set(xcols + [ycol]):
        sd = z[col].std()
        z[col] = (z[col] - z[col].mean()) / (sd if sd else 1)
    fit = sm.OLS(z[ycol], sm.add_constant(z[xcols])).fit()
    return fit.params
ratios, indirects = [], []
for b in range(B_BOOT):
    samp = rng.choice(ctry, len(ctry), replace=True)
    boot = pd.concat([grp[c] for c in samp], ignore_index=True)
    try:
        a = zcoef(boot, "researchers_per_m", ["sci_literacy"])["sci_literacy"]
        m3 = zcoef(boot, "innovation_score", ["sci_literacy", "researchers_per_m"])
        c_tot = zcoef(boot, "innovation_score", ["sci_literacy"])["sci_literacy"]
        ind = a * m3["researchers_per_m"]
        indirects.append(ind)
        if c_tot != 0:
            ratios.append(100 * ind / c_tot)
    except Exception:
        continue
ratios = np.array(ratios)
print(f"  매개비율: 평균 {ratios.mean():.1f}% · 중앙 {np.median(ratios):.1f}% · "
      f"95% CI [{np.percentile(ratios,2.5):.1f}, {np.percentile(ratios,97.5):.1f}]%  (n={len(ratios)})")
ind = np.array(indirects)
print(f"  간접효과 a·b: 평균 {ind.mean():+.3f} · 95% CI [{np.percentile(ind,2.5):+.3f}, {np.percentile(ind,97.5):+.3f}] "
      f"→ 0 포함? {'예(비유의)' if np.percentile(ind,2.5)<0<np.percentile(ind,97.5) else '아니오(유의)'}")

# ── T3/T4. CV R² 다중시드 안정성 ──
print("\n" + "=" * 70)
print(f"■ T3/T4 XGBoost CV R² — {N_SEED_CV}시드 안정성 (GroupKFold 국가)")
print("=" * 70)
cv_r2s, cv_rmses = [], []
for s in range(N_SEED_CV):
    oof = np.full(len(y), np.nan)
    for tr, va in GroupKFold(5).split(X, y, groups):
        m = xgb.XGBRegressor(n_estimators=300, random_state=s, verbosity=0, **xgb_params)
        m.fit(X.iloc[tr], y.iloc[tr]); oof[va] = m.predict(X.iloc[va])
    cv_r2s.append(r2_score(y, oof)); cv_rmses.append(np.sqrt(mean_squared_error(y, oof)))
print(f"  OOF R²: {np.mean(cv_r2s):.3f} ± {np.std(cv_r2s):.3f}  (범위 {min(cv_r2s):.3f}–{max(cv_r2s):.3f})")
print(f"  RMSE  : {np.mean(cv_rmses):.3f} ± {np.std(cv_rmses):.3f}")
print("\n### B 안정성 완료 (read-only)")
