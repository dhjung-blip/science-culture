# -*- coding: utf-8 -*-
"""논문 표 T2(기술통계)·T4(모델 클래스 비교) 실측. read-only, /tmp.
정본 전처리(누수 제거 후 run_preprocessing) 사용 → 본 분석과 정합. 44국 pisa-gii 실 WB."""
import warnings
warnings.filterwarnings("ignore")
import sys
import os as _os
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, _os.environ.get("SC_CODE_DIR") or str(_Path(__file__).resolve().parents[1] / "code"))
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import data_collector as d
import preprocessor as p
import model_trainer as mt

CC = d.resolve_country_set("pisa-gii")
raw = d.build_full_dataset(countries=CC, years=d.YEARS, use_worldbank=True, use_fred=False, seed=42)
processed, X, y, feats, pipe, X_raw = p.run_preprocessing(raw, target="innovation_score")
groups = processed.loc[y.index, "country"].values
n_countries = processed.loc[y.index, "country"].nunique()
yr = processed.loc[y.index, "year"]
print(f"### 본 분석 표본: {n_countries}국 · {yr.min()}–{yr.max()} · N={len(y)} · 피처 {len(feats)}\n")

# ── T2. 기술통계 (원 스케일, processed_df) ──
print("=" * 78)
print("■ T2. 기술통계 (주요 변수, 원 스케일)")
print("=" * 78)
desc_vars = [
    ("innovation_score", "GII 종합점수 (타깃)"),
    ("sci_literacy", "PISA 과학소양"),
    ("pisa_math", "PISA 수학"),
    ("rd_gdp_pct", "R&D/GDP (%)"),
    ("researchers_per_m", "연구원 (백만명당)"),
    ("gdp_per_capita", "1인당 GDP (USD)"),
    ("gov_edu_exp_pct", "교육지출/GDP (%)"),
    ("internet_users_pct", "인터넷 보급"),
    ("relative_salary", "전문직 임금비"),
    ("youth_unemp_pct", "청년 실업률 (%)"),
]
src = processed.loc[y.index]
print(f"{'변수':28}{'N':>5}{'평균':>12}{'표준편차':>12}{'최소':>11}{'최대':>11}")
for col, label in desc_vars:
    if col not in src.columns:
        continue
    s = src[col].dropna()
    print(f"{label:28}{len(s):>5}{s.mean():>12.2f}{s.std():>12.2f}{s.min():>11.2f}{s.max():>11.2f}")

# ── T4. 모델 클래스 비교 (동일 X·동일 GroupKFold) ──
print("\n" + "=" * 78)
print("■ T4. 모델 클래스 비교 (44국, GroupKFold(국가) 5-fold, 표준화 X 동일)")
print("=" * 78)
xgb_params = dict(mt.DEFAULT_XGB_PARAMS) if hasattr(mt, "DEFAULT_XGB_PARAMS") else {}
for _k in ("n_estimators", "random_state", "verbosity", "early_stopping_rounds", "n_jobs"):
    xgb_params.pop(_k, None)
models = {
    "XGBoost (주 모델)": xgb.XGBRegressor(n_estimators=300, random_state=42, verbosity=0, **xgb_params),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    "Linear (OLS)": LinearRegression(),
}
gkf = GroupKFold(n_splits=5)
print(f"{'모델':22}{'CV R² (평균±SD)':>22}{'RMSE':>10}{'MAE':>10}{'OOF R²':>10}")
rows = []
for name, model in models.items():
    r2s, rmses, maes = [], [], []
    oof = np.full(len(y), np.nan)
    for tr, va in gkf.split(X, y, groups):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[va])
        oof[va] = pred
        r2s.append(r2_score(y.iloc[va], pred))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[va], pred)))
        maes.append(mean_absolute_error(y.iloc[va], pred))
    oof_r2 = r2_score(y, oof)
    print(f"{name:22}{f'{np.mean(r2s):+.3f} ± {np.std(r2s):.3f}':>22}"
          f"{np.mean(rmses):>10.3f}{np.mean(maes):>10.3f}{oof_r2:>10.3f}")
    rows.append((name, np.mean(r2s), np.std(r2s), np.mean(rmses), np.mean(maes), oof_r2))

best = max(rows, key=lambda r: r[5])
print(f"\n  → 최고 OOF R²: {best[0]} ({best[5]:.3f}). 선형·트리 모델 간 격차 = "
      f"{best[5]-min(r[5] for r in rows):.3f}")
print("  해석: 모델 클래스를 바꿔도 CV가 비슷하면 '정확도 한계는 모델이 아니라 데이터(표본)'.")
print("\n### T2·T4 완료 (read-only)")
