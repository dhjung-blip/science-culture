# -*- coding: utf-8 -*-
"""탈순환 실험 1단계 — 타깃을 GII에서 '혁신 산출 직접측정'(특허·논문 per capita)으로 전환. [findings §7.7]
read-only 탐색: 정본·산출물 미변경, stdout만.

핵심 결과(이 탐색 스크립트 실측 — 간이 결측처리·n=516, 2026-06 재실행 일치):
  특허/백만명 횡단 β +0.451*** · within +0.233*** (p=0.0002) · 매개 28%;
  논문/백만명 횡단 +0.543*** · within −0.03 n.s. · 매개 27%; 문화만 CV 0.656 > R&D투입만 0.513.
  ※ 정본 KNN 경로(`main.py --target patents_per_m_log`, n=427)는 특허 within +0.349*** —
     이 탐색치(+0.233)와 방향·유의성 일치, 크기 차이는 결측처리(간이 vs KNN)·표본(516 vs 427).
     findings §7.7 line 20(탐색)·30(정본) 참조.

※ 보존본(2026-06): 원본은 /tmp에서 휘발 → findings §7.7 구조 기반 복원 후 재실행해 위 실측 확인.
   인구 분모(researchers_per_m 과 공유)로 인한 ratio correlation 주의 — within FE는 국가더미가 흡수."""
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
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import data_collector as d

CC = d.resolve_country_set("pisa-gii")
YEARS = d.YEARS

# ── 데이터 (PISA 2022 반영본) — 정본이 탈순환 타깃을 이미 파생 ──
# data_collector L891-896: population 있을 때 articles/patents_per_m_log = log1p(산출/백만명).
# 실 WB 모드 전용(--no-wb 엔 부재). 별도 인구 merge 불필요(정본 raw 에 내장 — 029a79e).
raw = d.build_full_dataset(countries=CC, years=YEARS, use_worldbank=True, use_fred=False, seed=42)
assert "patents_per_m_log" in raw.columns, "탈순환 타깃 부재 — 실 WB 모드(use_worldbank=True) 필요"
TARGETS = ["articles_per_m_log", "patents_per_m_log"]

# ── 피처 정의 (산출·순환 제거) ──
CULTURE = ["sci_literacy", "pisa_math", "pisa_reading", "sci_trust", "tech_acceptance",
           "tertiary_enroll", "tertiary_attainment"]
RND_INPUT = ["rd_gdp_pct", "researchers_per_m"]
REWARD = ["relative_salary", "rd_budget_per_researcher", "youth_unemp_pct"]
INFRA = ["gov_edu_exp_pct", "internet_users_pct", "mobile_subs_per100", "secure_servers_per_m"]
DEV = ["gdp_per_capita"]

# 결측 처리: 국가내 보간 + 남는 건 열 중앙값 (preprocessor 로직 간이판 — 탈순환 탐색 전용)
work = raw.copy().sort_values(["country", "year"])
feat_all = CULTURE + RND_INPUT + REWARD + INFRA + DEV
for c in feat_all + TARGETS:
    work[c] = work.groupby("country")[c].transform(lambda s: s.interpolate(limit_direction="both"))
for c in feat_all:
    work[c] = work[c].fillna(work[c].median())

work = work.dropna(subset=TARGETS)
g = work["country"].values
print(f"### 탈순환 실험 — {work['country'].nunique()}국, {len(work)}행 (PISA 2022 반영)")
print("### 타깃: log(백만명당 논문/특허) — GII 미사용 = 순환 0\n")

def star(pv):
    return "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else "n.s."

scaler = StandardScaler()

def run_block(target_col, target_name):
    print("=" * 90)
    print(f"■ 타깃: {target_name}")
    print("=" * 90)
    y = work[target_col].values
    sets = {
        "문화만 (비순환)": CULTURE,
        "R&D투입만": RND_INPUT,
        "문화+투입+보상+인프라": CULTURE + RND_INPUT + REWARD + INFRA,
        "+발전수준(GDP)": CULTURE + RND_INPUT + REWARD + INFRA + DEV,
    }
    print(f"{'피처셋':28}{'CV R²':>10}{'RMSE':>9}")
    cvr = {}
    for name, cols in sets.items():
        X = scaler.fit_transform(work[cols])
        oof = np.full(len(y), np.nan)
        for tr, va in GroupKFold(5).split(X, y, g):
            m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 random_state=42, verbosity=0)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])
        cvr[name] = r2_score(y, oof)
        print(f"{name:28}{cvr[name]:>10.3f}{np.sqrt(mean_squared_error(y, oof)):>9.3f}")
    print(f"  → 문화·보상·인프라의 한계기여 (투입 대비): "
          f"{cvr['문화+투입+보상+인프라'] - cvr['R&D투입만']:+.3f}")

    # Pooled OLS — 과학소양 β (R&D 투입 + GDP 통제, 클러스터 SE)
    Xcols = ["sci_literacy"] + RND_INPUT + DEV
    Xs = pd.DataFrame(scaler.fit_transform(work[Xcols]), columns=Xcols)
    ys = pd.Series(scaler.fit_transform(work[[target_col]]).ravel())
    mo = sm.OLS(ys, sm.add_constant(Xs)).fit(cov_type="cluster", cov_kwds={"groups": g})
    print(f"\n  Pooled OLS (R&D투입+GDP 통제): sci_literacy β = {mo.params['sci_literacy']:+.3f} "
          f"{star(mo.pvalues['sci_literacy'])} (p={mo.pvalues['sci_literacy']:.4f})")

    # 매개 — 소양 → 연구원밀도 → 산출
    def fit(cols, dep):
        Xd = pd.DataFrame(scaler.fit_transform(work[cols]), columns=cols)
        return sm.OLS(dep, sm.add_constant(Xd)).fit(cov_type="cluster", cov_kwds={"groups": g})
    M = "researchers_per_m"
    m1 = fit(["sci_literacy"], ys)
    m2 = fit(["sci_literacy"], pd.Series(scaler.fit_transform(work[[M]]).ravel()))
    m3 = fit(["sci_literacy", M], ys)
    a, b_med, c_tot = (m2.params["sci_literacy"], m3.params[M], m1.params["sci_literacy"])
    ind = a * b_med
    sobel_z = ind / np.sqrt(b_med**2 * m2.bse["sci_literacy"]**2 + a**2 * m3.bse[M]**2)
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
    print(f"  매개 (소양→연구원밀도→산출): 총효과 {c_tot:+.3f}{star(m1.pvalues['sci_literacy'])}"
          f" · a {a:+.3f}{star(m2.pvalues['sci_literacy'])} · b {b_med:+.3f}{star(m3.pvalues[M])}"
          f" · 간접 {ind:+.3f} (Sobel z={sobel_z:+.2f} {star(sobel_p)}) · 매개 {100*ind/c_tot:.0f}%")

    # within FE — 국가 더미
    Dc = pd.get_dummies(pd.Series(g), prefix="c", drop_first=True).astype(float).reset_index(drop=True)
    Xfe = pd.concat([Xs.reset_index(drop=True), Dc], axis=1)
    mfe = sm.OLS(ys.reset_index(drop=True), sm.add_constant(Xfe)).fit(
        cov_type="cluster", cov_kwds={"groups": g})
    print(f"  국가 FE within: sci_literacy β = {mfe.params['sci_literacy']:+.3f} "
          f"{star(mfe.pvalues['sci_literacy'])} (p={mfe.pvalues['sci_literacy']:.4f})\n")

run_block("articles_per_m_log", "log(백만명당 과학논문) — 지식 산출")
run_block("patents_per_m_log", "log(백만명당 특허출원) — 기술 산출")

# 비교 기준: 동일 비순환 피처로 GII(순환 타깃) — 격차 확인
print("=" * 90)
print("■ 비교 기준: 동일 피처 → GII 타깃 (순환 구도와의 대비)")
print("=" * 90)
y_gii = work["innovation_score"].values
cols = CULTURE + RND_INPUT + REWARD + INFRA + DEV
X = scaler.fit_transform(work[cols])
oof = np.full(len(y_gii), np.nan)
for tr, va in GroupKFold(5).split(X, y_gii, g):
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=42, verbosity=0)
    m.fit(X[tr], y_gii[tr])
    oof[va] = m.predict(X[va])
print(f"  GII 타깃 (산출 피처 제외 후): CV R² = {r2_score(y_gii, oof):.3f}")
print("\n### 탈순환 실험 완료 (read-only)")
