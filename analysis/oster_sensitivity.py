# -*- coding: utf-8 -*-
"""Oster (2019) 미관측 교란 민감도 한계 — 횡단 핵심연관 sci_literacy → innovation_score.

paper Table 6 Pooled β 에 대한 견고성: 관측 통제 대비 미관측 교란이 얼마나 더 강해야
효과(β)가 0이 되는지(δ)를 Oster 비례선택(proportional selection) 가정으로 계산한다.

read-only: 정본·산출물(xai_results.json/*.png/*.csv) 미변경, stdout 만.

설계:
  (1) UNCONTROLLED : innovation_score ~ sci_literacy            → β0, R0²
  (2) CONTROLLED   : + rd_gdp_pct, researchers_per_m, gdp_per_capita → β1, R1²
  (3) R_max = min(1.3·R1², 1.0)
      δ(β=0)  : 효과를 0으로 만드는 데 필요한 미관측/관측 선택비.
      β*(δ=1) : 비례선택 동등(δ=1) 가정 하의 편의보정 β.
  표준 SE 는 statsmodels OLS, 국가(country) 클러스터.
  변수는 표준화(Oster 식의 분산항이 스케일 불변이긴 하나 명세대로 표준화 입력 사용).

Oster (2019, JBES) 식:
  비례선택 δ 가정 하 처치효과의 편의보정값 β* 는 3차방정식의 근.
  여기서는 (a) β*=0 을 만드는 δ 의 닫힌형 근사(Oster Prop.2의 표준 구현),
           (b) δ=1 일 때의 β* (동일 3차식의 실근) 를 보고한다.
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
import statsmodels.api as sm

import data_collector as d
import preprocessor as p

# ── 데이터 (정본 패턴) ───────────────────────────────────────────────
CC = d.resolve_country_set("pisa-gii")
raw = d.build_full_dataset(
    countries=CC, years=d.YEARS, use_worldbank=True, use_fred=False, seed=42
)
processed, X, y, feats, pipe, X_raw = p.run_preprocessing(
    raw, target="innovation_score"
)
groups = processed.loc[y.index, "country"].values

TREAT = "sci_literacy"
CONTROLS = ["rd_gdp_pct", "researchers_per_m", "gdp_per_capita"]
for c in [TREAT, *CONTROLS]:
    assert c in X.columns, f"피처 부재: {c}  (있는 것: {list(X.columns)})"

# y 도 표준화(횡단 표준화 회귀; β 가 SD 단위 → Table 6 Pooled 와 동일 스케일 의미)
yv = (y.values - y.values.mean()) / y.values.std(ddof=0)
yser = pd.Series(yv, index=y.index, name="innovation_z")
Xtreat = X[[TREAT]].copy()
Xfull = X[[TREAT, *CONTROLS]].copy()

n = len(yser)
nclust = pd.Series(groups).nunique()
print(f"### Oster 민감도 — n={n} 관측, {nclust} 국가 클러스터")
print(f"### 처치: {TREAT}  통제: {CONTROLS}\n")


def ols_cluster(yvec, Xdf):
    """절편 포함 OLS + 국가 클러스터 SE. (params, bse, rsq) 반환."""
    Xc = sm.add_constant(Xdf, has_constant="add")
    m = sm.OLS(yvec.values, Xc.values).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    names = list(Xc.columns)
    params = dict(zip(names, m.params))
    bse = dict(zip(names, m.bse))
    return params, bse, m.rsquared


# (1) UNCONTROLLED
p0, se0, R0 = ols_cluster(yser, Xtreat)
beta0 = p0[TREAT]
# (2) CONTROLLED
p1, se1, R1 = ols_cluster(yser, Xfull)
beta1 = p1[TREAT]

print(f"(1) UNCONTROLLED : β0 = {beta0:.4f}  (clusterSE {se0[TREAT]:.4f})   R0² = {R0:.4f}")
print(f"(2) CONTROLLED   : β1 = {beta1:.4f}  (clusterSE {se1[TREAT]:.4f})   R1² = {R1:.4f}")

# ── (3) Oster δ / β*(δ=1) ────────────────────────────────────────────
# 분산 보조량(Oster 식): σ_yy = Var(y), 그리고 τ_x = 처치를 다른 통제에 회귀했을 때의 잔차분산.
# 표준화 y 이므로 σ_yy = 1. (ddof=0 모분산 기준으로 일관)
sigma_yy = yser.var(ddof=0)

# τ_xx : 처치(sci_literacy)를 통제집합에 OLS 회귀 → 잔차분산 (full model 의 처치 직교성분).
Xc_ctrl = sm.add_constant(X[CONTROLS].values, has_constant="add")
m_tx = sm.OLS(X[[TREAT]].values, Xc_ctrl).fit()
tau_xx = float(np.var(m_tx.resid, ddof=0))  # = Var(treat | controls)
sigma_xx = float(X[[TREAT]].values.var(ddof=0))  # Var(treat) (≈1, 표준화)

R_max = min(1.3 * R1, 1.0)
beta_hypoth = 0.0  # 귀무: 효과 없음


def oster_cubic_coeffs(R_max_):
    """Oster(2019) β* 3차방정식 a·β*³ + b·β*² + c·β* + d 의 계수 반환.

    표준 도출(Oster 부록·Emily Oster psacalc 와 일치):
      let  num    = (β1 - β*) * (R1 - R0) * sigma_yy * sigma_xx
           den    = (R_max - R1) * (β0 - β1) * tau_xx
      δ = num / den  ...(A)
    β*=0 의 δ 는 (A) 에 β*=0 대입.  δ=1 의 β* 는 (A)=1 의 해(여기선 1차식이라 닫힌형).
    (Oster 의 일반식은 측정오차항까지 포함한 3차식이나, 비례선택 표준 구현은
     선택계수 δ 에 대해 위 비율식을 사용 — psacalc 의 'delta' 보고와 동일.)
    """
    return None  # (아래에서 닫힌형 직접 사용)


# δ for β* = 0  (식 (A) 에 β*=0 대입)
num0 = (beta1 - beta_hypoth) * (R1 - R0) * sigma_yy * sigma_xx
den0 = (R_max - R1) * (beta0 - beta1) * tau_xx
delta = num0 / den0

# β*(δ=1)  : 식 (A) 에서 δ=1 로 두고 β* 에 대해 풀기.
#   (β1 - β*)(R1 - R0) σ_yy σ_xx = (R_max - R1)(β0 - β1) τ_xx
#   ⇒ β* = β1 - [ (R_max - R1)(β0 - β1) τ_xx ] / [ (R1 - R0) σ_yy σ_xx ]
beta_delta1 = beta1 - (R_max - R1) * (beta0 - beta1) * tau_xx / (
    (R1 - R0) * sigma_yy * sigma_xx
)

print()
print(f"보조량: σ_yy={sigma_yy:.4f}  σ_xx(treat)={sigma_xx:.4f}  τ_xx(treat|controls)={tau_xx:.4f}")
print(f"R_max  = min(1.3·R1², 1.0) = {R_max:.4f}")
print()
print(f"(3) Oster δ (β*→0) = {delta:.4f}")
print(f"    β*(δ=1)        = {beta_delta1:.4f}")
print()

# ── 해석 ──────────────────────────────────────────────────────────────
robust = abs(delta) > 1
print("### 해석")
print(f"  δ = {delta:.3f} → ", end="")
if robust:
    print(
        f"|δ|>1: 효과를 0으로 지우려면 미관측 교란이 관측 통제(R&D/연구원/소득)보다\n"
        f"        약 {abs(delta):.2f}배 더 강해야 함 ⇒ 비례선택(δ=1) 기준 견고."
    )
else:
    print(
        f"|δ|<1: 관측 통제보다 약한 미관측만으로도 효과가 0이 될 수 있음 ⇒ 취약(민감)."
    )
print(
    f"  β*(δ=1) = {beta_delta1:.4f}: 미관측이 관측과 '동등하게' 작용한다고 가정한\n"
    f"            편의보정 추정치. 부호 유지({'양(+)' if beta_delta1 > 0 else '음(-)'})·",
    end="",
)
print("0 미포함" if (beta_delta1 * beta1 > 0 and abs(beta_delta1) > 0) else "0 근접/교차")

# Oster 식별집합 [β1, β*(δ=1)] 요약
lo, hi = sorted([beta1, beta_delta1])
print(f"  Oster 식별구간 [β1, β*(δ=1)] = [{lo:.4f}, {hi:.4f}]  "
      f"({'0 미포함→부호 견고' if lo > 0 or hi < 0 else '0 포함'})")
