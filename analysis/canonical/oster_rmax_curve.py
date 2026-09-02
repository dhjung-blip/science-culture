# -*- coding: utf-8 -*-
"""Oster δ 의 Rmax 민감도 곡선 — confounder-only 사양(R&D·발전수준 통제, 연구원밀도=매개 제외).

read-only: 동결 analysis/canonical/processed_dataset.csv 사용, stdout 만.
목적: 추가 강건성 분석 — Rmax=1.0 한 점이 아니라 보수적 상한(1.3·R̃ 등)에서의 δ·β*(δ=1),
그리고 δ 가 1 아래로 내려가는 임계 Rmax 를 보고하여 강건성 주장을 완성한다.

Oster(2019) 비례선택: δ = β1·(R1²-R0²)·σ_yy·σ_xx / [ (Rmax-R1²)·(β0-β1)·τ_xx ]  (β*=0)
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("analysis/canonical/processed_dataset.csv")
df = df.dropna(subset=["country", "innovation_score", "sci_literacy", "rd_gdp_pct", "gdp_per_capita"]).reset_index(drop=True)


def z(s):
    return (s - s.mean()) / s.std(ddof=0)


y = z(df["innovation_score"])
treat = z(df["sci_literacy"])
CTRL = ["rd_gdp_pct", "gdp_per_capita"]
Xc = pd.DataFrame({c: z(df[c]) for c in CTRL})

b0 = sm.OLS(y, sm.add_constant(treat)).fit().params["sci_literacy"]
R0 = sm.OLS(y, sm.add_constant(treat)).fit().rsquared
mfull = sm.OLS(y, sm.add_constant(pd.DataFrame({"sci_literacy": treat, **{c: Xc[c] for c in CTRL}}))).fit()
b1, R1 = mfull.params["sci_literacy"], mfull.rsquared
tau = float(np.var(sm.OLS(treat, sm.add_constant(Xc)).fit().resid, ddof=0))
syy, sxx = 1.0, float(treat.var(ddof=0))

print(f"n={len(df)} {df.country.nunique()}국 · 통제={CTRL}")
print(f"β0={b0:.4f} R0²={R0:.4f}  |  β1={b1:.4f} R1²={R1:.4f}  |  τ_xx={tau:.4f}")


def delta(Rmax):
    return b1 * (R1 - R0) * syy * sxx / ((Rmax - R1) * (b0 - b1) * tau)


def bstar(Rmax):
    return b1 - (Rmax - R1) * (b0 - b1) * tau / ((R1 - R0) * syy * sxx)


print("\n── Rmax 민감도 곡선 (δ: 효과를 0으로 만드는 미관측/관측 선택비) ──")
grid = [round(R1 + 1e-4, 4), 0.85, 0.90, 0.95, min(1.3 * R1, 1.0), 1.0]
for Rmax in sorted(set(x for x in grid if x > R1)):
    tag = "  ← Oster 권장 1.3·R̃ (cap 1.0)" if abs(Rmax - min(1.3 * R1, 1.0)) < 1e-6 else ("  ← 표준 보고값" if abs(Rmax - 1.0) < 1e-6 else "")
    print(f"  Rmax={Rmax:.4f}: δ={delta(Rmax):6.3f}  β*(δ=1)={bstar(Rmax):+.4f}{tag}")

Rmax_crit = R1 + b1 * (R1 - R0) * syy * sxx / ((b0 - b1) * tau)
print(f"\n── δ=1 임계 Rmax = {Rmax_crit:.4f} ──")
if Rmax_crit > 1.0:
    print(f"  임계 Rmax({Rmax_crit:.3f})>1.0 이고 R²는 1을 넘을 수 없으므로,")
    print(f"  가능한 모든 Rmax∈(R1², 1.0] 에서 δ>1 이 보장된다 (강건).")
else:
    print(f"  Rmax>{Rmax_crit:.3f} 에서 δ<1 — 그만큼 강한 미관측 R²를 가정해야 효과 소멸.")

print("\n### Oster Rmax 곡선 완료 (read-only)")
