# -*- coding: utf-8 -*-
"""와일드 클러스터 부트스트랩 — 소수 클러스터(G=44 국가) 보정 하의 sci_literacy 유의성.

동기: 통제 Pooled OLS(innovation_score ~ sci_literacy + rd_gdp_pct + researchers_per_m
+ gdp_per_capita, 모두 표준화)에서 일반 클러스터-로버스트 SE 는 클러스터 수가 적을 때
(여기 G=44 국가) p-value 를 과소추정(과대 유의)하는 경향이 있다. Cameron–Gelbach–Miller(2008)
의 와일드 클러스터 부트스트랩-t(WCB-t) 로 이를 보정한다.

방법(restricted / null-imposed residual bootstrap):
  1. 제약모형 적합 — sci_literacy 를 뺀 회귀로 H0: β_sci=0 을 부과.
     ŷ_r = 제약 적합치, û_r = 제약 잔차.
  2. 클러스터 단위 Rademacher 가중 w_g ∈ {-1,+1} (국가별 1개, 그 국가 모든 관측에 동일 적용).
  3. 부트스트랩 DGP: y* = ŷ_r + û_r · w_{g(i)}  (귀무 부과 → 검정의 size 정확).
  4. 검정통계량: 전체모형(sci_literacy 포함)에서 sci_literacy 의 클러스터-로버스트 t
     (국가로 재클러스터링)을 매 복제마다 재계산.
  5. p-value = ( #{|t*| ≥ |t_obs|} + 1 ) / (B + 1)  — 대칭(절댓값) 양측, +1 보정.

read-only: 정본·산출물 미변경, stdout 만. 실 WB 로드는 수 분 소요(정상).

데이터 패턴은 analysis/decircle_experiment.py 헤더를 따른다(정본 전처리 경유).
X(X_final) 는 이미 StandardScaler 표준화됨. y 도 표준화해 완전 표준화 회귀로 맞춘다.
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
from scipy import stats

import data_collector as d
import preprocessor as p

SEED = 42
B = 999  # 부트스트랩 복제수
FEATURES = ["sci_literacy", "rd_gdp_pct", "researchers_per_m", "gdp_per_capita"]
KEY = "sci_literacy"
TARGET = "innovation_score"

# ── 데이터 (정본 경유) ──────────────────────────────────────────────
CC = d.resolve_country_set("pisa-gii")
raw = d.build_full_dataset(countries=CC, years=d.YEARS, use_worldbank=True,
                           use_fred=False, seed=SEED)
processed, X, y, feats, pipe, X_raw = p.run_preprocessing(raw, target=TARGET)
groups = processed.loc[y.index, "country"].values

# 통제 회귀 설계행렬 — X 는 표준화 완료, y 도 표준화(완전 표준화 회귀)
Xc = X[FEATURES].reset_index(drop=True).astype(float)
yv = pd.Series(y.to_numpy(dtype=float))
yv = (yv - yv.mean()) / yv.std(ddof=0)          # y 표준화
yv = yv.reset_index(drop=True)
g = pd.Series(groups).reset_index(drop=True).to_numpy()

n = len(yv)
clusters = np.unique(g)
G = len(clusters)
print(f"### 와일드 클러스터 부트스트랩 — N={n} 관측, G={G} 국가 클러스터, B={B}")
print(f"### 통제 Pooled OLS: {TARGET} ~ {' + '.join(FEATURES)} (모두 표준화)\n")


def cluster_t(Xmat, yvec, gvec, key_col):
    """전체모형 OLS + 국가 클러스터-로버스트 SE → key_col 의 (계수, t통계량, p)."""
    Xd = sm.add_constant(Xmat, has_constant="add")
    res = sm.OLS(yvec, Xd).fit(cov_type="cluster", cov_kwds={"groups": gvec})
    beta = res.params[key_col]
    se = res.bse[key_col]
    return beta, beta / se, res.pvalues[key_col], res


# ── 1) 관측 통계량 (일반 클러스터-로버스트) ─────────────────────────
beta_obs, t_obs, p_crve, full_res = cluster_t(Xc, yv, g, KEY)
# statsmodels 클러스터 t 의 양측 p (정규근사 기반 — 보정 전 기준선)
df_g = G - 1
p_crve_t = 2 * (1 - stats.t.cdf(abs(t_obs), df_g))  # G-1 자유도 t 근사(보수적 기준)
print("■ 일반 클러스터-로버스트(보정 전)")
print(f"   sci_literacy: β = {beta_obs:+.4f}, 클러스터-로버스트 t = {t_obs:+.3f}")
print(f"   p (statsmodels, 정규근사)         = {p_crve:.5f}")
print(f"   p (t(G-1={df_g}) 근사)              = {p_crve_t:.5f}\n")

# ── 2) 제약(귀무 부과) 모형 — sci_literacy 제거 ────────────────────
restricted_cols = [c for c in FEATURES if c != KEY]
Xr = sm.add_constant(Xc[restricted_cols], has_constant="add")
res_r = sm.OLS(yv, Xr).fit()
yhat_r = res_r.fittedvalues.to_numpy()
uhat_r = res_r.resid.to_numpy()

# ── 3) 와일드 클러스터 부트스트랩-t (Rademacher, restricted) ───────
rng = np.random.default_rng(SEED)
# 클러스터→관측 인덱스 매핑(반복 적용 방지)
cl_index = {c: np.where(g == c)[0] for c in clusters}

t_star = np.empty(B)
n_valid = 0
for b in range(B):
    # 국가별 Rademacher ±1
    w_cluster = rng.choice([-1.0, 1.0], size=G)
    w = np.empty(n)
    for j, c in enumerate(clusters):
        w[cl_index[c]] = w_cluster[j]
    # 귀무 부과 부트스트랩 종속변수
    y_star = yhat_r + uhat_r * w
    # 전체모형에서 sci_literacy 의 클러스터-로버스트 t 재계산(국가로 재클러스터)
    try:
        _, t_b, _, _ = cluster_t(Xc, pd.Series(y_star), g, KEY)
        if np.isfinite(t_b):
            t_star[n_valid] = t_b
            n_valid += 1
    except Exception:
        continue

t_star = t_star[:n_valid]

# ── 4) p-value (대칭 절댓값 양측, +1 보정) ─────────────────────────
p_wcb = (np.sum(np.abs(t_star) >= abs(t_obs)) + 1) / (n_valid + 1)

print("■ 와일드 클러스터 부트스트랩-t (Rademacher · restricted · 국가 클러스터)")
print(f"   유효 복제수            = {n_valid}/{B}")
print(f"   |t*| 분포 분위수: p50={np.percentile(np.abs(t_star),50):.3f}, "
      f"p95={np.percentile(np.abs(t_star),95):.3f}, "
      f"p99={np.percentile(np.abs(t_star),99):.3f}")
print(f"   |t_obs|                = {abs(t_obs):.3f}")
print(f"   WCB p-value            = {p_wcb:.5f}\n")


def verdict(pv):
    return ("*** (p<.001)" if pv < .001 else "** (p<.01)" if pv < .01
            else "* (p<.05)" if pv < .05 else "n.s. (p≥.05)")


print("=" * 78)
print("■ 비교 — 소수 클러스터 보정의 효과")
print("=" * 78)
print(f"   일반 클러스터-로버스트 p (정규근사) : {p_crve:.5f}  {verdict(p_crve)}")
print(f"   일반 클러스터-로버스트 p (t(G-1))   : {p_crve_t:.5f}  {verdict(p_crve_t)}")
print(f"   와일드 클러스터 부트스트랩 p        : {p_wcb:.5f}  {verdict(p_wcb)}")
sig_naive = p_crve < .05
sig_wcb = p_wcb < .05
print()
if sig_naive and sig_wcb:
    print("   → 결론: 유의성이 소수-클러스터 보정 후에도 유지된다(survives).")
elif sig_naive and not sig_wcb:
    print("   → 결론: 보정 전 유의했으나 WCB 후 유의성 소멸(does NOT survive) — "
          "일반 SE 가 과대유의였다.")
elif not sig_naive and sig_wcb:
    print("   → 결론: 보정 후 오히려 유의(드묾) — 재확인 필요.")
else:
    print("   → 결론: 두 방법 모두 비유의 — sci_literacy 독립기여 통계적 미검출.")
print("\n### 와일드 클러스터 부트스트랩 완료 (read-only)")
