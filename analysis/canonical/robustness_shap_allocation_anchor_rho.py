# -*- coding: utf-8 -*-
"""추가 강건성 분석 3종 — 동결 데이터 강건성.

(3) SHAP 상관-배분 강건성: 인지축 5피처·태도축 2피처를 각각 단일 복합지수로 묶어
    Shapley 상관-배분 임의성을 제거한 뒤 인지/태도 SHAP 비율을 재산출. 인지 우세가
    상관-피처 배분 인공물이 아니라 신호 실체이면 단일 복합 비교에서도 유지되어야 한다.
(4) within 앵커연도 재추정: 특허 within(+0.238)의 처치 PISA가 ~69% 보간임을 통제하기 위해
    PISA 실측 앵커 연도(2012·2015·2018·2022)만 사용해 국가 FE within 재추정. 보간 인공물이면
    앵커 전용 추정에서 약화/소멸해야 한다.
(5) 매개 ρ-민감도(Imai–Tingley LSEM 근사): 매개변수-결과 미관측 교란 상관 ρ가 ACME를 0으로
    만드는 임계 ρ*(=b·σ_εM/σ_εY)를 산출. |ρ*|가 클수록 순차적 무시가능성 위반에 강건.

read-only: 동결 processed_dataset.csv, stdout + JSON 동결용 요약.
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
import statsmodels.api as sm
import xgboost as xgb

import config
import model_trainer as mt
import shap_analyzer as sh

DF = pd.read_csv("analysis/canonical/processed_dataset.csv")
COG = ["sci_literacy", "pisa_math", "pisa_reading", "tertiary_enroll", "tertiary_attainment"]
ATT = ["sci_trust", "tech_acceptance"]
OUTPUT = {"patent_apps", "journal_articles", "hi_tech_export_pct",
          "hi_tech_export_pct_lag3", "hi_tech_export_pct_lag5", "hi_tech_export_pct_lag7"}
ANCHOR = [2012, 2015, 2018, 2022]
PARAMS = {k: v for k, v in mt.DEFAULT_XGB_PARAMS.items() if k not in ("n_estimators", "random_state")}
PARAMS["n_estimators"] = 300
PARAMS["random_state"] = 42


def z(s):
    return (s - s.mean()) / s.std(ddof=0)


# ── (3) SHAP 상관-배분: 단일 인지/태도 복합 ──────────────────────────────
def shap_block(target):
    sub = DF.dropna(subset=[target]).copy()
    sub["idx_cog_single"] = sub[COG].apply(z).mean(axis=1)
    sub["idx_att_single"] = sub[ATT].apply(z).mean(axis=1)
    ctrl = [c for c in config.ALL_FEATURES if c not in COG + ATT and c not in OUTPUT]
    feat = ["idx_cog_single", "idx_att_single"] + [c for c in ctrl if c in sub.columns]
    X = sub[feat]
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X, sub[target])
    sv, _ = sh.compute_shap_values(model, X)
    ma = np.abs(sv).mean(axis=0)
    imp = dict(zip(list(X.columns), ma))
    cog, att = float(imp["idx_cog_single"]), float(imp["idx_att_single"])
    ratio = cog / att if att else float("inf")
    print(f"  [SHAP블록] {target}: 인지복합 {cog:.4f} vs 태도복합 {att:.4f} · 비율 {ratio:.2f}x (n={len(sub)})")
    return dict(target=target, cog_single=round(cog, 4), att_single=round(att, 4),
               ratio=round(ratio, 2), n=int(len(sub)))


# ── (4) within 앵커연도 재추정 ─────────────────────────────────────────
def within_anchor(target):
    sub = DF[DF.year.isin(ANCHOR)].dropna(
        subset=[target, "sci_literacy", "rd_gdp_pct", "gdp_per_capita", "country"]).copy()
    X = pd.DataFrame({"sci": z(sub["sci_literacy"]).values,
                     "rd": z(sub["rd_gdp_pct"]).values, "gdp": z(sub["gdp_per_capita"]).values},
                    index=sub.index)
    dummies = pd.get_dummies(sub["country"], prefix="c", drop_first=True).astype(float)
    Xf = sm.add_constant(pd.concat([X, dummies], axis=1))
    m = sm.OLS(z(sub[target]).values, Xf.values).fit(
        cov_type="cluster", cov_kwds={"groups": sub["country"].values})
    b, p = m.params[1], m.pvalues[1]  # const=0, sci=1
    print(f"  [앵커within] {target}: 표준화 β(sci) = {b:+.4f}  p={p:.4f}  (n={len(sub)} · {sub.country.nunique()}국 · 앵커연도만)")
    return dict(target=target, n=int(len(sub)), n_countries=int(sub.country.nunique()),
               beta_std=round(float(b), 4), p=round(float(p), 4))


# ── (5) ρ-민감도 (Imai–Tingley LSEM 근사) ─────────────────────────────
def rho_sens(target):
    sub = DF.dropna(subset=[target, "sci_literacy", "researchers_per_m",
                            "rd_gdp_pct", "gdp_per_capita", "country"]).copy()
    C = pd.DataFrame({"rd": z(sub["rd_gdp_pct"]).values, "gdp": z(sub["gdp_per_capita"]).values})
    sci = z(sub["sci_literacy"]).values
    Xa = sm.add_constant(pd.DataFrame({"sci": sci, **C.to_dict("series")}))
    ma = sm.OLS(z(sub["researchers_per_m"]).values, Xa.values).fit()
    a, sM = ma.params[1], float(np.std(ma.resid, ddof=0))
    Xb = sm.add_constant(pd.DataFrame({"sci": sci, "M": z(sub["researchers_per_m"]).values, **C.to_dict("series")}))
    mb = sm.OLS(z(sub[target]).values, Xb.values).fit()
    b, sY = mb.params[2], float(np.std(mb.resid, ddof=0))  # const0 sci1 M2
    rho_star = float(b * sM / sY)  # ACME=0 임계 ε_M–ε_Y 상관 (LSEM 근사)
    acme = float(a * b)
    print(f"  [ρ민감도] {target}: a={a:+.4f} b={b:+.4f} ACME={acme:+.4f} · ρ*(ACME=0)={rho_star:+.4f}")
    return dict(target=target, a=round(float(a), 4), b=round(float(b), 4),
               acme=round(acme, 4), rho_star=round(rho_star, 4))


print("=" * 80)
print("추가 검증 ③ 실질 분석 — SHAP 상관배분 · within 앵커 · ρ-민감도 (동결 데이터)")
print("=" * 80)
print("\n(3) SHAP 상관-배분 강건성 (단일 인지복합 vs 단일 태도복합):")
shap_res = [shap_block("patents_per_m_log"), shap_block("articles_per_m_log")]
print("\n(4) within 앵커연도(2012·2015·2018·2022) 재추정:")
anchor_res = [within_anchor("patents_per_m_log")]
print("\n(5) 매개 ρ-민감도 (Imai–Tingley LSEM 근사):")
rho_res = [rho_sens("patents_per_m_log"), rho_sens("articles_per_m_log")]

print("\n" + "=" * 80)
print("[JSON 동결용 요약]")
print(json.dumps({"shap_block": shap_res, "within_anchor": anchor_res, "rho_sens": rho_res}, ensure_ascii=False))
print("\n### 추가 검증 ③ 분석 완료 (read-only)")
