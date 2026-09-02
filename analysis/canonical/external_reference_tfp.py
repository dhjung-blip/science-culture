# -*- coding: utf-8 -*-
"""외부 준거 확장 — GII 구성체 가족 *밖* 완전독립 외부준거(PWT TFP)에서의 인지 인적자본 연관.

본 설계의 핵심 한계: "특허·논문 per-capita 도 GII 산출 구성체 가족 내부 → 합성지수→단일 구성요소
이동일 뿐". 이를 깨기 위해 GII 와 0중첩인 거시 경제결과(Penn World Table 11.0 TFP, Feenstra·
Inklaar·Timmer 2015, CC BY 4.0)를 외부준거 타깃으로 교체해 동일 구도(confounder-only·시드42)를 재실행.

타깃: ctfp(USA=1 수준, 횡단)·rtfpna(자기기준 성장, within). 처치 sci_literacy, 통제 R&D/GDP·1인당GDP.
정직: TFP 는 혁신의 하류 경제결과라 효과 희석 가능 — 약화/미검출도 그대로 보고한다.

read-only: 동결 processed_dataset.csv + external_tfp_pwt110.csv, stdout + JSON 동결용.
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
TFP = pd.read_csv("analysis/canonical/external_tfp_pwt110.csv")
M = DF.merge(TFP, left_on=["country", "year"], right_on=["countrycode", "year"], how="left")
CTRL = ["rd_gdp_pct", "gdp_per_capita"]
COG = ["sci_literacy", "pisa_math", "pisa_reading", "tertiary_enroll", "tertiary_attainment"]
ATT = ["sci_trust", "tech_acceptance"]
OUTPUT = {"patent_apps", "journal_articles", "hi_tech_export_pct",
          "hi_tech_export_pct_lag3", "hi_tech_export_pct_lag5", "hi_tech_export_pct_lag7"}
PARAMS = {k: v for k, v in mt.DEFAULT_XGB_PARAMS.items() if k not in ("n_estimators", "random_state")}
PARAMS["n_estimators"] = 300
PARAMS["random_state"] = 42


def z(s):
    return (s - s.mean()) / s.std(ddof=0)


def cross_oster(target):
    sub = M.dropna(subset=[target, "sci_literacy", *CTRL, "country"]).reset_index(drop=True)
    y = z(sub[target]).values
    t = z(sub["sci_literacy"]).values
    Xc = pd.DataFrame({"sci": t, "rd": z(sub["rd_gdp_pct"]).values, "gdp": z(sub["gdp_per_capita"]).values})
    m = sm.OLS(y, sm.add_constant(Xc)).fit(cov_type="cluster", cov_kwds={"groups": sub["country"].values})
    b1, p, R1 = m.params["sci"], m.pvalues["sci"], m.rsquared
    m0 = sm.OLS(y, sm.add_constant(t)).fit()
    b0, R0 = m0.params[1], m0.rsquared
    tau = float(np.var(sm.OLS(t, sm.add_constant(Xc[["rd", "gdp"]])).fit().resid, ddof=0))
    delta = b1 * (R1 - R0) / ((1.0 - R1) * (b0 - b1) * tau) if (b0 - b1) and (1 - R1) and tau else np.nan
    print(f"  [횡단·Oster] {target}: β={b1:+.4f} p={p:.4f} n={len(sub)} · Oster δ={delta:.3f}")
    return dict(target=target, beta=round(float(b1), 4), p=round(float(p), 4), n=int(len(sub)),
               oster_delta=round(float(delta), 3))


def within_fe(target):
    sub = M.dropna(subset=[target, "sci_literacy", *CTRL, "country"])
    X = pd.DataFrame({"sci": z(sub["sci_literacy"]).values, "rd": z(sub["rd_gdp_pct"]).values,
                     "gdp": z(sub["gdp_per_capita"]).values}, index=sub.index)
    d = pd.get_dummies(sub["country"], drop_first=True).astype(float)
    Xf = sm.add_constant(pd.concat([X, d], axis=1))
    m = sm.OLS(z(sub[target]).values, Xf.values).fit(cov_type="cluster", cov_kwds={"groups": sub["country"].values})
    print(f"  [within FE] {target}: β={m.params[1]:+.4f} p={m.pvalues[1]:.4f}")
    return dict(target=target, within_beta=round(float(m.params[1]), 4), within_p=round(float(m.pvalues[1]), 4))


def shap_axis(target):
    sub = M.dropna(subset=[target]).copy()
    base = [c for c in config.ALL_FEATURES if c not in OUTPUT]
    lags = [c for c in sub.columns if "_lag" in c and c not in OUTPUT]
    idxs = [c for c in sub.columns if c.startswith("idx_")]
    feat = [c for c in base + lags + idxs if c in sub.columns]
    X = sub[feat]
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X, sub[target])
    sv, _ = sh.compute_shap_values(model, X)
    ma = np.abs(sv).mean(axis=0)
    imp = dict(zip(list(X.columns), ma))
    total = float(ma.sum()) or 1.0

    def bo(c):
        return c.split("_lag")[0]
    cog = float(sum(v for c, v in imp.items() if bo(c) in COG))
    att = float(sum(v for c, v in imp.items() if bo(c) in ATT))
    ratio = cog / att if att else float("inf")
    print(f"  [SHAP축] {target}: 인지/태도 {ratio:.2f}x (인지 {100*cog/total:.1f}% vs 태도 {100*att/total:.1f}%)")
    return dict(target=target, cog_att_ratio=round(ratio, 2),
               cog_pct=round(100 * cog / total, 2), att_pct=round(100 * att / total, 2))


print("=" * 80)
print("외부 준거 — PWT TFP (GII 완전독립) 위 인지 인적자본 연관")
print("=" * 80)
print("\n[횡단 β + Oster δ] (confounder-only: R&D/GDP·1인당GDP 통제)")
cross = [cross_oster("ctfp"), cross_oster("rtfpna")]
print("\n[within 국가FE]")
within = [within_fe("ctfp"), within_fe("rtfpna")]
print("\n[SHAP 인지/태도 축분해]")
axis = [shap_axis("ctfp"), shap_axis("rtfpna")]

# ── 삼극특허 (GII-PCT와 분리된 특허 질 외부준거; OECD 삼극 패밀리, INVENTOR·PRIORITY·PF·Total) ──
print("\n[삼극특허 per-capita 외부준거]")
TRI = pd.read_csv("analysis/canonical/external_triadic_oecd.csv")
MT = DF.merge(TRI, on=["country", "year"], how="left")
MT["triadic_pc_log"] = np.log1p(MT["triadic"] / (MT["population"] / 1e6))
subt = MT.dropna(subset=["triadic_pc_log", "sci_literacy", *CTRL, "country"]).reset_index(drop=True)
yt = z(subt["triadic_pc_log"]).values
tt = z(subt["sci_literacy"]).values
Xt = pd.DataFrame({"sci": tt, "rd": z(subt["rd_gdp_pct"]).values, "gdp": z(subt["gdp_per_capita"]).values})
mtri = sm.OLS(yt, sm.add_constant(Xt)).fit(cov_type="cluster", cov_kwds={"groups": subt["country"].values})
bt, pt, R1t = mtri.params["sci"], mtri.pvalues["sci"], mtri.rsquared
m0t = sm.OLS(yt, sm.add_constant(tt)).fit()
b0t, R0t = m0t.params[1], m0t.rsquared
taut = float(np.var(sm.OLS(tt, sm.add_constant(Xt[["rd", "gdp"]])).fit().resid, ddof=0))
deltat = bt * (R1t - R0t) / ((1.0 - R1t) * (b0t - bt) * taut)
triadic = dict(target="triadic_pc_log", beta=round(float(bt), 4), p=round(float(pt), 4),
              n=int(len(subt)), n_countries=int(subt.country.nunique()), oster_delta=round(float(deltat), 3))
print(f"  삼극 per-capita 횡단 β={bt:+.4f} p={pt:.4f} δ={deltat:.3f} n={len(subt)} {subt.country.nunique()}국")

print("\n" + "=" * 80)
print("[JSON 동결용 요약]")
print(json.dumps({"cross_oster": cross, "within": within, "shap_axis": axis, "triadic": triadic}, ensure_ascii=False))
print("\n### 외부준거(TFP) 완료 (read-only)")
