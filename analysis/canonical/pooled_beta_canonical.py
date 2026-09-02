# -*- coding: utf-8 -*-
"""헤드라인 Pooled β 재현·동결 — sci_literacy → innovation_score, R&D·발전수준 통제.

read-only: 동결 analysis/canonical/processed_dataset.csv (input_hash cf2344519aad771a)
사용. 정본 code·xai_results.json 미변경, stdout 만.
목적: results_main.json 의 pooled_regression._flag("단일 재현 스크립트 미확정") 해소 —
β 의 정확한 스케일·통제·표본 사양을 코드로 확정하고 VIF(sci_literacy) 를 재산출한다.
사양별로 표본을 분리(통제변수 결측 행만 제외)하여 표본 축소가 계수에 미치는 영향을 분리한다.
"""
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

CSV = "analysis/canonical/processed_dataset.csv"
df = pd.read_csv(CSV)


def fit(spec, xstd, ystd, label):
    sub = df.dropna(subset=spec + ["innovation_score", "country"]).reset_index(drop=True)
    gg = sub["country"].values

    def zz(s):
        return (s - s.mean()) / s.std(ddof=0)

    X = pd.DataFrame({c: (zz(sub[c]) if xstd else sub[c]) for c in spec})
    y = zz(sub["innovation_score"]) if ystd else sub["innovation_score"]
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": gg})
    b, se, p = m.params["sci_literacy"], m.bse["sci_literacy"], m.pvalues["sci_literacy"]
    ysd = sub["innovation_score"].std(ddof=0)
    print(f"  {label:46s} n={len(sub)} {sub.country.nunique()}국  β={b:+8.4f} SE={se:.4f} p={p:.4f}  (Y_SD={ysd:.3f})")
    return b


print("── sci_literacy 계수 — 통제: R&D/GDP·1인당GDP (사양별 표본 분리) ──")
fit(["sci_literacy", "rd_gdp_pct", "gdp_per_capita"], True, False, "X·통제 표준화 · Y 원단위 (GII점/1SD)")
fit(["sci_literacy", "rd_gdp_pct", "gdp_per_capita"], True, True, "완전표준화 (X·Y·통제 z)")
print("── +연구원밀도 통제 (Oster mediator-included 사양과 동일 통제) ──")
fit(["sci_literacy", "rd_gdp_pct", "gdp_per_capita", "researchers_per_m"], True, False, "X·통제 표준화 · Y 원단위")
fit(["sci_literacy", "rd_gdp_pct", "gdp_per_capita", "researchers_per_m"], True, True, "완전표준화")

print("\n── VIF (표준화, 통제 R&D/GDP·1인당GDP·연구원밀도; 표본=4변수 공통결측 제외) ──")
sub = df.dropna(subset=["sci_literacy", "rd_gdp_pct", "gdp_per_capita", "researchers_per_m"]).reset_index(drop=True)
Xv = sm.add_constant(pd.DataFrame({c: (sub[c] - sub[c].mean()) / sub[c].std(ddof=0) for c in ["sci_literacy", "rd_gdp_pct", "gdp_per_capita", "researchers_per_m"]}))
for i, c in enumerate(Xv.columns):
    if c != "const":
        print(f"  VIF[{c:18s}] = {variance_inflation_factor(Xv.values, i):.3f}")

print("\n### Pooled β 재현 완료 (read-only)")
