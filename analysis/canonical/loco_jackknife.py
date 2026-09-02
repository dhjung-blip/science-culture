# -*- coding: utf-8 -*-
"""헤드라인 β 의 LOCO(leave-one-country-out) jackknife — 선진국 소수 영향관측 견고성.

read-only: 동결 analysis/canonical/processed_dataset.csv 사용, stdout 만.
목적: 추가 강건성 분석 — 44국 중 1국씩 제외하여 완전표준화 β 분포를 보고,
소수 영향관측(예: 룩셈부르크·싱가포르)이 결론을 좌우하지 않음을 입증한다.
사양: sci_literacy → innovation_score, R&D·발전수준 통제, 완전표준화, 국가 클러스터 SE.
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("analysis/canonical/processed_dataset.csv")
df = df.dropna(subset=["country", "innovation_score", "sci_literacy", "rd_gdp_pct", "gdp_per_capita"]).reset_index(drop=True)


def beta_full(sub):
    sub = sub.reset_index(drop=True)

    def z(s):
        return (s - s.mean()) / s.std(ddof=0)

    X = pd.DataFrame({"sci_literacy": z(sub["sci_literacy"]), "rd": z(sub["rd_gdp_pct"]), "gdp": z(sub["gdp_per_capita"])})
    m = sm.OLS(z(sub["innovation_score"]), sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": sub["country"].values})
    return m.params["sci_literacy"], m.pvalues["sci_literacy"]


base, base_p = beta_full(df)
res = []
for c in sorted(df.country.unique()):
    b, p = beta_full(df[df.country != c])
    res.append((c, b, p))

bs = [b for _, b, _ in res]
ps = [p for _, _, p in res]
print(f"전체 표본 β(완전표준화) = {base:+.4f} (p={base_p:.4f}), {df.country.nunique()}국")
print(f"LOCO β: min={min(bs):+.4f}  max={max(bs):+.4f}  mean={np.mean(bs):+.4f}  범위={max(bs)-min(bs):.4f}")
print(f"모든 44개 LOCO 추정 유의(p<0.05): {all(p < 0.05 for p in ps)}  (최대 p={max(ps):.4f})")
print(f"부호 안정(전부 양수): {all(b > 0 for b in bs)}")

res.sort(key=lambda x: abs(x[1] - base), reverse=True)
print("\n── 제외 시 β 변동이 가장 큰 5개국 ──")
for c, b, p in res[:5]:
    print(f"  {c} 제외: β={b:+.4f}  (Δ={b - base:+.4f}, {100 * (b - base) / base:+.1f}%)  p={p:.4f}")

print("\n### LOCO jackknife 완료 (read-only)")
