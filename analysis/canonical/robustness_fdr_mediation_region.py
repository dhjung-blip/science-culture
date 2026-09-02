# -*- coding: utf-8 -*-
"""추가 강건성 분석 — L1 FDR 다중검정 · L2 직접산출 매개 CI · L3 leave-region · L4 between-only + 유효표본.

L1 (다중검정 FDR): within 3타깃·1차 횡단 3타깃 p값에 Benjamini–Hochberg 보정 → 특허 within 단일
    양성이 FDR 통과하는지(검토 논점 다중비교 관점 과대해석 경계).
L2 (직접산출 매개 CI): 특허·논문 매개비율(연구원 밀도 경유)에 국가 클러스터 부트스트랩 CI 부여
    (검토 논점 헤드라인 직접산출 매개에 CI 부재).
L3 (leave-region-out): 대륙(Europe·Asia·Americas·Oceania) 단위로 빼며 특허 횡단 β 재추정
    (검토 논점 지역 지렛대·외적 타당도).
L4 (between-only + 유효표본): 국가 평균 기반 횡단 β(순수 between) + 변수별 유효 정보량 표
    (검토 논점 528은 명목치·실효 ≈44국).

read-only: 동결 processed_dataset.csv, stdout + JSON 동결용 요약.
"""
import warnings

warnings.filterwarnings("ignore")
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

DF = pd.read_csv("analysis/canonical/processed_dataset.csv")
CTRL = ["rd_gdp_pct", "gdp_per_capita"]
CONTINENT = {**{c: "Europe" for c in ["AUT","BEL","CHE","CZE","DEU","DNK","ESP","EST","FIN","FRA","GBR","GRC","HUN","IRL","ISL","ITA","LTU","LUX","LVA","NLD","NOR","POL","PRT","RUS","SVK","SVN","SWE","TUR"]},
              **{c: "Asia" for c in ["HKG","IDN","JPN","KOR","SGP","ISR"]},
              **{c: "Americas" for c in ["BRA","CAN","CHL","COL","CRI","MEX","PER","USA"]},
              **{c: "Oceania" for c in ["AUS","NZL"]}}
RNG = np.random.default_rng(42)


def z(s):
    return (s - s.mean()) / s.std(ddof=0)


def cross_beta(df, target, ctrl=CTRL):
    sub = df.dropna(subset=[target, "sci_literacy", *ctrl, "country"])
    y = z(sub[target]).values
    X = pd.DataFrame({"sci": z(sub["sci_literacy"]).values})
    for c in ctrl:
        X[c] = z(sub[c]).values
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": sub["country"].values})
    return m.params["sci"], m.pvalues["sci"], len(sub), sub.country.nunique()


# ── L1: FDR 다중검정 ───────────────────────────────────────────────
print("=" * 78)
print("L1. 다중검정 FDR 보정 (Benjamini–Hochberg)")
print("=" * 78)
within_p = {"특허 within": 0.0006, "GII산출 within": 0.1192, "GII종합 within": 0.0521}
rej_w, padj_w, _, _ = multipletests(list(within_p.values()), alpha=0.05, method="fdr_bh")
print("  [within 3타깃]")
for (k, p), r, pa in zip(within_p.items(), rej_w, padj_w):
    print(f"    {k:14} p={p:.4f}  FDR p_adj={pa:.4f}  {'유의(FDR<.05)' if r else 'n.s.'}")
cross_p = {"특허 횡단": 0.0003, "논문 횡단": 0.0001, "GII산출 횡단": 0.001}
rej_c, padj_c, _, _ = multipletests(list(cross_p.values()), alpha=0.05, method="fdr_bh")
print("  [1차 횡단 3타깃]")
for (k, p), r, pa in zip(cross_p.items(), rej_c, padj_c):
    print(f"    {k:14} p={p:.4f}  FDR p_adj={pa:.4f}  {'유의' if r else 'n.s.'}")
L1 = {"within_fdr": {k: round(float(pa), 4) for k, pa in zip(within_p, padj_w)},
      "within_fdr_sig": {k: bool(r) for k, r in zip(within_p, rej_w)},
      "cross_fdr": {k: round(float(pa), 4) for k, pa in zip(cross_p, padj_c)},
      "cross_fdr_all_sig": bool(all(rej_c))}


# ── L2: 직접산출 매개 부트스트랩 CI ────────────────────────────────
print("\n" + "=" * 78)
print("L2. 직접산출 매개비율 국가 클러스터 부트스트랩 CI (2000회)")
print("=" * 78)
def mediation_ratio(df, target):
    sub = df.dropna(subset=[target, "sci_literacy", "researchers_per_m", *CTRL, "country"]).reset_index(drop=True)
    C = {c: z(sub[c]).values for c in CTRL}
    sci, M, Y = z(sub["sci_literacy"]).values, z(sub["researchers_per_m"]).values, z(sub[target]).values
    Xtot = sm.add_constant(pd.DataFrame({"sci": sci, **C}))
    c_tot = sm.OLS(Y, Xtot).fit().params["sci"]
    Xa = sm.add_constant(pd.DataFrame({"sci": sci, **C}))
    a = sm.OLS(M, Xa).fit().params["sci"]
    Xb = sm.add_constant(pd.DataFrame({"sci": sci, "M": M, **C}))
    b = sm.OLS(Y, Xb).fit().params["M"]
    ratio = (a * b) / c_tot if c_tot else np.nan
    # 국가 클러스터 부트스트랩
    countries = sub.country.unique()
    boot = []
    for _ in range(2000):
        pick = RNG.choice(countries, len(countries), replace=True)
        idx = np.concatenate([sub.index[sub.country == cc].values for cc in pick])
        s2 = sub.loc[idx]
        C2 = {c: z(s2[c]).values for c in CTRL}
        sci2, M2, Y2 = z(s2["sci_literacy"]).values, z(s2["researchers_per_m"]).values, z(s2[target]).values
        try:
            ct = sm.OLS(Y2, sm.add_constant(pd.DataFrame({"sci": sci2, **C2}))).fit().params["sci"]
            aa = sm.OLS(M2, sm.add_constant(pd.DataFrame({"sci": sci2, **C2}))).fit().params["sci"]
            bb = sm.OLS(Y2, sm.add_constant(pd.DataFrame({"sci": sci2, "M": M2, **C2}))).fit().params["M"]
            if ct: boot.append((aa * bb) / ct)
        except Exception:
            pass
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {target}: 매개비율 {100*ratio:+.1f}%  부트 95% CI [{100*lo:+.1f}%, {100*hi:+.1f}%]  (a={a:+.3f} b={b:+.3f} n={len(sub)})")
    return dict(target=target, ratio_pct=round(100 * ratio, 1), ci_lo_pct=round(100 * lo, 1),
               ci_hi_pct=round(100 * hi, 1), includes_zero=bool(lo <= 0 <= hi), n=len(sub))
L2 = [mediation_ratio(DF, "patents_per_m_log"), mediation_ratio(DF, "articles_per_m_log")]


# ── L3: leave-region-out ──────────────────────────────────────────
print("\n" + "=" * 78)
print("L3. leave-region-out — 특허 횡단 β (대륙 단위 제외)")
print("=" * 78)
full_b, full_p, _, _ = cross_beta(DF, "patents_per_m_log")
print(f"  전체 β={full_b:+.4f} (p={full_p:.4f})")
DF2 = DF.copy(); DF2["region"] = DF2.country.map(CONTINENT)
L3 = {"full_beta": round(float(full_b), 4), "by_region_excluded": {}}
for reg in ["Europe", "Asia", "Americas", "Oceania"]:
    b, p, n, nc = cross_beta(DF2[DF2.region != reg], "patents_per_m_log")
    print(f"  −{reg:9} 제외 → β={b:+.4f} (p={p:.4f}, {nc}국)")
    L3["by_region_excluded"][reg] = dict(beta=round(float(b), 4), p=round(float(p), 4), n_countries=int(nc))
betas = [v["beta"] for v in L3["by_region_excluded"].values()]
L3["beta_range"] = [round(min(betas), 4), round(max(betas), 4)]
L3["all_positive_sig"] = bool(all(v["beta"] > 0 and v["p"] < 0.05 for v in L3["by_region_excluded"].values()))


# ── L4: between-only + 유효표본 ───────────────────────────────────
print("\n" + "=" * 78)
print("L4. between-only(국가 평균) β + 유효표본")
print("=" * 78)
def between_beta(target):
    g = DF.groupby("country").mean(numeric_only=True).dropna(subset=[target, "sci_literacy", *CTRL])
    y = z(g[target]).values
    X = pd.DataFrame({"sci": z(g["sci_literacy"]).values, **{c: z(g[c]).values for c in CTRL}})
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC1")
    return m.params["sci"], m.pvalues["sci"], len(g)
bb, bp, bn = between_beta("patents_per_m_log")
ba, pa2, na = between_beta("articles_per_m_log")
print(f"  특허 between-only β={bb:+.4f} (p={bp:.4f}, {bn}국)")
print(f"  논문 between-only β={ba:+.4f} (p={pa2:.4f}, {na}국)")
eff = {"nominal_obs": 528, "effective_countries": 44, "pisa_anchor_pct": 31, "attitude_anchor_pct": 6.2,
       "patent_n": int(DF.patents_per_m_log.notna().sum()), "article_n": int(DF.articles_per_m_log.notna().sum()),
       "sci_trust_n": int(DF.sci_trust.notna().sum()), "researchers_n": int(DF.researchers_per_m.notna().sum())}
print(f"  유효표본: 명목 528 / 국가 44 / PISA앵커 31% / 태도앵커 6.2% / 특허 n={eff['patent_n']} 논문 n={eff['article_n']}")
L4 = {"patent_between_beta": round(float(bb), 4), "patent_between_p": round(float(bp), 4), "patent_between_n": int(bn),
      "article_between_beta": round(float(ba), 4), "article_between_p": round(float(pa2), 4),
      "effective_sample": eff}

print("\n" + "=" * 78)
print("[JSON 동결용 요약]")
print(json.dumps({"L1_fdr": L1, "L2_mediation_ci": L2, "L3_leave_region": L3, "L4_between_effn": L4}, ensure_ascii=False))
print("\n### 완료 (read-only)")
