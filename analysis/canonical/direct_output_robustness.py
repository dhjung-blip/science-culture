# -*- coding: utf-8 -*-
"""직접 산출(특허·논문 per-capita) 1차 결과변수에 대한 강건성 검정 — Oster δ·LOCO·WCB.

read-only: 동결 analysis/canonical/processed_dataset.csv 사용, stdout 만.
목적: 핵심 검토 논점 — 1차 결과변수를 직접 산출로 승격했으므로 Oster·와일드
클러스터 부트스트랩·LOCO 강건성도 GII pooled 가 아니라 직접 산출 타깃에 직접 수행해 정렬한다.
'GII 강건성을 직접 산출로 일반화'하지 않고 실측한다.
처치 sci_literacy, 통제 R&D/GDP·1인당GDP(confounder-only), 국가 클러스터.
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("analysis/canonical/processed_dataset.csv")
CTRL = ["rd_gdp_pct", "gdp_per_capita"]


def z(s):
    return (s - s.mean()) / s.std(ddof=0)


def analyze(target):
    sub = df.dropna(subset=[target, "sci_literacy", *CTRL, "country"]).reset_index(drop=True)
    g = sub["country"].values
    y = z(sub[target])
    t = z(sub["sci_literacy"])
    Xc = pd.DataFrame({"sci_literacy": t, "rd": z(sub["rd_gdp_pct"]), "gdp": z(sub["gdp_per_capita"])})
    m = sm.OLS(y, sm.add_constant(Xc)).fit(cov_type="cluster", cov_kwds={"groups": g})
    b1, se, p, R1 = m.params["sci_literacy"], m.bse["sci_literacy"], m.pvalues["sci_literacy"], m.rsquared
    m0 = sm.OLS(y, sm.add_constant(t)).fit()
    b0, R0 = m0.params["sci_literacy"], m0.rsquared
    tau = float(np.var(sm.OLS(t, sm.add_constant(Xc[["rd", "gdp"]])).fit().resid, ddof=0))
    delta = b1 * (R1 - R0) / ((1.0 - R1) * (b0 - b1) * tau)
    Rmax_crit = R1 + b1 * (R1 - R0) / ((b0 - b1) * tau)

    # LOCO
    locos = []
    for c in sorted(sub.country.unique()):
        s2 = sub[sub.country != c].reset_index(drop=True)
        X2 = pd.DataFrame({"sci_literacy": z(s2["sci_literacy"]), "rd": z(s2["rd_gdp_pct"]), "gdp": z(s2["gdp_per_capita"])})
        mm = sm.OLS(z(s2[target]), sm.add_constant(X2)).fit(cov_type="cluster", cov_kwds={"groups": s2["country"].values})
        locos.append((mm.params["sci_literacy"], mm.pvalues["sci_literacy"]))
    lb = [x[0] for x in locos]

    # 와일드 클러스터 부트스트랩 (Rademacher · null-imposed β_sci=0 · 999회)
    t_obs = b1 / se
    Xr = sm.add_constant(pd.DataFrame({"rd": z(sub["rd_gdp_pct"]), "gdp": z(sub["gdp_per_capita"])})).values
    mr = sm.OLS(y.values, Xr).fit()
    fit_r, resid_r = mr.fittedvalues, mr.resid
    Xf = sm.add_constant(Xc).values
    clusters = np.unique(g)
    rng = np.random.default_rng(42)
    cnt, B = 0, 999
    for _ in range(B):
        w = {c: rng.choice([-1.0, 1.0]) for c in clusters}
        wv = np.array([w[gi] for gi in g])
        ys = fit_r + resid_r * wv
        ms = sm.OLS(ys, Xf).fit(cov_type="cluster", cov_kwds={"groups": g})
        ts = ms.params[1] / ms.bse[1]
        if abs(ts) >= abs(t_obs):
            cnt += 1
    p_wcb = (cnt + 1) / (B + 1)

    print(f"\n■ 타깃: {target}  (n={len(sub)} · {sub.country.nunique()}국)")
    print(f"  횡단 β(sci, R&D·GDP 통제, 표준화) = {b1:+.4f}  cluster-SE {se:.4f}  p={p:.4f}  t={t_obs:.2f}")
    print(f"  Oster confounder-only δ(Rmax=1) = {delta:.3f}  (β0={b0:.3f} R0²={R0:.3f} | β1={b1:.3f} R1²={R1:.3f} | τ={tau:.3f})")
    print(f"    δ=1 임계 Rmax = {Rmax_crit:.3f} {'(>1 → 모든 가능 Rmax에서 δ>1)' if Rmax_crit > 1 else ''}")
    print(f"  LOCO β: [{min(lb):+.4f}, {max(lb):+.4f}]  전부 유의(p<.05)={all(pp < 0.05 for _, pp in locos)}  전부 양수={all(bb > 0 for bb in lb)}")
    print(f"  WCB(Rademacher·null-imposed·999) p = {p_wcb:.4f}  (관측 t={t_obs:.2f})")
    return dict(target=target, n=len(sub), beta=round(b1, 4), se=round(se, 4), p=round(p, 4),
               oster_delta=round(delta, 3), rmax_crit=round(Rmax_crit, 3),
               loco_min=round(min(lb), 4), loco_max=round(max(lb), 4),
               loco_all_sig=all(pp < 0.05 for _, pp in locos), loco_all_pos=all(bb > 0 for bb in lb),
               wcb_p=round(p_wcb, 4))


print("=== 직접 산출 1차 결과변수 강건성 (Oster·LOCO·WCB, confounder-only) ===")
res = [analyze("patents_per_m_log"), analyze("articles_per_m_log")]
print("\n--- JSON 동결용 요약 ---")
import json

print(json.dumps(res, ensure_ascii=False))
print("\n### 직접 산출 강건성 완료 (read-only)")
