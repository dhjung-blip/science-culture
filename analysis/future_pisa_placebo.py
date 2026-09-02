"""식별전략 — future-PISA placebo(반증 검정).

미래 PISA 가 *과거* 혁신 산출을 예측하는지 검정한다. 인과/시간선행이 실재한다면 미래 원인이
과거 결과를 예측해선 안 되므로, 이는 국가 고정특성·장기 발전수준 로딩을 점검하는 placebo 다.

    PastOutput_{i,t-k} = β·FuturePISA_{i,t} + γ·X_{i,t} + ε

해석(엄수):
- placebo β 가 약하면(또는 forward 대비 작으면) 시간 선행성 주장이 강화된다.
- placebo β 가 forward 와 비슷하게 강하면, PISA 계수는 '교육→혁신' 선행이 아니라 국가 장기특성
  로딩(부유·선진·과학지향국이 PISA 도 높고 과거에도 혁신적)일 가능성이 크다.
- 어떤 경우에도 결과를 숨기지 않는다. 본 데이터는 1~3 코호트로 매우 작아 탐색적이다.

회귀자 PISA 는 raw 실측 앵커만 사용(pisa_cohort_lag.py 와 동일 규약). 통제는 PISA 연도 t.
read-only: 동결 데이터. stdout + JSON 동결용.
재현: cd <package-root>; python analysis/future_pisa_placebo.py
"""

import hashlib
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

RAW = "analysis/canonical/raw_dataset.csv"
PROC = "analysis/canonical/processed_dataset.csv"
TRI = "analysis/canonical/external_triadic_oecd.csv"

ANCHORS = [2012, 2015, 2018, 2022]
YEAR_MIN = 2012
LAGS_BACK = [3, 6, 9]
CONTROLS = ["rd_gdp_pct", "gdp_per_capita"]
PISA_REGRESSORS = ["sci_literacy", "pisa_math"]
OUTCOMES = [
    ("patents_per_m_log", "World Bank 특허출원 per-capita 로그"),
    ("articles_per_m_log", "World Bank SCI논문 per-capita 로그"),
    ("triadic_pc_log", "OECD 삼극특허 per-capita 로그"),
    ("innovation_score", "WIPO GII 종합(benchmark)"),
]


def file_hash(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:16]


def z(s):
    s = np.asarray(s, dtype=float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd else s * 0.0


def load():
    raw = pd.read_csv(RAW)
    proc = pd.read_csv(PROC)
    tri = pd.read_csv(TRI)
    proc = proc.merge(tri, on=["country", "year"], how="left")
    proc["triadic_pc_log"] = np.log1p(proc["triadic"] / (proc["population"] / 1e6))
    return raw, proc


def placebo_frame(raw, proc, reg, outcome, k):
    """미래 앵커 t 의 PISA → 과거 t-k 의 outcome. 통제는 t."""
    anchors = raw[raw["year"].isin(ANCHORS)].dropna(subset=[reg])[["country", "year", reg]].copy()
    anchors = anchors.rename(columns={reg: "X_pisa", "year": "t"})
    anchors["ty"] = anchors["t"] - k
    ctrl = proc[["country", "year", *CONTROLS]].rename(columns={"year": "t"})
    out = proc[["country", "year", outcome]].rename(columns={"year": "ty", outcome: "Y"})
    df = anchors.merge(ctrl, on=["country", "t"], how="left").merge(
        out, on=["country", "ty"], how="left"
    )
    df = df[df["ty"] >= YEAR_MIN]
    df = df.dropna(subset=["X_pisa", "Y", *CONTROLS, "country"]).reset_index(drop=True)
    return df


def run_pooled(df):
    if len(df) < 8 or df["country"].nunique() < 5:
        return None
    yz, xz = z(df["Y"]), z(df["X_pisa"])
    Xc = pd.DataFrame({"pisa": xz, "rd": z(df["rd_gdp_pct"]), "gdp": z(df["gdp_per_capita"])})
    m = sm.OLS(yz, sm.add_constant(Xc)).fit(
        cov_type="cluster", cov_kwds={"groups": df["country"].values}
    )
    return dict(
        beta_std=round(float(m.params["pisa"]), 4),
        se_std=round(float(m.bse["pisa"]), 4),
        p=round(float(m.pvalues["pisa"]), 4),
        n_obs=int(len(df)),
        n_countries=int(df["country"].nunique()),
        cohorts=sorted(int(t) for t in df["t"].unique()),
        outcome_years=sorted(int(t) for t in df["ty"].unique()),
    )


def main():
    raw, proc = load()
    hashes = {
        "raw_dataset.csv": file_hash(RAW),
        "processed_dataset.csv": file_hash(PROC),
        "external_triadic_oecd.csv": file_hash(TRI),
    }
    print("=" * 90)
    print("future-PISA placebo (미래 PISA → 과거 혁신산출 반증검정)")
    print(f"앵커 {ANCHORS} · 패널시작 {YEAR_MIN} · 통제 {CONTROLS} · 표준화 β")
    print("=" * 90)
    res = {
        "_design": "PastOutput_{t-k} ~ FuturePISA_{t} + rd_gdp + gdp_pc (raw anchors)",
        "_lags_back": LAGS_BACK,
        "_input_hash": hashes,
        "by_regressor": {},
    }
    for reg in PISA_REGRESSORS:
        res["by_regressor"][reg] = {}
        for k in LAGS_BACK:
            feasible = [a for a in ANCHORS if a - k >= YEAR_MIN]
            print(f"\n── 회귀자 {reg} · k={k}년 역방향 (실현가능 앵커 {feasible}) ──")
            if not feasible:
                res["by_regressor"][reg][f"k{k}"] = {"feasible": False}
                print("   ✗ INFEASIBLE")
                continue
            kr = {"feasible": True, "feasible_anchors": feasible, "outcomes": {}}
            for outcome, desc in OUTCOMES:
                df = placebo_frame(raw, proc, reg, outcome, k)
                r = run_pooled(df)
                if r is None:
                    print(f"   {outcome:<20} n={len(df):>3} → 표본부족, 스킵")
                    kr["outcomes"][outcome] = {
                        "_skipped": True,
                        "n_obs": int(len(df)),
                        "_desc": desc,
                    }
                    continue
                sig = (
                    "***"
                    if r["p"] < 0.001
                    else "**"
                    if r["p"] < 0.01
                    else "*"
                    if r["p"] < 0.05
                    else "n.s."
                )
                print(
                    f"   {outcome:<20} β_std={r['beta_std']:+.3f} (SE {r['se_std']:.3f}, p={r['p']:.3f} {sig:>4}) "
                    f"n={r['n_obs']} {r['n_countries']}국 코호트{r['cohorts']}→과거{r['outcome_years']}"
                )
                r["_desc"] = desc
                kr["outcomes"][outcome] = r
            res["by_regressor"][reg][f"k{k}"] = kr
    res["_interpretation"] = (
        "placebo β 를 pisa_cohort_lag.py 의 forward β 와 같은 호라이즌에서 비교한다. 둘이 비슷하게 "
        "강하면(예: forward h=6 ≈ placebo k=6) PISA 계수는 시간 선행 신호가 아니라 국가 장기특성 로딩 "
        "가능성이 크다 — between 분산 95%·국가IQ 안정성과 정합. 본 표본(1~3코호트)에서 forward 와 placebo "
        "가 모두 유의하면 '시간 선행성을 갖춘 연관'을 약하게만 주장할 수 있고, between 횡단 해석이 더 안전하다."
    )
    print("\n" + "=" * 90)
    print("[JSON 동결용 요약]")
    print(json.dumps(res, ensure_ascii=False))
    print("\n### future_pisa_placebo 완료 (read-only)")
    return res


if __name__ == "__main__":
    main()
