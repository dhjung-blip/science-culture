"""외부 outcome 회귀 사양 — country-mean(주) + pooled-cluster(보조) + cohort-lag/placebo/민감도.

사양은 정본 direct_output_robustness.py 와 동일 정렬:
  처치(인지 인적자본 앵커) = sci_literacy (z)
  통제(confounder-only)    = rd_gdp_pct, gdp_per_capita (z)  — researchers_per_m(매개) 제외
  주 estimand             = 국가평균 between-only OLS (44국 횡단)
  보조 robustness         = pooled OLS + 국가 클러스터 SE (정본 cross β 와 동일 추정량)
인과·시간선행 주장 없음. placebo 는 boundary-setting, cohort-lag 는 exploratory.
모든 무작위 절차 seed=42.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

ANCHOR = "sci_literacy"
CONTROLS = ["rd_gdp_pct", "gdp_per_capita"]
PISA_YEARS = [2012, 2015, 2018, 2022]
YEAR_MAX = 2023
SEED = 42


def z(s: pd.Series | np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd else s * 0.0


def _round(x: float | None, n: int = 4) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def country_mean_between(df: pd.DataFrame, ycol: str) -> dict | None:
    """주 estimand: 국가평균 횡단(between-only) 표준화 OLS. anchor=sci_literacy 계수 반환."""
    cols = [ycol, ANCHOR, *CONTROLS, "country"]
    sub = df.dropna(subset=cols)[cols].copy()
    if sub.empty:
        return None
    cm = sub.groupby("country").mean(numeric_only=True)
    if len(cm) < 8:
        return None
    X = pd.DataFrame(
        {"anchor": z(cm[ANCHOR]), "rd": z(cm["rd_gdp_pct"]), "gdp": z(cm["gdp_per_capita"])}
    )
    m = sm.OLS(z(cm[ycol]), sm.add_constant(X)).fit()
    return dict(
        spec="country_mean_between_only",
        beta_std=_round(m.params["anchor"]),
        se=_round(m.bse["anchor"]),
        p=_round(m.pvalues["anchor"]),
        n_obs=int(len(cm)),
        n_countries=int(len(cm)),
        r2=_round(m.rsquared, 3),
        countries=sorted(cm.index.tolist()),
    )


def pooled_cluster(df: pd.DataFrame, ycol: str) -> dict | None:
    """보조: pooled OLS + 국가 클러스터 SE (정본 cross β 와 동일 추정량)."""
    cols = [ycol, ANCHOR, *CONTROLS, "country"]
    sub = df.dropna(subset=cols)[cols].copy()
    if len(sub) < 8 or sub["country"].nunique() < 5:
        return None
    X = pd.DataFrame(
        {"anchor": z(sub[ANCHOR]), "rd": z(sub["rd_gdp_pct"]), "gdp": z(sub["gdp_per_capita"])}
    )
    m = sm.OLS(z(sub[ycol]), sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": sub["country"].values}
    )
    return dict(
        spec="pooled_country_cluster_se",
        beta_std=_round(m.params["anchor"]),
        se=_round(m.bse["anchor"]),
        p=_round(m.pvalues["anchor"]),
        n_obs=int(len(sub)),
        n_countries=int(sub["country"].nunique()),
    )


def cohort_lag(df: pd.DataFrame, ycol: str, h: int = 6) -> dict | None:
    """exploratory: PISA(sci_literacy) at t (PISA years) → outcome at t+h. pooled cluster SE."""
    anch = (
        df[df["year"].isin(PISA_YEARS)].dropna(subset=[ANCHOR])[["country", "year", ANCHOR]].copy()
    )
    anch = anch.rename(columns={"year": "t", ANCHOR: "anchor_t"})
    anch["ty"] = anch["t"] + h
    ctrl = df[["country", "year", *CONTROLS]].rename(columns={"year": "t"})
    out = df[["country", "year", ycol]].rename(columns={"year": "ty", ycol: "Y"})
    d = anch.merge(ctrl, on=["country", "t"]).merge(out, on=["country", "ty"])
    d = d[d["ty"] <= YEAR_MAX].dropna(subset=["anchor_t", "Y", *CONTROLS]).reset_index(drop=True)
    if len(d) < 8 or d["country"].nunique() < 5:
        return None
    X = pd.DataFrame(
        {"anchor": z(d["anchor_t"]), "rd": z(d["rd_gdp_pct"]), "gdp": z(d["gdp_per_capita"])}
    )
    m = sm.OLS(z(d["Y"]), sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": d["country"].values}
    )
    return dict(
        spec=f"cohort_lag_h{h}_exploratory",
        beta_std=_round(m.params["anchor"]),
        p=_round(m.pvalues["anchor"]),
        n_obs=int(len(d)),
        n_countries=int(d["country"].nunique()),
        cohorts=sorted(int(t) for t in d["t"].unique()),
    )


def placebo_future_anchor(df: pd.DataFrame, ycol: str, h: int = 6) -> dict | None:
    """boundary-setting(식별 아님): future PISA(t+h) → past outcome(t). 시간선행 위배 검정."""
    fut = (
        df[df["year"].isin([y for y in PISA_YEARS])]
        .dropna(subset=[ANCHOR])[["country", "year", ANCHOR]]
        .copy()
    )
    fut = fut.rename(columns={"year": "tf", ANCHOR: "anchor_future"})
    fut["t"] = fut["tf"] - h  # outcome 시점은 anchor 보다 h 앞(과거)
    ctrl = df[["country", "year", *CONTROLS]].rename(columns={"year": "t"})
    out = df[["country", "year", ycol]].rename(columns={"year": "t", ycol: "Y"})
    d = fut.merge(out, on=["country", "t"]).merge(ctrl, on=["country", "t"])
    d = (
        d[(d["t"] >= 2012) & (d["t"] <= YEAR_MAX)]
        .dropna(subset=["anchor_future", "Y", *CONTROLS])
        .reset_index(drop=True)
    )
    if len(d) < 8 or d["country"].nunique() < 5:
        return None
    X = pd.DataFrame(
        {"anchor": z(d["anchor_future"]), "rd": z(d["rd_gdp_pct"]), "gdp": z(d["gdp_per_capita"])}
    )
    m = sm.OLS(z(d["Y"]), sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": d["country"].values}
    )
    return dict(
        spec=f"placebo_future_anchor_h{h}_boundary",
        beta_std=_round(m.params["anchor"]),
        p=_round(m.pvalues["anchor"]),
        n_obs=int(len(d)),
        n_countries=int(d["country"].nunique()),
        _note="boundary-setting only; not identification",
    )


# 대륙/지역 그룹 (leave-region 민감도)
REGION = {
    "EU_W": {"AUT", "BEL", "DEU", "FRA", "NLD", "LUX", "IRL", "GBR", "CHE"},
    "EU_N": {"DNK", "FIN", "ISL", "NOR", "SWE", "EST", "LVA", "LTU"},
    "EU_S": {"ESP", "GRC", "ITA", "PRT"},
    "EU_E": {"CZE", "HUN", "POL", "SVK", "SVN", "RUS"},
    "AMER": {"BRA", "CAN", "CHL", "COL", "CRI", "MEX", "PER", "USA"},
    "ASIA_OC": {"AUS", "HKG", "IDN", "JPN", "KOR", "NZL", "SGP", "TUR", "ISR"},
}


def leave_region(df: pd.DataFrame, ycol: str) -> dict | None:
    """민감도: 한 지역씩 제외하고 country-mean β 재추정 → 범위 보고."""
    betas = []
    for rname, members in REGION.items():
        sub = df[~df["country"].isin(members)]
        r = country_mean_between(sub, ycol)
        if r:
            betas.append((rname, r["beta_std"], r["p"]))
    if len(betas) < 3:
        return None
    bvals = [b for _r, b, _p in betas if b is not None]
    return dict(
        spec="leave_one_region_country_mean",
        beta_min=_round(min(bvals)),
        beta_max=_round(max(bvals)),
        all_positive=all(b > 0 for b in bvals),
        all_sig=all(p is not None and p < 0.05 for _r, _b, p in betas),
        drops=[{"region": r, "beta": b, "p": p} for r, b, p in betas],
    )


def denominator_separation(df: pd.DataFrame, key: str) -> dict | None:
    """per-capita 결과의 분모분리: log(raw count)·log(pop) 를 분리 투입(country-mean)."""
    lograw, logpop = f"{key}__log_raw", "log_pop"
    cols = [lograw, logpop, ANCHOR, *CONTROLS, "country"]
    sub = df.dropna(subset=cols)[cols].copy()
    if sub.empty:
        return None
    cm = sub.groupby("country").mean(numeric_only=True)
    if len(cm) < 10:
        return None
    X = pd.DataFrame(
        {
            "anchor": z(cm[ANCHOR]),
            "rd": z(cm["rd_gdp_pct"]),
            "gdp": z(cm["gdp_per_capita"]),
            "logpop": z(cm[logpop]),
        }
    )
    m = sm.OLS(z(cm[lograw]), sm.add_constant(X)).fit()
    return dict(
        spec="denominator_separation_country_mean",
        beta_anchor_on_lograw=_round(m.params["anchor"]),
        p_anchor=_round(m.pvalues["anchor"]),
        beta_logpop=_round(m.params["logpop"]),
        n_countries=int(len(cm)),
        _note="numerator(log raw count) on anchor with log(population) separated as covariate",
    )


def bh_fdr(pairs: list[tuple[str, float]]) -> list[dict]:
    """Benjamini-Hochberg FDR. pairs=[(label, p), ...] → [{label,p,q,reject_05}]."""
    valid = [(lab, p) for lab, p in pairs if p is not None]
    m = len(valid)
    order = sorted(range(m), key=lambda i: valid[i][1])
    q = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        val = valid[i][1] * m / k
        prev = min(prev, val)
        q[i] = prev
    return [
        {
            "label": valid[i][0],
            "p": _round(valid[i][1]),
            "q_fdr": _round(q[i]),
            "reject_05": q[i] < 0.05,
        }
        for i in range(m)
    ]
