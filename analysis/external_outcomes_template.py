"""외부 준거 결과변수 — ingestion 템플릿 + (있으면) 분석.

data/external/ 의 CSV 를 자동 감지해, 외부 산출 결과변수에서 인지 인적자본(PISA)–혁신산출 연관을
재검한다. 파일이 없으면 빈 결과 + TODO 만 기록한다(데이터 도착 시 곧바로 분석되도록 계약만 고정).

기대 파일(schema 는 data/external/README.md):
  - triadic_patents.csv  (country_iso3, year, triadic_patent_families)        [1순위·현재 제공]
  - pct_applications.csv (country_iso3, year, pct_applications_origin)         [미제공→TODO]
  - openalex_top_cited.csv (country_iso3, year, top10_cited_publications, ...) [미제공→TODO]

변환: raw / per_million / per_researcher / log1p(raw) / log1p(per_million).
사양: confounder-only(R&D/GDP·1인당GDP 통제, 연구원 밀도=매개변수 제외), 횡단 pooled(국가 클러스터 SE)
      + cohort-lag h=6(raw PISA 앵커). 표준화 β.

read-only: 동결 processed_dataset.csv + data/external/*.csv. stdout + JSON 동결용.
재현: cd <package-root>; python analysis/external_outcomes_template.py
"""

import hashlib
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROC = "analysis/canonical/processed_dataset.csv"
RAW = "analysis/canonical/raw_dataset.csv"
EXT_DIR = "data/external"
CONTROLS = ["rd_gdp_pct", "gdp_per_capita"]
ANCHORS = [2012, 2015, 2018, 2022]
YEAR_MAX = 2023

# (파일, count컬럼, 라벨, 메모)
SPECS = [
    (
        "triadic_patents.csv",
        "triadic_patent_families",
        "OECD 삼극특허 패밀리",
        "1순위 외부준거. GII-PCT와 출처 분리. 외부 준거 분석에서 검증됨(횡단 +0.189·δ1.996).",
    ),
    (
        "pct_applications.csv",
        "pct_applications_origin",
        "WIPO PCT 출원(origin)",
        "미제공 시 TODO. 후보: WIPO IP Statistics Data Center.",
    ),
    (
        "openalex_top_cited.csv",
        "top10_cited_publications",
        "OpenAlex 상위10% 피인용 논문",
        "미제공 시 TODO. 분야·연도 정규화 필수. 후보: OpenAlex snapshot/API.",
    ),
]


def file_hash(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:16]


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def z(s):
    s = np.asarray(s, dtype=float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd else s * 0.0


def transforms(df, count_col):
    """5개 변환 컬럼을 df 에 부착(분모: population·researchers_per_m)."""
    pop_m = df["population"] / 1e6
    researchers = df["researchers_per_m"] * pop_m  # 총 연구원 수
    out = {}
    out["raw"] = df[count_col]
    out["per_million"] = df[count_col] / pop_m
    out["per_researcher"] = df[count_col] / researchers.replace(0, np.nan)
    out["log1p_raw"] = np.log1p(df[count_col].clip(lower=0))
    out["log1p_per_million"] = np.log1p((df[count_col] / pop_m).clip(lower=0))
    return out


def pooled(y, df):
    sub = pd.DataFrame(
        {
            "y": y,
            "pisa": df["sci_literacy"],
            "rd": df["rd_gdp_pct"],
            "gdp": df["gdp_per_capita"],
            "country": df["country"],
        }
    ).dropna()
    if len(sub) < 8 or sub["country"].nunique() < 5:
        return None
    X = pd.DataFrame({"pisa": z(sub["pisa"]), "rd": z(sub["rd"]), "gdp": z(sub["gdp"])})
    m = sm.OLS(z(sub["y"]), sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": sub["country"].values}
    )
    return dict(
        beta_std=round(float(m.params["pisa"]), 4),
        se=round(float(m.bse["pisa"]), 4),
        p=round(float(m.pvalues["pisa"]), 4),
        n=int(len(sub)),
        n_countries=int(sub["country"].nunique()),
    )


def cohort_lag_h6(raw, merged, ycol):
    """raw PISA 앵커 t → ycol at t+6 (2012→2018·2015→2021)."""
    anchors = (
        raw[raw["year"].isin(ANCHORS)]
        .dropna(subset=["sci_literacy"])[["country", "year", "sci_literacy"]]
        .copy()
    )
    anchors = anchors.rename(columns={"year": "t"})
    anchors["ty"] = anchors["t"] + 6
    ctrl = merged[["country", "year", *CONTROLS]].rename(columns={"year": "t"})
    out = merged[["country", "year", ycol]].rename(columns={"year": "ty", ycol: "Y"})
    df = anchors.merge(ctrl, on=["country", "t"]).merge(out, on=["country", "ty"])
    df = (
        df[(df["ty"] <= YEAR_MAX)]
        .dropna(subset=["sci_literacy", "Y", *CONTROLS])
        .reset_index(drop=True)
    )
    if len(df) < 8 or df["country"].nunique() < 5:
        return None
    X = pd.DataFrame(
        {"pisa": z(df["sci_literacy"]), "rd": z(df["rd_gdp_pct"]), "gdp": z(df["gdp_per_capita"])}
    )
    m = sm.OLS(z(df["Y"]), sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": df["country"].values}
    )
    return dict(
        beta_std=round(float(m.params["pisa"]), 4),
        p=round(float(m.pvalues["pisa"]), 4),
        n=int(len(df)),
        n_countries=int(df["country"].nunique()),
        cohorts=sorted(int(t) for t in df["t"].unique()),
    )


def main():
    proc = pd.read_csv(PROC)
    raw = pd.read_csv(RAW)
    print("=" * 90)
    print("외부 준거 결과변수 ingestion + 분석 (data/external/*.csv 자동감지)")
    print("=" * 90)
    res = {
        "_design": "external outcomes; confounder-only pooled + cohort-lag h6; PISA=sci_literacy",
        "_transforms": ["raw", "per_million", "per_researcher", "log1p_raw", "log1p_per_million"],
        "_input_hash": {"processed_dataset.csv": file_hash(PROC)},
        "outcomes": {},
    }
    for fname, ccol, label, memo in SPECS:
        path = os.path.join(EXT_DIR, fname)
        if not os.path.exists(path):
            print(f"\n[TODO] {label}: {path} 없음 → 분석 스킵, TODO 기록")
            res["outcomes"][fname] = {
                "status": "TODO",
                "label": label,
                "_note": memo,
                "expected_path": path,
            }
            continue
        ext = pd.read_csv(path)
        if "country_iso3" in ext.columns:
            ext = ext.rename(columns={"country_iso3": "country"})
        res["_input_hash"][fname] = file_hash(path)
        merged = proc.merge(ext, on=["country", "year"], how="left")
        tcols = transforms(merged, ccol)
        print(f"\n[분석] {label} ({path}, n_nonnull={merged[ccol].notna().sum()})")
        ores = {"status": "ANALYZED", "label": label, "_note": memo, "transforms": {}}
        for tname, yser in tcols.items():
            cross = pooled(yser, merged)
            mcol = f"__{ccol}__{tname}"
            merged[mcol] = yser
            lag = cohort_lag_h6(raw, merged, mcol)
            cs = (
                f"β={cross['beta_std']:+.3f} p={cross['p']:.3f} {sig(cross['p'])} n={cross['n']}"
                if cross
                else "표본부족"
            )
            ls = (
                f"β={lag['beta_std']:+.3f} p={lag['p']:.3f} {sig(lag['p'])} n={lag['n']}"
                if lag
                else "표본부족"
            )
            print(f"   {tname:<18} 횡단[{cs}]  cohort-lag h6[{ls}]")
            ores["transforms"][tname] = {"cross": cross, "cohort_lag_h6": lag}
        res["outcomes"][fname] = ores
    res["_summary"] = (
        "triadic 은 제공되어 분석됨(모든 변환에서 인지 인적자본 연관 유지 — 본 분석과 정합). "
        "PCT·OpenAlex 는 데이터 미제공으로 TODO. 데이터 도착 시 동일 스크립트가 자동 분석."
    )
    print("\n" + "=" * 90)
    print("[JSON 동결용 요약]")
    print(json.dumps(res, ensure_ascii=False))
    print("\n### external_outcomes_template 완료 (read-only)")
    return res


if __name__ == "__main__":
    main()
