"""식별전략 — PISA 코호트-시차(cohort-lag) 시간 선행성 검정.

PISA 를 15세 학생 코호트의 *미래 인적자본 stock* 으로 보고, PISA 실측 시점 t 이후
h(=6/9/12)년 뒤 혁신 산출을 예측하는지 검정한다(Hanushek·Woessmann 인적자본 전통).

    Output_{i,t+h} = β·PISA_{i,t} + γ·X_{i,t} + ε_{i,t+h}

핵심 설계 결정: 회귀자(PISA)는 **raw_dataset.csv 의 실측 앵커**(2012·2015·2018·2022)만 사용한다.
processed_dataset.csv 의 PISA 는 앵커 사이를 선형보간한 값이라(within 변동의 ~69%가 보간 인공물,
본문 §5.4(2)), 보간된 PISA 를 회귀자로 쓰면 '시간 선행성'이 측정 절차가 만든 매끄러움일 수 있다.
앵커만 쓰면 이 인공물이 제거되는 대신 표본이 급감한다(아래 coverage 참조) — 그 trade-off 를 정직하게 보고한다.

식별 한계(엄수): 본 검정은 인과가 아니라 '시간 선행성을 갖춘 조건부 연관'을 본다. 표본이
사실상 1~2개 PISA 코호트로 줄어 국가 고정효과(FE)는 추정 불능/불안정하므로 pooled(국가
클러스터 SE)를 1차로 보고하고 FE 는 자유도 부족을 명시한다. 결과가 약하면 약하다고 보고한다.

read-only: 동결 raw_dataset.csv + processed_dataset.csv + external_*.csv. stdout + JSON 동결용.
재현: cd <package-root>; python analysis/pisa_cohort_lag.py
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
TFP = "analysis/canonical/external_tfp_pwt110.csv"

ANCHORS = [2012, 2015, 2018, 2022]  # PISA 실측 시행연도
YEAR_MAX = 2023  # 동결 패널 마지막 연도
HORIZONS = [6, 9, 12]
CONTROLS = ["rd_gdp_pct", "gdp_per_capita"]  # confounder-only: 연구원 밀도는 매개변수로 제외
PISA_REGRESSORS = ["sci_literacy", "pisa_math"]


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:16]


def z(s):
    s = np.asarray(s, dtype=float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd else s * 0.0


def load():
    raw = pd.read_csv(RAW)
    proc = pd.read_csv(PROC)
    # 외부 산출 타깃 머지 (processed 에 per-capita 변환 부착)
    tri = pd.read_csv(TRI)
    proc = proc.merge(tri, on=["country", "year"], how="left")
    proc["triadic_pc_log"] = np.log1p(proc["triadic"] / (proc["population"] / 1e6))
    tfp = pd.read_csv(TFP).rename(columns={"countrycode": "country"})
    proc = proc.merge(tfp[["country", "year", "ctfp"]], on=["country", "year"], how="left")
    return raw, proc


# 결과변수 우선순위(사전 지정 7단계): 삼극 > [PCT n/a] > [top-cited n/a] > WB특허 > WB논문 > GII산출서브(별도) > GII종합
OUTCOMES = [
    ("triadic_pc_log", "OECD 삼극특허 패밀리 per-capita 로그(외부준거·1순위; GII-PCT와 출처 분리)"),
    ("patents_per_m_log", "World Bank 특허출원 per-capita 로그(비합성 직접 산출)"),
    ("articles_per_m_log", "World Bank SCI논문 per-capita 로그(비합성 직접 산출)"),
    ("innovation_score", "WIPO GII 종합점수(순환 포함 benchmark/대조군 — primary 아님)"),
    ("ctfp", "Penn World Table TFP 수준(GII 완전독립 외부준거·먼 거시결과)"),
]


def cohort_frame(raw, proc, regressor, outcome, h):
    """앵커 t 의 PISA → t+h 의 outcome. 통제·outcome 은 proc 에서 (i,t)·(i,t+h) 로 조인."""
    anchors = (
        raw[raw["year"].isin(ANCHORS)]
        .dropna(subset=[regressor])[["country", "year", regressor]]
        .copy()
    )
    anchors = anchors.rename(columns={regressor: "X_pisa", "year": "t"})
    anchors["ty"] = anchors["t"] + h
    # 통제는 앵커연도 t (PISA 와 동시점), outcome 은 t+h
    ctrl = proc[["country", "year", *CONTROLS]].rename(columns={"year": "t"})
    out = proc[["country", "year", outcome]].rename(columns={"year": "ty", outcome: "Y"})
    df = anchors.merge(ctrl, on=["country", "t"], how="left").merge(
        out, on=["country", "ty"], how="left"
    )
    df = df[df["ty"] <= YEAR_MAX]
    df = df.dropna(subset=["X_pisa", "Y", *CONTROLS, "country"]).reset_index(drop=True)
    return df


def run_pooled(df):
    """pooled OLS, 국가 클러스터 강건 SE. 표준화 β + 원단위 β."""
    if len(df) < 8 or df["country"].nunique() < 5:
        return None
    yz, xz = z(df["Y"]), z(df["X_pisa"])
    Xc = pd.DataFrame({"pisa": xz, "rd": z(df["rd_gdp_pct"]), "gdp": z(df["gdp_per_capita"])})
    m = sm.OLS(yz, sm.add_constant(Xc)).fit(
        cov_type="cluster", cov_kwds={"groups": df["country"].values}
    )
    # 원단위(PISA 1점당 outcome 변화)
    Xr = pd.DataFrame(
        {
            "pisa": df["X_pisa"].values,
            "rd": df["rd_gdp_pct"].values,
            "gdp": df["gdp_per_capita"].values,
        }
    )
    mr = sm.OLS(df["Y"].values, sm.add_constant(Xr)).fit(
        cov_type="cluster", cov_kwds={"groups": df["country"].values}
    )
    return dict(
        beta_std=round(float(m.params["pisa"]), 4),
        se_std=round(float(m.bse["pisa"]), 4),
        p=round(float(m.pvalues["pisa"]), 4),
        beta_rawunit=round(float(mr.params["pisa"]), 6),
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
    print("PISA 코호트-시차(cohort-lag) 시간 선행성 검정 (raw 앵커 회귀자)")
    print(f"앵커연도 {ANCHORS} · 패널끝 {YEAR_MAX} · 통제 {CONTROLS}(confounder-only) · 표준화 β")
    print("=" * 90)

    results = {
        "_design": "Output_{t+h} ~ PISA_{t} + rd_gdp + gdp_pc (raw PISA anchors only)",
        "_horizons_requested": HORIZONS,
        "_anchors": ANCHORS,
        "_input_hash": hashes,
        "by_regressor": {},
    }

    for reg in PISA_REGRESSORS:
        results["by_regressor"][reg] = {}
        for h in HORIZONS:
            # 실현 가능 코호트(앵커 + h ≤ 2023)
            feasible = [a for a in ANCHORS if a + h <= YEAR_MAX]
            print(f"\n── 회귀자 {reg} · h={h}년 (실현가능 앵커 {feasible}) ──")
            if not feasible:
                print(
                    f"   ✗ INFEASIBLE: 어떤 앵커도 +{h} 가 패널({YEAR_MAX}) 안에 떨어지지 않음 → 미실행."
                )
                results["by_regressor"][reg][f"h{h}"] = {
                    "feasible": False,
                    "_note": f"h={h} 불가: 최이른 앵커 2012+{h}={2012 + h}>{YEAR_MAX}. 동결 패널이 12년이라 장기 시차는 데이터 한계로 미실행.",
                }
                continue
            hres = {"feasible": True, "feasible_anchors": feasible, "outcomes": {}}
            for outcome, desc in OUTCOMES:
                df = cohort_frame(raw, proc, reg, outcome, h)
                r = run_pooled(df)
                if r is None:
                    print(f"   {outcome:<20} n={len(df):>3} → 표본부족, 스킵")
                    hres["outcomes"][outcome] = {
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
                    f"n={r['n_obs']} {r['n_countries']}국 코호트{r['cohorts']}→{r['outcome_years']}"
                )
                r["_desc"] = desc
                r["_fe_note"] = (
                    "FE 불능: 단일 코호트(국가당 1관측) → 국가 내 변동 없음"
                    if len(r["cohorts"]) == 1
                    else "FE 불안정: 국가당 ~2관측, within 자유도 1 → pooled 만 신뢰"
                )
                hres["outcomes"][outcome] = r
            results["by_regressor"][reg][f"h{h}"] = hres

    results["_interpretation"] = (
        "표본이 1~2 PISA 코호트로 줄어 사실상 국가 간(between) 횡단 추정이다. 유의해도 '시간 선행성을 "
        "갖춘 조건부 연관'이며 인과가 아니다. h=12 는 동결 패널 12년 한계로 미실행. future_pisa_placebo.py 의 "
        "반증 결과와 함께 해석해야 한다 — placebo 가 강하면 본 계수도 국가 장기특성 로딩일 수 있다."
    )
    print("\n" + "=" * 90)
    print("[JSON 동결용 요약]")
    print(json.dumps(results, ensure_ascii=False))
    print("\n### pisa_cohort_lag 완료 (read-only)")
    return results


if __name__ == "__main__":
    main()
