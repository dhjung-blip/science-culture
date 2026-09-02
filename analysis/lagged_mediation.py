"""식별전략 — 시차 매개분석(lagged mediation), 탐색적.

본 분석의 Baron–Kenny 횡단 매개(GII 49.1%[32,67]·동결 직접산출 논문 +21%·특허 불안정)를 *대체가
아니라 보완*한다. 시간 순서를 부여한 구조:

    PISA_{i,2012} → Researchers_{i,2018} → Output_{i,2021}

a path: PISA_2012 → researchers_2018 ; b path: researchers_2018 → output_2021 (PISA 통제) ;
total: PISA_2012 → output_2021 ; direct: total | researchers_2018 ; indirect = a·b ; prop = indirect/total.
국가 부트스트랩 CI(나라 단위 재표집). PISA 는 raw 2012 앵커만 사용.

한계(엄수): 동결 패널이 12년이라 t+12 산출(2024)은 불가 → t+9(2021) 단일 코호트만 가능. 따라서
표본은 ~38국 단일 코호트로, 순차적 무시가능성(sequential ignorability)·시간선행이 매개경로 전체에
대해 검증되지 않는 **탐색적** 분석이다. future_pisa_placebo 가 강하므로(시간선행 약함) 본 매개도
관찰적 경로분해로만 해석한다. 결과가 불안정하면 그대로 보고한다.

read-only: 동결 데이터. stdout + JSON 동결용.
재현: cd <package-root>; python analysis/lagged_mediation.py
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

PISA_YEAR = 2012
MED_YEAR = 2018  # researchers t+6
OUT_YEAR = 2021  # output t+9 (t+12=2024 동결패널 밖 → 불가)
CONTROLS = ["rd_gdp_pct", "gdp_per_capita"]
N_BOOT = 2000
SEED = 42
OUTCOMES = ["patents_per_m_log", "articles_per_m_log"]


def file_hash(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:16]


def z(s):
    s = np.asarray(s, dtype=float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd else s * 0.0


def build(outcome):
    raw = pd.read_csv(RAW)
    proc = pd.read_csv(PROC)
    pisa = raw[(raw["year"] == PISA_YEAR)].dropna(subset=["sci_literacy"])[
        ["country", "sci_literacy"]
    ]
    med = proc[proc["year"] == MED_YEAR][["country", "researchers_per_m", *CONTROLS]]
    out = proc[proc["year"] == OUT_YEAR][["country", outcome]].rename(columns={outcome: "Y"})
    df = pisa.merge(med, on="country").merge(out, on="country")
    return df.dropna().reset_index(drop=True)


def mediation(df):
    """완전표준화 a·b·total·direct·indirect·proportion."""
    X, M, Y = z(df["sci_literacy"]), z(df["researchers_per_m"]), z(df["Y"])
    C = pd.DataFrame({"rd": z(df["rd_gdp_pct"]), "gdp": z(df["gdp_per_capita"])})
    a = sm.OLS(M, sm.add_constant(pd.concat([pd.Series(X, name="X"), C], axis=1))).fit().params["X"]
    mb = sm.OLS(
        Y, sm.add_constant(pd.concat([pd.Series(M, name="M"), pd.Series(X, name="X"), C], axis=1))
    ).fit()
    b, direct = mb.params["M"], mb.params["X"]
    total = (
        sm.OLS(Y, sm.add_constant(pd.concat([pd.Series(X, name="X"), C], axis=1))).fit().params["X"]
    )
    indirect = a * b
    prop = indirect / total if total else np.nan
    return dict(a=a, b=b, total=total, direct=direct, indirect=indirect, prop=prop)


def boot_ci(df, reps, seed):
    rng = np.random.default_rng(seed)
    countries = df["country"].unique()
    props, inds = [], []
    for _ in range(reps):
        samp = rng.choice(countries, size=len(countries), replace=True)
        bdf = pd.concat([df[df["country"] == c] for c in samp], ignore_index=True)
        try:
            r = mediation(bdf)
            if np.isfinite(r["prop"]):
                props.append(r["prop"])
                inds.append(r["indirect"])
        except Exception:
            continue

    def ci(v):
        return (
            [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            if v
            else None
        )

    return ci(props), ci(inds), len(props)


def main():
    hashes = {"raw_dataset.csv": file_hash(RAW), "processed_dataset.csv": file_hash(PROC)}
    print("=" * 90)
    print(f"시차 매개(탐색): PISA_{PISA_YEAR} → researchers_{MED_YEAR} → output_{OUT_YEAR}")
    print(f"부트스트랩 {N_BOOT}회(국가 재표집) · seed {SEED} · 완전표준화 · t+12 불가(동결패널 밖)")
    print("=" * 90)
    res = {
        "_design": f"PISA_{PISA_YEAR} -> researchers_{MED_YEAR} -> output_{OUT_YEAR}",
        "_n_boot": N_BOOT,
        "_seed": SEED,
        "_input_hash": hashes,
        "_t12_infeasible": "산출 t+12=2024 는 동결 패널(끝 2023) 밖 → t+9(2021) 단일 코호트만 실행",
        "by_outcome": {},
    }
    for outcome in OUTCOMES:
        df = build(outcome)
        pt = mediation(df)
        pci, ici, nb = boot_ci(df, N_BOOT, SEED)
        sig = pci is not None and not (pci[0] <= 0 <= pci[1])
        print(f"\n── {outcome} (n={len(df)}국 단일 코호트) ──")
        print(
            f"   a(X→M)={pt['a']:+.3f}  b(M→Y)={pt['b']:+.3f}  total={pt['total']:+.3f}  direct={pt['direct']:+.3f}"
        )
        print(
            f"   indirect(a·b)={pt['indirect']:+.3f}  prop={100 * pt['prop']:+.1f}%  "
            f"boot prop CI={[round(100 * x, 1) for x in pci] if pci else None}%  유의={sig}"
        )
        res["by_outcome"][outcome] = {
            "a_path": round(float(pt["a"]), 4),
            "b_path": round(float(pt["b"]), 4),
            "total": round(float(pt["total"]), 4),
            "direct": round(float(pt["direct"]), 4),
            "indirect": round(float(pt["indirect"]), 4),
            "prop_mediated_pct": round(float(100 * pt["prop"]), 1),
            "prop_ci_pct": [round(100 * x, 1) for x in pci] if pci else None,
            "indirect_ci": ici,
            "boot_valid_reps": nb,
            "ci_excludes_zero": bool(sig),
            "n_countries": int(len(df)),
            "_note": "단일 코호트(국가당 1관측) 탐색적 시차매개. 시간선행은 a·b·total 의 연도순서로만 "
            "확보되고 순차적 무시가능성은 미검증. CI 가 넓거나 0 포함이면 불안정으로 보고.",
        }
    res["_compare_v2"] = (
        "횡단 매개(주 사양): GII 49.1%[32.2,67.2](상향편의·매개변수 M⊂Y), 동결 직접산출 "
        "논문 +21.0%[5.9,44.4] 유의·특허 −13.9%[−61.6,36.0] 불안정. 본 시차매개는 더 "
        "낮은 강도의 보조증거로 위치 — 시차를 줘도 표본 한계로 탐색적임을 명시."
    )
    print("\n" + "=" * 90)
    print("[JSON 동결용 요약]")
    print(json.dumps(res, ensure_ascii=False))
    print("\n### lagged_mediation 완료 (read-only)")
    return res


if __name__ == "__main__":
    main()
