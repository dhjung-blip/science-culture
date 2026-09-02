# -*- coding: utf-8 -*-
"""3단계 — GII Output Sub-Index 타깃 본 분석 (정공법 탈순환). [findings §7.8]
타깃: WIPO GII Innovation Output Sub-Index (Mendeley xrr862ssjd, CC BY 4.0, 2013-2022)
피처: 산출(특허·논문·hi_tech) 제외한 문화·투입·보상·인프라 → '투입·문화가 산출을 설명'
보조: manufacturing_va(WB NV.IND.MANF.ZS) robustness. read-only, /tmp만.

핵심 결과(재실행 검증 — findings §7.8 일치): sanity 우리GII↔xlsx 상관 1.0000(ISO3 결함 NAME2ISO 복구);
  GII Output within +0.385(p=0.12)·종합 +0.236(p=0.052) 방향 일치하나 미달(합성지수 sticky 함);
  문화 예측 우위는 직접측정 타깃만 — GII Output 은 투입과 비등(−0.03); 매개 41%(Sobel z=3.15**).

※ 보존본(2026-06): 원본은 /tmp에서 휘발 → 대화 기록 기반 복원, 재실행해 위 §7.8 수치와 일치 확인. analysis/README.md 참조.
   gii_dataset_DIB.xlsx 는 Mendeley(https://data.mendeley.com/datasets/xrr862ssjd/1)에서 받아
   ~/Downloads/ 에 둔다(ISO3 컬럼 시프트 결함 → Economy Name 기준 재매핑으로 복구)."""
import warnings
warnings.filterwarnings("ignore")
import re
import sys
import os as _os
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, _os.environ.get("SC_CODE_DIR") or str(_Path(__file__).resolve().parents[1] / "code"))
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import data_collector as d

CC = d.resolve_country_set("pisa-gii")
YEARS = d.YEARS

# ── GII Output Sub-Index 로드 (wide → long) ──
gx = pd.read_excel("os.environ.get('GII_DIB_XLSX', 'data/external/gii_dataset_DIB.xlsx')")
keep = {"Innovation Output Sub-Index": "gii_output", "Innovation Input Sub-Index": "gii_input",
        "Global Innovation Index": "gii_total_chk",
        "Knowledge and technology outputs index": "gii_ktech",
        "Creative outputs index": "gii_creative"}
# ⚠ 데이터 결함 복구: 원본 xlsx의 Economy ISO3 컬럼이 한 칸 밀려 있음(저자 버그,
# 예: Albania→ARE). Economy Name이 정답(2013 상위권이 공식 순위와 정확히 일치) →
# Name 기준 ISO3 재매핑. 우리 44국(pisa-gii)만 필요.
NAME2ISO = {
    "Australia": "AUS", "Austria": "AUT", "Belgium": "BEL", "Brazil": "BRA",
    "Canada": "CAN", "Switzerland": "CHE", "Chile": "CHL", "Colombia": "COL",
    "Costa Rica": "CRI", "Czechia": "CZE", "Germany": "DEU", "Denmark": "DNK",
    "Spain": "ESP", "Estonia": "EST", "Finland": "FIN", "France": "FRA",
    "United Kingdom": "GBR", "Greece": "GRC", "Hong Kong SAR, China": "HKG",
    "Hungary": "HUN", "Indonesia": "IDN", "Ireland": "IRL", "Iceland": "ISL",
    "Israel": "ISR", "Italy": "ITA", "Japan": "JPN", "Korea, Rep.": "KOR",
    "Lithuania": "LTU", "Luxembourg": "LUX", "Latvia": "LVA", "Mexico": "MEX",
    "Netherlands": "NLD", "Norway": "NOR", "New Zealand": "NZL", "Peru": "PER",
    "Poland": "POL", "Portugal": "PRT", "Russian Federation": "RUS",
    "Singapore": "SGP", "Slovak Republic": "SVK", "Slovenia": "SVN",
    "Sweden": "SWE", "Turkiye": "TUR", "United States": "USA",
}
rows = []
for _, r in gx.iterrows():
    iso = NAME2ISO.get(str(r["Economy Name"]))
    if iso is None:
        continue  # 분석 외 국가
    for c, v in r.items():
        m = re.match(r"^(\d{4})_(.+)$", str(c))
        if m and m.group(2) in keep:
            rows.append({"country": iso, "year": int(m.group(1)), keep[m.group(2)]: v})
sub = pd.DataFrame(rows).groupby(["country", "year"]).first().reset_index()
print(f"### GII Sub-Index 패널: {sub['country'].nunique()}국 {sub['year'].min()}-{sub['year'].max()}")

# ── 파이프라인 데이터 + 병합 ──
raw = d.build_full_dataset(countries=CC, years=YEARS, use_worldbank=True, use_fred=False, seed=42)
mva = d.fetch_worldbank("NV.IND.MANF.ZS", CC, YEARS).rename(columns={"value": "manufacturing_va"})
work = raw.merge(sub, on=["country", "year"], how="left").merge(mva, on=["country", "year"], how="left")

# sanity: 우리 innovation_score(GII 종합) ↔ 이 파일의 GII 종합/서브지수 정합
both = work.dropna(subset=["innovation_score", "gii_total_chk"])
r_chk = both["innovation_score"].corr(both["gii_total_chk"])
recon = work.dropna(subset=["gii_input", "gii_output", "gii_total_chk"])
r_recon = ((recon["gii_input"] + recon["gii_output"]) / 2).corr(recon["gii_total_chk"])
print(f"  sanity — 우리 GII ↔ 이 파일 GII 상관: {r_chk:.4f} / (Input+Output)/2 ↔ 종합: {r_recon:.4f}")
print(f"  gii_output 보유: {work['gii_output'].notna().sum()}행 / {work['country'][work['gii_output'].notna()].nunique()}국")

# ── 피처 (§7.7과 동일 — 산출 제외) ──
CULTURE = ["sci_literacy", "pisa_math", "pisa_reading", "sci_trust", "tech_acceptance",
           "tertiary_enroll", "tertiary_attainment"]
RND_INPUT = ["rd_gdp_pct", "researchers_per_m"]
REWARD = ["relative_salary", "rd_budget_per_researcher", "youth_unemp_pct"]
INFRA = ["gov_edu_exp_pct", "internet_users_pct", "mobile_subs_per100", "secure_servers_per_m"]
DEV = ["gdp_per_capita"]
feat_all = CULTURE + RND_INPUT + REWARD + INFRA + DEV

work = work.sort_values(["country", "year"])
for c in feat_all:
    work[c] = work.groupby("country")[c].transform(lambda s: s.interpolate(limit_direction="both"))
    work[c] = work[c].fillna(work[c].median())

def star(pv):
    return "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else "n.s."

scaler = StandardScaler()

def run_block(df, target_col, target_name, do_mediation=True):
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    g = df["country"].values
    y = df[target_col].values
    print("=" * 92)
    print(f"■ 타깃: {target_name}  (n={len(df)}, {df['country'].nunique()}국)")
    print("=" * 92)
    sets = {
        "문화만 (비순환)": CULTURE,
        "R&D투입만": RND_INPUT,
        "문화+투입+보상+인프라": CULTURE + RND_INPUT + REWARD + INFRA,
        "+발전수준(GDP)": feat_all,
    }
    print(f"{'피처셋':28}{'CV R²':>10}{'RMSE':>9}")
    cvr = {}
    for name, cols in sets.items():
        X = scaler.fit_transform(df[cols])
        oof = np.full(len(y), np.nan)
        for tr, va in GroupKFold(5).split(X, y, g):
            m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 random_state=42, verbosity=0)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])
        cvr[name] = r2_score(y, oof)
        print(f"{name:28}{cvr[name]:>10.3f}{np.sqrt(mean_squared_error(y, oof)):>9.3f}")
    print(f"  → 문화 우위(문화만−투입만): {cvr['문화만 (비순환)'] - cvr['R&D투입만']:+.3f}")

    Xcols = ["sci_literacy"] + RND_INPUT + DEV
    Xs = pd.DataFrame(scaler.fit_transform(df[Xcols]), columns=Xcols)
    ys = pd.Series(scaler.fit_transform(df[[target_col]]).ravel())
    mo = sm.OLS(ys, sm.add_constant(Xs)).fit(cov_type="cluster", cov_kwds={"groups": g})
    print(f"\n  Pooled OLS (R&D투입+GDP 통제): sci_literacy β = {mo.params['sci_literacy']:+.3f} "
          f"{star(mo.pvalues['sci_literacy'])} (p={mo.pvalues['sci_literacy']:.4f})")

    if do_mediation:
        def fit(cols, dep):
            Xd = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)
            return sm.OLS(dep, sm.add_constant(Xd)).fit(cov_type="cluster", cov_kwds={"groups": g})
        M = "researchers_per_m"
        ms = pd.Series(scaler.fit_transform(df[[M]]).ravel())
        m1, m2, m3 = fit(["sci_literacy"], ys), fit(["sci_literacy"], ms), fit(["sci_literacy", M], ys)
        a, b = m2.params["sci_literacy"], m3.params[M]
        ind = a * b
        sz = ind / np.sqrt(b**2 * m2.bse["sci_literacy"]**2 + a**2 * m3.bse[M]**2)
        sp = 2 * (1 - stats.norm.cdf(abs(sz)))
        c_tot = m1.params["sci_literacy"]
        print(f"  매개 (소양→연구원→산출): 총 {c_tot:+.3f}{star(m1.pvalues['sci_literacy'])} · a {a:+.3f}"
              f"{star(m2.pvalues['sci_literacy'])} · b {b:+.3f}{star(m3.pvalues[M])} · "
              f"간접 {ind:+.3f} (Sobel z={sz:+.2f} {star(sp)}) · 매개 {100*ind/c_tot:.0f}%")

    Dc = pd.get_dummies(pd.Series(g), prefix="c", drop_first=True).astype(float).reset_index(drop=True)
    Xfe = pd.concat([Xs.reset_index(drop=True), Dc], axis=1)
    mfe = sm.OLS(ys.reset_index(drop=True), sm.add_constant(Xfe)).fit(
        cov_type="cluster", cov_kwds={"groups": g})
    print(f"  국가 FE within: sci_literacy β = {mfe.params['sci_literacy']:+.3f} "
          f"{star(mfe.pvalues['sci_literacy'])} (p={mfe.pvalues['sci_literacy']:.4f})\n")
    return cvr

# ① 정공법: GII Output Sub-Index (표준 지수의 산출 절반 — 피처와 구성 중복 없음)
run_block(work, "gii_output", "GII Innovation OUTPUT Sub-Index (정공법 — 투입·문화→산출)")
# ② 참조: GII Input Sub-Index (rd_gdp 등이 직접 구성요소 = 순환 대조군)
run_block(work, "gii_input", "GII Input Sub-Index (대조군 — rd_gdp·연구원이 구성요소, 순환)", do_mediation=False)
# ③ 참조: 같은 표본 GII 종합(순환 절반 포함)
run_block(work.dropna(subset=["gii_output"]), "innovation_score", "GII 종합 (같은 표본 비교용)", do_mediation=False)
# ④ robustness: manufacturing VA (GII와 r≈0 — 의미 거리 caveat)
run_block(work, "manufacturing_va", "제조업 부가가치 %GDP (robustness — 비순환 외부 outcome)", do_mediation=False)

print("### 3단계 본 분석 완료 (read-only)")
