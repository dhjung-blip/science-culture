#!/usr/bin/env python
"""
build_data.py — 파이프라인 산출물(xai_results.json + processed_dataset.csv)을
프론트엔드 src/data.js (DATASETS) 로 변환한다.

사용:
  # v2 만 (backward compat)
  python scripts/build_data.py <V2_DIR>

  # v1 + v2 둘 다 (대시보드에서 토글 가능)
  python scripts/build_data.py <V2_DIR> <V1_DIR>

  기본 V2_DIR=/tmp/sc_front, V1_DIR=/tmp/sc_paper_v1
  V2_DIR 은 `main.py --country-set pisa-gii --output <V2_DIR>` 산출물
  V1_DIR 은 `replicate_paper.py --output <V1_DIR>` 산출물 (초기 사양)

mock 을 실데이터로 교체. 재실행하면 갱신된다(재현성).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# ── 매핑 테이블 (모드 공용) ──────────────────────────────────
CAT_ID = {"과학문화": "culture", "물적규모": "physical", "운영구조": "ops", "보상수준": "reward"}
CAT_VAR = {"culture": "var(--c-culture)", "physical": "var(--c-physical)",
           "ops": "var(--c-ops)", "reward": "var(--c-reward)"}
CAT_DESC = {"culture": "PISA 과학·수학·읽기 + WVS 신뢰·기술수용 + 고등교육",
            "physical": "R&D 투자·연구원·특허·논문·1인당 GDP",
            "ops": "교육지출·인터넷 인프라",
            "reward": "전문직 임금비(ILOSTAT) + 연구원당 R&D 자원 + 첨단수출"}
CAT_DESC_V1 = {"culture": "PISA 과학소양·자아효능감·과학신뢰·기술수용 (합성)",
               "physical": "R&D 투자·연구원·특허·논문·1인당 GDP",
               "ops": "과제기간·행정부담·은퇴연령·교육지출·인터넷 (합성 3종 포함)",
               "reward": "상대임금·연구원당 R&D 예산·첨단수출 (합성 2종 포함)"}
FEAT_KO = {
    "sci_literacy": "과학소양(PISA)", "sci_trust": "과학기술 신뢰", "tech_acceptance": "기술 수용도",
    "self_efficacy": "과학 자아효능감", "pisa_math": "PISA 수학", "pisa_reading": "PISA 읽기",
    "tertiary_enroll": "고등교육 진학률",
    "tertiary_attainment": "학사 보유율",      # 2026-06 추가 (과학문화)
    "rd_gdp_pct": "GDP 대비 R&D", "researchers_per_m": "백만명당 연구원", "patent_apps": "특허 출원",
    "journal_articles": "과학 논문", "gdp_per_capita": "1인당 GDP",
    "gov_edu_exp_pct": "교육 지출/GDP", "internet_users_pct": "인터넷 보급",
    "mobile_subs_per100": "모바일 보급",        # 2026-06 추가 (운영구조)
    "secure_servers_per_m": "보안 인터넷 서버", # 2026-06 추가 (운영구조)
    "avg_proj_duration": "평균 과제기간", "admin_burden_ratio": "행정 부담",
    "researcher_retire_age": "연구자 은퇴연령",
    "relative_salary": "상대 연봉", "rd_budget_per_researcher": "연구원당 R&D 예산",
    "youth_unemp_pct": "청년 실업률",            # 2026-06 추가 (보상수준 · 비순환)
    "hi_tech_export_pct": "첨단기술 수출", "idx_sci_culture": "과학문화 지수",
    "idx_hard_rd": "물적규모 지수", "idx_structure": "운영구조 지수", "idx_economic": "보상수준 지수",
}
CTRY = {"KOR": ("한국", "🇰🇷", "kr"), "JPN": ("일본", "🇯🇵", "jp"), "USA": ("미국", "🇺🇸", "us"),
        "DEU": ("독일", "🇩🇪", "de"), "FIN": ("핀란드", "🇫🇮", "fi"), "SWE": ("스웨덴", "🇸🇪", "se"),
        "GBR": ("영국", "🇬🇧", "gb"), "CHN": ("중국", "🇨🇳", "cn")}


def feat_ko(f: str) -> str:
    base = f
    suffix = ""
    for lag in ("_lag3", "_lag5", "_lag7"):
        if f.endswith(lag):
            base = f[: -len(lag)]
            suffix = f" ({lag[4:]}년 시차)"
            break
    return FEAT_KO.get(base, base) + suffix


def build_dataset(run_dir: str, mode: str) -> dict:
    """run_dir 의 파이프라인 산출물을 대시보드 데이터셋으로 변환.
    mode: 'v1' (초기 사양) | 'v2' (확장 분석)"""
    res = json.load(open(os.path.join(run_dir, "xai_results.json"), encoding="utf-8"))
    df = pd.read_csv(os.path.join(run_dir, "processed_dataset.csv")).sort_values(["country", "year"])
    # 정본 단일 JSON (v2 20시드 SHAP 우선 — 단일실행 출렁임 방지, results_main.json)
    _canon_path = os.path.join(run_dir, "results_main.json")
    canon = json.load(open(_canon_path, encoding="utf-8")) if os.path.exists(_canon_path) else None
    is_v2_canon = (mode != "v1") and canon is not None

    mp = res["model_performance"]
    if is_v2_canon:
        cc = {k: v[0] for k, v in canon["shap_raw_20seed_pct"].items() if not k.startswith("_")}
    else:
        cc = {c["category"]: c["weight_pct"] for c in res["category_contribution"]}

    n_ctry = df["country"].nunique()
    n_yr = int(df["year"].max() - df["year"].min() + 1)
    phys_pct = cc.get("물적규모", 0)
    is_v1 = mode == "v1"
    desc_map = CAT_DESC_V1 if is_v1 else CAT_DESC

    # ── KPI ────────────────────────────────────────────────
    if is_v1:
        kpis = [
            {"id": "culture", "label": "과학문화 기여도", "value": f"{cc.get('과학문화', 0):.1f}",
             "unit": "%", "note": f"초기 사양 · 물적규모 {phys_pct:.0f}%와 함께 상위",
             "tone": "culture", "big": True},
            {"id": "r2", "label": "교차검증 R²", "value": f"{mp['r2']:.2f}",
             "unit": "", "note": f"GroupKFold · {n_ctry}국 (논문 0.509)", "tone": "neutral"},
            {"id": "rmse", "label": "RMSE", "value": f"{mp['rmse']:.2f}",
             "unit": "pt", "note": "GII 오차 (논문 2.29)", "tone": "neutral"},
            {"id": "scope", "label": "분석 범위", "value": f"{n_ctry}",
             "unit": f"개국×{n_yr}년", "note": f"{len(df)} 패널 (논문 사양)", "tone": "neutral"},
            {"id": "feat", "label": "데이터", "value": "합성",
             "unit": "", "note": "PISA·WVS·ILOSTAT 합성(논문 P111)", "tone": "neutral"},
        ]
    else:
        kpis = [
            {"id": "culture", "label": "과학문화 기여도", "value": f"{cc.get('과학문화', 0):.1f}",
             "unit": "%", "note": f"SHAP 기여 · 물적규모 {phys_pct:.0f}%와 함께 상위(raw)",
             "tone": "culture", "big": True},
            {"id": "r2", "label": "교차검증 R²", "value": f"{mp['r2']:.2f}",
             "unit": "", "note": f"GroupKFold(국가) · {n_ctry}국 일반화", "tone": "neutral"},
            {"id": "rmse", "label": "RMSE", "value": f"{mp['rmse']:.2f}",
             "unit": "pt", "note": "GII 오차 · 선진국 미세차(2~5점)는 오차 내",
             "tone": "neutral"},
            {"id": "scope", "label": "분석 범위", "value": f"{n_ctry}",
             "unit": f"개국×{n_yr}년", "note": f"{len(df)} 패널 표본 (PISA∩GII)", "tone": "neutral"},
            {"id": "feat", "label": "실 데이터 출처", "value": "5",
             "unit": "종", "note": "WB · PISA · WVS · GII · ILOSTAT (합성 4피처 모델 제외)",
             "tone": "neutral"},
        ]

    # ── 카테고리 기여 ──────────────────────────────────────
    categories = []
    if is_v2_canon:
        for k, v in canon["shap_raw_20seed_pct"].items():
            if k.startswith("_"):
                continue
            cid = CAT_ID[k]
            categories.append({"id": cid, "name": k, "value": v[0],
                               "color": CAT_VAR[cid], "desc": desc_map[cid]})
        categories.sort(key=lambda c: -c["value"])
    else:
        for c in res["category_contribution"]:
            cid = CAT_ID[c["category"]]
            categories.append({"id": cid, "name": c["category"], "value": round(c["weight_pct"], 1),
                               "color": CAT_VAR[cid], "desc": desc_map[cid]})

    # ── 통제 분석 카테고리 (v2 만; v1 은 합성 데이터라 의미 약함) ─
    categories_ctrl = []
    if is_v2_canon:
        for k, v in canon["shap_controlled_20seed_pct"].items():
            if k.startswith("_"):
                continue
            categories_ctrl.append({"id": CAT_ID[k], "name": k, "value": v[0],
                                    "color": CAT_VAR[CAT_ID[k]]})
        categories_ctrl.sort(key=lambda c: -c["value"])
        print(f"   [{mode}] 통제 카테고리(정본 20시드): {[(c['name'], c['value']) for c in categories_ctrl]}")
    elif not is_v1:
        try:
            import xgboost as _xgb
            import shap as _shap
            _CF = {"sci_literacy": "과학문화", "sci_trust": "과학문화", "tech_acceptance": "과학문화",
                   "pisa_math": "과학문화", "pisa_reading": "과학문화", "tertiary_enroll": "과학문화",
                   "tertiary_attainment": "과학문화",  # 2026-06 추가
                   "idx_sci_culture": "과학문화", "sci_literacy_lag3": "과학문화",
                   "sci_trust_lag3": "과학문화", "tech_acceptance_lag3": "과학문화",
                   "rd_gdp_pct": "물적규모", "researchers_per_m": "물적규모",
                   "gov_edu_exp_pct": "운영구조", "internet_users_pct": "운영구조",
                   "mobile_subs_per100": "운영구조",     # 2026-06 추가
                   "secure_servers_per_m": "운영구조",   # 2026-06 추가
                   "idx_structure": "운영구조",
                   "relative_salary": "보상수준", "rd_budget_per_researcher": "보상수준",
                   "youth_unemp_pct": "보상수준"}      # 2026-06 추가 (비순환)
            _cf = [c for c in _CF if c in df.columns]
            _d = df.dropna(subset=["innovation_score"])
            _m = _xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   random_state=42, verbosity=0)
            _m.fit(_d[_cf], _d["innovation_score"])
            _imp = np.abs(_shap.TreeExplainer(_m).shap_values(_d[_cf])).mean(0)
            _share = pd.DataFrame({"cat": [_CF[f] for f in _cf],
                                   "p": 100 * _imp / _imp.sum()}).groupby("cat")["p"].sum()
            categories_ctrl = [{"id": CAT_ID[k], "name": k, "value": round(float(v), 1),
                                "color": CAT_VAR[CAT_ID[k]]}
                               for k, v in _share.sort_values(ascending=False).items()]
            print(f"   [{mode}] 통제 카테고리: {[(c['name'], c['value']) for c in categories_ctrl]}")
        except Exception as e:
            print(f"   ⚠ [{mode}] 통제 카테고리 스킵: {e}")

    # ── 글로벌 피처 중요도 ────────────────────────────────
    features = [{"name": feat_ko(f["feature"]), "value": round(f["importance_pct"] / 100, 3),
                 "cat": CAT_ID[f["category"]]} for f in res["top10_features"]]

    # ── 한국 병목 (tornado) ───────────────────────────────
    kb = sorted(res["korea_bottlenecks"], key=lambda x: abs(x["shap_value"]), reverse=True)[:9]
    tornado = [{"name": feat_ko(b["feature"]), "value": round(b["shap_value"], 2),
                "cat": CAT_ID[b["category"]]} for b in kb]

    # ── What-if 국가 & 시나리오 ────────────────────────────
    base = res["scenario_matrix"]["baseline"]
    countries = []
    for code, b in base.items():
        if code in CTRY:
            name, flag, cid = CTRY[code]
            countries.append({"id": cid, "name": name, "flag": flag, "base": round(b, 1)})

    POL_LABEL_V2 = {"culture": "과학문화 확산", "physical": "R&D 투자 규모",
                    "ops": "교육·인프라 투자", "reward": "첨단산업 보상"}
    if is_v1:
        # v1 은 시나리오가 S1~S5 5개. cat 매핑: S1→culture, S2/S5/S4→reward/physical/종합
        v1_scen_keys = [k for k in res["scenario_matrix"].keys() if k != "baseline"]
        V1_POL = {"S1_과학문화_강화": ("culture", "S1 과학문화 강화"),
                  "S2_보상_개선": ("reward", "S2 보상 개선"),
                  "S3_과제기간_연장": ("ops", "S3 과제기간 연장"),
                  "S4_종합_혁신패키지": ("culture", "S4 종합 혁신패키지"),
                  "S5_물적규모_확대": ("physical", "S5 물적규모 확대")}
        policies = [{"id": f"p{i+1}",
                     "label": V1_POL.get(k, (None, k))[1],
                     "cat": V1_POL.get(k, ("culture", k))[0],
                     "weight": 0.2,
                     "scenario_key": k}
                    for i, k in enumerate(v1_scen_keys) if k in V1_POL]
        sm = res["scenario_matrix"]
        whatif = {}
        for code in base:
            if code in CTRY:
                whatif[CTRY[code][2]] = {V1_POL[k][0] + "_" + k: round(sm[k][code], 2)
                                         for k in v1_scen_keys if k in V1_POL and code in sm[k]}
                # also keyed by scenario_key for direct lookup
                for k in v1_scen_keys:
                    if k in sm and code in sm[k]:
                        whatif[CTRY[code][2]][k] = round(sm[k][code], 2)
    else:
        policies = [{"id": f"p{i+1}", "label": POL_LABEL_V2[CAT_ID[c['category']]],
                     "cat": CAT_ID[c["category"]], "weight": round(c["weight_pct"] / 100, 3)}
                    for i, c in enumerate(res["category_contribution"])]
        sm = res.get("scenario_matrix", {})
        whatif = {}
        for code in base:
            if code in CTRY:
                whatif[CTRY[code][2]] = {CAT_ID[cat]: round(sm[cat][code], 2)
                                         for cat in CAT_ID if cat in sm and code in sm[cat]}

    # ── Time-lag 상관 ─────────────────────────────────────
    def lag_corr(col: str) -> list:
        if col not in df.columns:
            return [0.0] * 8
        out, inn = [], df["innovation_score"]
        for L in range(8):
            s = df.groupby("country")[col].shift(L)
            m = s.notna() & inn.notna()
            out.append(round(float(np.corrcoef(s[m], inn[m])[0, 1]), 2) if m.sum() > 20 else 0.0)
        return out

    timelag = {"lags": list(range(8)),
               "culture": lag_corr("sci_literacy"),
               "physical": lag_corr("idx_hard_rd")}

    # ── 과학소양 지수 (PISA 실측연도 기준; v1 합성도 같은 방식) ─
    PISA_REF_YEAR = 2018
    latest = df[df["year"] == PISA_REF_YEAR]
    if latest.empty:
        latest = df[df["year"] == df["year"].max()]
    if "sci_literacy" in latest.columns and len(latest) > 0:
        _smin, _smax = float(latest["sci_literacy"].min()), float(latest["sci_literacy"].max())

        def _cidx(v: float) -> int:
            return round((v - _smin) / (_smax - _smin) * 100) if _smax > _smin else 50
        ci_rows = []
        for code in ["FIN", "SWE", "USA", "DEU", "JPN", "KOR"]:
            r = latest[latest["country"] == code]
            if not r.empty and code in CTRY:
                ci_rows.append({"id": CTRY[code][2], "name": CTRY[code][0],
                                "value": _cidx(float(r["sci_literacy"].iloc[0])),
                                "korea": code == "KOR"})
        ci_rows.sort(key=lambda x: x["value"], reverse=True)
    else:
        ci_rows = []

    # ── 3층위 종합표 (v2 만; v1 은 비공개 회귀 추론) ───────
    methods = []
    methods_takeaway = ""
    if not is_v1:
        methods = [
            {"label": "XGBoost 한계기여", "value": "+0.086", "verdict": "예측 증분 중간",
             "ns": False, "detail": "R&D 통제 후 과학소양 단독 추가 효과"},
            {"label": "Pooled OLS β (R&D 통제)", "value": "+3.84***", "verdict": "횡단 연관 매우 견고",
             "ns": False, "detail": "원단위 GII점/1SD · 완전표준화 +0.394 · 국가 클러스터 SE · VIF 2.08"},
            {"label": "패널 FE β (within)", "value": "+1.30 n.s.",
             "verdict": "시간 내 효과 미검출", "ns": True,
             "detail": "국가 baseline 제거 시 사라짐 — PISA 느린 변동이 검정력 제한"},
        ]
        methods_takeaway = ("연관은 실재하나 나라 간(횡단)이고, 인과적 within 효과는 이 데이터로 "
                            "단정 못한다. 'X를 올리면 혁신이 오른다'는 인과 주장은 자제.")

    # ── 매개분석 (§7.5 Baron-Kenny + Sobel · v2 만) ────────
    mediation: dict = {}
    if not is_v1:
        mediation = {
            "x": "과학소양 (PISA)",
            "y": "혁신 (GII)",
            "pathways": [
                {"m": "researchers_per_m", "m_label": "연구원 밀도",
                 "a": 0.750, "b": 5.151,
                 "c_prime": 3.532, "indirect": 3.863,
                 "sobel_p": "<0.001", "mediated_pct": 52.2,
                 "highlight": True},
                {"m": "idx_hard_rd", "m_label": "물적규모 종합",
                 "a": 0.634, "b": 5.138,
                 "c_prime": 4.137, "indirect": 3.258,
                 "sobel_p": "<0.001", "mediated_pct": 44.1,
                 "highlight": False},
                {"m": "rd_gdp_pct", "m_label": "R&D 강도",
                 "a": 0.525, "b": 4.251,
                 "c_prime": 5.162, "indirect": 2.233,
                 "sobel_p": "<0.001", "mediated_pct": 30.2,
                 "highlight": False},
            ],
            "total_effect": 7.395,
            "note": ("Baron-Kenny 4단계 + Sobel test, 44국 패널, "
                     "국가 클러스터 강건 SE. ★ 횡단 매개 — "
                     "시간 선행성 미보장(Imai-Tingley 후속 과제). "
                     "헤드라인 매개비율은 부트스트랩 49.1%[32.2,67.2](점추정 52.2%)."),
        }

    # ── 탈순환 검증 (§7.7·§7.8 · v2 만; v1 은 초기 사양 범위 밖 → None) ──
    # 수치 출처: 원고 §4.2 · analysis/decircle_experiment.py(탈순환 실험 — 직접측정, within 은 정본 KNN 재검증치
    # +0.349***/−0.037) + §7.8(GII Output Sub-Index 본 분석). mediation 패턴과 동일하게
    # 분석 상수 하드코딩(파이프라인 산출물이 아닌 별도 분석 결과).
    decircle = None
    if not is_v1:
        decircle = {
            "intro": ("타깃이 GII(합성지수)일 때의 순환성('GII를 제 재료로 재구성') 반론을 "
                      "검증하기 위해, GII 종합을 우회하는 세 가지 타깃(직접측정 2종 + 산출 "
                      "서브지수)으로 동일 분석을 반복했다."),
            "rows": [
                {"target": "특허/백만명 (log)", "kind": "직접측정",
                 "beta": "+0.410***", "within": "+0.349*** (p=0.0006)", "withinSig": True,
                 "mediation": "−13.9% n.s.", "cvCulture": 0.402, "cvRnd": 0.302},
                {"target": "논문/백만명 (log)", "kind": "직접측정",
                 "beta": "+0.564***", "within": "−0.037 n.s.", "withinSig": False,
                 "mediation": "+21.0%*", "cvCulture": 0.656, "cvRnd": 0.513},
                {"target": "GII Output Sub-Index", "kind": "표준 지수(산출 절반)",
                 "beta": "+0.325***", "within": "+0.385 n.s. (p=0.12)", "withinSig": False,
                 "mediation": "—", "cvCulture": None, "cvRnd": None},
            ],
            "robustness": ("제조업 부가가치(%GDP, 혁신과 무관한 외부 지표)와는 전 분석 "
                           "무관(n.s.) — 신호가 혁신 산출에 특이적(분별 타당도)."),
            "conclusion": ("횡단 연관(β)은 세 구도 모두 생존 — '연관은 GII 순환 덕'이라는 "
                           "대안 설명 기각. within(시간 내)은 특허 직접측정에서만 유의 — "
                           "합성지수의 sticky함이 within 검출을 막는다는 해석과 부합."),
            "caveats": ("β는 표준화 계수(R&D 투입+GDP 통제, 국가 클러스터 SE). "
                        "문화 예측 우위(CV)는 직접측정 타깃에서만 성립. "
                        "인과 입증이 아니라 대안 설명 1개의 제거. "
                        "GII Output은 Mendeley 정리본(CC BY 4.0, ISO3 결함을 국가명 기준 복구) "
                        "2013–2022. "
                        "매개비율은 동결 confounder-only 부트스트랩(국가 클러스터 2000회) 기준 — 논문 +21.0%[5.9,44.4] 유의, 특허 −13.9%[−61.6,36.0] 0 포함·불안정, GII Output 은 동결 사양에서 미보고(—)."),
        }

    # ── Caveats ──────────────────────────────────────────
    if is_v1:
        caveats = [
            {"title": "초기 사양 재현 모드",
             "body": "본 화면은 초기 사양(합성 데이터)으로 재현한 결과. "
                     "20개국·12년·240행, PISA/WVS/ILOSTAT 합성 시뮬레이션, 17변수·5시나리오."},
            {"title": "합성 데이터 한계",
             "body": "PISA·WVS 실데이터가 아닌 국가 프로파일 기반 합성. 실데이터 통합 시 결과 변동 "
                     "가능 — v2 모드에서 실데이터 확장 결과 확인 가능."},
            {"title": "합성 데이터 — 실측 기반 보정 (2026-06)",
             "body": "COUNTRY_PROFILES 4-tuple 을 PISA 2018·WB R&D/GDP·ILOSTAT 임금비·GII 실측 "
                     "기반으로 재산정해 합성 절대수치 왜곡을 해소(예: 한국 PISA 374→519 근방, "
                     "중국 PISA 350→590 근방). 함수 형태·시나리오 정의 등 논문 로직은 그대로 보존. "
                     "본 실행 CV R²·카테고리%는 논문 0.509·24.2%와 약간 다를 수 있다"
                     "(데이터 보정의 결과)."},
            {"title": "표본 규모 (논문 §5.4)",
             "body": "20개국 × 12년 = 240 관측치는 비교적 소규모 패널. CV R² ≈ 0.51은 국가 간 "
                     "혁신 격차의 약 절반을 설명. 모형 일반화 가능성에 제약."},
            {"title": "GII 순환성",
             "body": "타깃 GII는 R&D지출·특허·논문 등 ~80지표의 합성 인덱스. 우리 물적규모 피처와 "
                     "구성요소 중복 → 예측 R²의 상당부분이 동어반복."},
            {"title": "근사 재현",
             "body": "본 실행 CV R² ≈ 0.49·RMSE ≈ 2.7 (2026-06 보정 후). 논문 0.509·2.29와 "
                     "근접하나 카테고리 % / 시나리오 부호는 합성 입력(국가 프로파일) 실측 보정의 "
                     "결과로 변동. 정량 비교 자제, 방법론 시연으로만 읽을 것."},
        ]
    else:
        caveats = [
            {"title": "GII 순환성 (2026-06 검증)",
             "body": "타깃 GII는 R&D지출·특허·논문 등 ~80지표의 합성 인덱스. 우리 물적규모 피처와 "
                     "일부 구성요소 중복 → 예측 R²의 상당부분이 동어반복. "
                     "→ 탈순환 3구도(특허·논문·GII Output) 재검증에서 횡단 연관 전부 "
                     "생존(§7.7·§7.8) — 순환이 연관의 *원천*이라는 대안 설명은 기각. "
                     "단 예측 R²의 절대값은 여전히 순환·발전수준으로 부풀려질 수 있음."},
            {"title": "PISA 3년 주기·보간",
             "body": "PISA 과학·수학·읽기는 2012/15/18 앵커만 실측, 사이 연도는 국가별 선형 보간. "
                     "느린 변동이 within 검정력을 제한."},
            {"title": "표본 한계",
             "body": f"{n_ctry}국·{len(df)} 패널(2012–2023). GroupKFold(국가) CV는 "
                     "'처음 보는 국가' 일반화를 측정. 절대 오차 RMSE~5 GII점은 "
                     "선진국 미세차(2~5점)를 가르기에 부족."},
            {"title": "통제 결과의 실행 민감성",
             "body": "발전수준·순환 통제 시 raw의 62 vs 28 격차가 ~42–49 vs ~42–49로 비등해지나, "
                     "순서는 실행·표본에 민감(다른 실행선 뒤집힘). '격차 축소'로 읽되 '역전 단정'은 금지."},
            {"title": "분류체계 한계",
             "body": "4대 카테고리(과학문화·물적규모·운영구조·보상수준)는 본 연구의 분석 프레임으로, "
                     "범주 내 피처 수 비대칭과 일부 이질성이 있다."},
            {"title": "SHAP ≠ 정책 진단",
             "body": "음의 SHAP은 '모델이 그 피처값으로 기대한 점수보다 실제가 낮음'을 뜻하지 "
                     "현실의 부족이 아니다. 예: 한국 인터넷 보급 SHAP −0.44지만 실측은 97.4%로 "
                     "44국 중 5위(세계 최상위권). 단일 국가 local SHAP을 정책 단정으로 읽으면 오해 — "
                     "category·횡단 비교와 실측값을 함께 봐야 한다."},
            {"title": "데이터 출처(실데이터화)",
             "body": "World Bank(9지표) · WVS(W6/W7) · PISA(과학·수학·읽기) · GII(타깃) · "
                     "ILOSTAT(전문직 임금비) · WB 파생(연구원당 R&D 자원). 합성 4피처는 모델 제외."},
            {"title": "두 사양의 관계",
             "body": "v2 는 실데이터 5종을 통합하고 국가를 44개국으로 확장한 본 분석이다. "
                     "v1 은 합성 데이터·20개국의 초기 사양으로, 입력·피처·표본이 달라 두 모드의 "
                     "수치는 직접 비교하지 않는다(상단 토글로 전환)."},
            {"title": "매개분석 — 횡단·시간 선행성 미보장",
             "body": "§7.5 매개분석은 Baron-Kenny 4단계+Sobel 횡단 추정으로, "
                     "'과학소양 → 연구원밀도 → 혁신' 경로의 *연관 매개*를 보인다. "
                     "시간 선행성·반사실적 인과는 미보장(Imai-Tingley 시차 매개 후속 과제). "
                     "'경로(pathway)'로 읽되 'X→Y 인과'로 단정 금지."},
            {"title": "피처 보강 2026-06 (4범주 고정)",
             "body": "운영구조 빈약(2→4) 해소 + 보상수준 비순환 청년실업 + 과학문화 학사보유율. "
                     "⚠ tertiary_attainment는 GII Human Capital 일부 가중 — 부분 순환."},
        ]

    # ── v2 culture KPI note 강화: 통제 결과(c̄)를 부제 + KPI에 동적 삽입 ──
    if not is_v1 and categories_ctrl:
        _ctrl_cult = next((c["value"] for c in categories_ctrl if c["id"] == "culture"), None)
        _ctrl_phys = next((c["value"] for c in categories_ctrl if c["id"] == "physical"), None)
        if _ctrl_cult is not None:
            _raw_cult = cc.get("과학문화", 0)
            for _k in kpis:
                if _k["id"] == "culture":
                    _k["note"] = (f"raw {_raw_cult:.1f}% · "
                                  f"통제 후 {_ctrl_cult:.0f}%(발전수준·순환 제거)")
                    break

    label = ("v1 — 초기 사양 (20국·합성·17피처·5시나리오)"
             if is_v1 else "v2 — 확장 분석 (44국·실데이터·확장 피처)")
    if is_v1:
        subtitle = "XGBoost + SHAP — 초기 사양 (PISA·WVS 합성)"
    else:
        subtitle = ("XGBoost + SHAP — 과학소양(PISA)이 국가혁신과 횡단 연관 "
                    "(R&D 통제 후 견고; 특허 직접측정에서는 within +0.349*** 검출 — §탈순환)")
        if categories_ctrl:
            _cc = next((c["value"] for c in categories_ctrl if c["id"] == "culture"), None)
            _cp = next((c["value"] for c in categories_ctrl if c["id"] == "physical"), None)
            if _cc is not None and _cp is not None:
                subtitle += (f" · 발전수준 통제 시 비등 "
                             f"(과학문화 ~{_cc:.0f}% vs 물적 ~{_cp:.0f}%)")

    return {
        "kpis": kpis, "categories": categories, "features": features,
        "catColor": {k: CAT_VAR[k] for k in CAT_VAR},
        "catLabel": {"reward": "보상", "physical": "물적", "culture": "과학문화", "ops": "운영"},
        "tornado": tornado, "countries": countries, "policies": policies,
        "timelag": timelag, "cultureIndex": ci_rows, "whatif": whatif,
        "categoriesControlled": categories_ctrl,
        "methods": methods, "methodsTakeaway": methods_takeaway,
        "mediation": mediation,
        "decircle": decircle,
        "caveats": caveats,
        "pisaRefYear": PISA_REF_YEAR,
        "label": label, "subtitle": subtitle, "mode": mode,
        "modePerf": f"CV R²={mp['r2']:.3f} · RMSE={mp['rmse']:.2f} · N={len(df)}",
    }


def main():
    args = sys.argv[1:]
    v2_dir = args[0] if len(args) >= 1 else "/tmp/sc_front"
    v1_dir = args[1] if len(args) >= 2 else "/tmp/sc_paper_v1"
    out = args[2] if len(args) >= 3 else os.path.join(
        os.path.dirname(__file__), "..", "src", "data.js")

    datasets = {}
    if os.path.isdir(v2_dir) and os.path.exists(os.path.join(v2_dir, "xai_results.json")):
        datasets["v2"] = build_dataset(v2_dir, "v2")
        print(f"   ✅ v2 빌드: {v2_dir} → {datasets['v2']['modePerf']}")
    else:
        print(f"   ⚠ v2 산출물 없음: {v2_dir}")

    if os.path.isdir(v1_dir) and os.path.exists(os.path.join(v1_dir, "xai_results.json")):
        datasets["v1"] = build_dataset(v1_dir, "v1")
        print(f"   ✅ v1 빌드: {v1_dir} → {datasets['v1']['modePerf']}")
    else:
        print(f"   ⚠ v1 산출물 없음(v2만 빌드): {v1_dir}")

    if not datasets:
        print("⚠ 빌드할 산출물이 없음 — exit")
        sys.exit(1)

    default_mode = "v2" if "v2" in datasets else "v1"
    header = (
        "/* ============================================================\n"
        "   DATA — 실제 파이프라인 산출물 기반. v1/v2 두 모드 지원.\n"
        "   생성: scripts/build_data.py  (mock 아님)\n"
        f"   기본 모드: {default_mode}\n"
        f"   포함 모드: {sorted(datasets.keys())}\n"
        "   재생성:\n"
        "     python main.py --country-set pisa-gii --output /tmp/sc_front      # v2\n"
        "     python replicate_paper.py --output /tmp/sc_paper_v1               # v1\n"
        "     python scripts/build_data.py /tmp/sc_front /tmp/sc_paper_v1       # 둘 다\n"
        "   ============================================================ */\n"
    )
    js = (header
          + "export const DATASETS = "
          + json.dumps(datasets, ensure_ascii=False, indent=2) + ";\n"
          + f"export const DEFAULT_MODE = {json.dumps(default_mode)};\n"
          + f"export const DATA = DATASETS[{json.dumps(default_mode)}];\n")
    with open(out, "w", encoding="utf-8") as f:
        f.write(js)
    print(f"✅ {out} ({len(js)} bytes) · 기본 {default_mode} · 포함 {sorted(datasets.keys())}")


if __name__ == "__main__":
    main()
