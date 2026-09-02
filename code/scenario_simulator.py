"""
scenario_simulator.py
=====================
What-if 시나리오 시뮬레이션 모듈

- 단일 지표 변화 시 혁신 성과 변동 예측
- 복합 정책 시나리오 분석
- 국가별 병목 지점 진단
- 예산 배분 최적화 시뮬레이션
"""

from typing import Any

import pandas as pd
from config import GAP_REFERENCE_COUNTRIES, SCENARIO_DEFAULT_COUNTRIES
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# 1. 단일 지표 What-if 시뮬레이션
# ──────────────────────────────────────────────
def single_feature_whatif(model: Any,
                            X: pd.DataFrame,
                            scaler: StandardScaler,
                            feature_name: str,
                            change_pct: float,
                            processed_df: pd.DataFrame,
                            target_country: str = "KOR") -> dict:
    """
    특정 피처를 change_pct% 변화시켰을 때 혁신 성과 예측 변화
    (원본 스케일에서 변화를 계산 후 scaled space로 변환)

    Parameters
    ----------
    change_pct  : 변화율 (예: +15 → 15% 증가, -10 → 10% 감소)

    Notes
    -----
    `scaler` 인자는 시그니처 호환성을 위해 유지하지만 본 함수에서는 사용하지 않는다.
    X는 이미 표준화된 공간(StandardScaler 후) 이므로 sigma_shift 를 직접 더한다.
    """
    # scaler 는 시그니처 호환성 목적으로만 유지 (X 가 이미 표준화 공간)
    del scaler

    if feature_name not in X.columns:
        candidates = [c for c in X.columns if feature_name in c]
        if not candidates:
            return {"error": f"피처 '{feature_name}'를 찾을 수 없습니다."}
        feature_name = candidates[0]
        print(f"   → 유사 피처 사용: {feature_name}")

    # 해당 국가 최근 연도 행
    mask = processed_df.loc[X.index, "country"] == target_country
    if not mask.any():
        X_base = X.mean(axis=0).to_frame().T
        X_counter = X_base.copy()
    else:
        latest_yr_idx = processed_df.loc[X.index[mask], "year"].idxmax()
        X_base    = X.loc[[latest_yr_idx]].copy()
        X_counter = X_base.copy()

    # scaled space에서 sigma 이동
    # 과거 코드는 max(abs(chg)/100, 0.3) 으로 작은 충격을 0.3σ까지 강제로 키워
    # +5%·+10%·+20% 가 사실상 동일 충격이 되었음 → What-if 비교 의미가 깨짐.
    # 정직한 선형 매핑으로 변경하되, 모델 학습 분포 밖으로 멀리 외삽되지 않도록
    # ±1.0σ 로 캡한다.
    sigma_shift = max(-1.0, min(1.0, change_pct / 100.0))
    X_counter[feature_name] = X_counter[feature_name] + sigma_shift

    baseline    = float(model.predict(X_base)[0])
    counterfact = float(model.predict(X_counter)[0])
    delta       = counterfact - baseline

    return {
        "country":        target_country,
        "feature":        feature_name,
        "change_pct":     change_pct,
        "delta_sigma":    round(sigma_shift, 3),
        "baseline":       round(baseline, 4),
        "counterfactual": round(counterfact, 4),
        "delta":          round(delta, 4),
        "delta_pct":      round(delta / (abs(baseline) + 1e-9) * 100, 2),
    }


# ──────────────────────────────────────────────
# 2. 복합 정책 시나리오
# ──────────────────────────────────────────────
# 타입 주석: 값 dict 가 str(description)·dict(changes) 혼합이라 주석 없이는 mypy 가
# 값 타입을 Collection[str] 로 좁혀 .items() 접근을 거부한다(구 baseline 1건).
# replicate_paper.py 의 monkey-patch 대상 — 이름·값·구조 변경 금지(주석만).
POLICY_SCENARIOS: dict[str, dict[str, Any]] = {
    "과학문화": {
        "description": "과학문화 강화 (소양·신뢰·기술수용 +20%)",
        "changes": {"sci_literacy": +20, "sci_trust": +20, "tech_acceptance": +20},
    },
    "물적규모": {
        "description": "R&D 규모 확대 (R&D투자·연구원·특허·논문 +20%)",
        "changes": {
            "rd_gdp_pct": +20,
            "researchers_per_m": +20,
            "patent_apps": +20,
            "journal_articles": +20,
        },
    },
    "운영구조": {
        "description": "교육·인프라 투자 (교육지출·인터넷 +20%)",
        "changes": {"gov_edu_exp_pct": +20, "internet_users_pct": +20},
    },
    "보상수준": {
        "description": "첨단산업 확대 (첨단기술 수출 +20%)",
        "changes": {"hi_tech_export_pct": +20},
    },
}


def run_policy_scenarios(model: Any,
                          X: pd.DataFrame,
                          scaler: StandardScaler,
                          processed_df: pd.DataFrame,
                          countries: list | None = None) -> pd.DataFrame:
    """
    여러 국가 × 여러 시나리오 조합으로 예측 결과 행렬 생성

    Returns
    -------
    DataFrame: country × scenario → delta_pct

    Notes
    -----
    `scaler` 인자는 시그니처 호환성을 위해 유지하지만 본 함수에서는 사용하지 않는다.
    X는 이미 표준화된 공간(StandardScaler 후) 이므로 sigma_shift 를 직접 더한다.
    """
    # scaler 는 시그니처 호환성 목적으로만 유지 (X 가 이미 표준화 공간)
    del scaler

    countries = countries or SCENARIO_DEFAULT_COUNTRIES
    print("🎯 정책 시나리오 시뮬레이션 중...")
    records = []

    for country in countries:
        mask = processed_df.loc[X.index, "country"] == country
        if not mask.any():
            continue
        latest_yr_idx = processed_df.loc[X.index[mask], "year"].idxmax()
        X_base = X.loc[[latest_yr_idx]].copy()
        baseline = float(model.predict(X_base)[0])

        row = {"country": country, "baseline": round(baseline, 4)}

        for scen_id, scen_info in POLICY_SCENARIOS.items():
            X_cf = X_base.copy()
            for feat, chg_pct in scen_info["changes"].items():
                # StandardScaler 공간: change_pct/100 sigma 이동
                # 과거 코드는 max(abs(chg)/100, 0.3) 으로 작은 충격을 0.3σ까지 강제
                # 키워 +5%·+10%·+20% 가 사실상 동일 충격이 되었음. 정직한 선형
                # 매핑으로 변경하되 ±1.0σ 로 캡(학습 분포 밖 외삽 방지).
                sigma_shift = max(-1.0, min(1.0, chg_pct / 100.0))
                targets = [c for c in X.columns if c.startswith(feat)]
                for t in targets:
                    X_cf[t] = X_cf[t] + sigma_shift

            cf_val = float(model.predict(X_cf)[0])
            delta_pct = (cf_val - baseline) / (abs(baseline) + 1e-9) * 100
            row[scen_id] = round(delta_pct, 2)

        records.append(row)
        print(f"   {country}: {row}")

    result = pd.DataFrame(records).set_index("country")
    print("✅ 시나리오 분석 완료")
    return result


# ──────────────────────────────────────────────
# 3. 국가별 병목 진단
# ──────────────────────────────────────────────
def diagnose_bottleneck(shap_country_profile: pd.DataFrame,
                         target_country: str = "KOR") -> pd.DataFrame:
    """
    SHAP Local 분석 결과에서 혁신 성과를 저해하는 상위 병목 피처 진단

    Returns
    -------
    DataFrame: 저해 요인 순위 (음(-)의 SHAP 기여)
    """
    if shap_country_profile.empty:
        return pd.DataFrame()

    bottlenecks = (
        shap_country_profile[shap_country_profile["shap_value"] < 0]
        .sort_values("shap_value")
        .head(10)
        .copy()
    )
    bottlenecks["country"] = target_country
    bottlenecks["severity"] = bottlenecks["shap_value"].abs()
    return bottlenecks


# ──────────────────────────────────────────────
# 4. 한국 집중 진단
# ──────────────────────────────────────────────
def korea_deep_dive(model: Any, X: pd.DataFrame,
                     processed_df: pd.DataFrame,
                     scaler: StandardScaler) -> pd.DataFrame:
    """
    한국(KOR) vs 핀란드(FIN), 미국(USA) 비교 분석
    → 각 지표에서 한국의 갭 및 갭 해소 시 기대 성과 변화 계산
    """
    ref_countries = GAP_REFERENCE_COUNTRIES
    records = []

    mask_kor = processed_df.loc[X.index, "country"] == "KOR"
    if not mask_kor.any():
        return pd.DataFrame()
    kor_idx = processed_df.loc[X.index[mask_kor], "year"].idxmax()
    X_kor   = X.loc[[kor_idx]].copy()
    kor_pred = float(model.predict(X_kor)[0])

    for ref in ref_countries:
        mask_ref = processed_df.loc[X.index, "country"] == ref
        if not mask_ref.any():
            continue
        ref_idx  = processed_df.loc[X.index[mask_ref], "year"].idxmax()
        X_ref    = X.loc[[ref_idx]]
        ref_pred = float(model.predict(X_ref)[0])

        # 피처별 갭
        feat_gap = (X_ref.values[0] - X_kor.values[0])

        # 갭의 50% → scaled space에서 직접 이동
        X_cf = X_kor.copy()
        X_cf.iloc[0] += feat_gap * 0.5
        cf_pred = float(model.predict(X_cf)[0])

        records.append({
            "reference":      ref,
            "kor_score":      round(kor_pred, 4),
            "ref_score":      round(ref_pred, 4),
            "gap":            round(ref_pred - kor_pred, 4),
            "halfgap_bridge": round(cf_pred, 4),
            "expected_gain":  round(cf_pred - kor_pred, 4),
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# 5. 통합 시뮬레이션 파이프라인
# ──────────────────────────────────────────────
def run_simulation(model: Any, X: pd.DataFrame,
                    processed_df: pd.DataFrame,
                    scaler: StandardScaler,
                    shap_results: dict) -> dict:
    print("🔭 시나리오 시뮬레이션 실행 중...")

    # 정책 시나리오 행렬
    scenario_matrix = run_policy_scenarios(model, X, scaler, processed_df)

    # 한국 진단
    kor_diagnosis = diagnose_bottleneck(
        shap_results.get("country_profile", pd.DataFrame()), "KOR"
    )

    # 한국 vs 선진국 갭 분석
    kor_gap = korea_deep_dive(model, X, processed_df, scaler)

    print("\n📊 한국 병목 요인 Top 5:")
    if not kor_diagnosis.empty:
        print(kor_diagnosis[["feature", "shap_value", "category"]].head(5).to_string(index=False))

    print("\n📊 한국 vs 선진국 갭 분석:")
    if not kor_gap.empty:
        print(kor_gap.to_string(index=False))

    return {
        "scenario_matrix": scenario_matrix,
        "kor_bottlenecks": kor_diagnosis,
        "kor_gap_analysis": kor_gap,
    }


# ──────────────────────────────────────────────
# 6. 결과 직렬화 (xai_results.json 용)
# ──────────────────────────────────────────────
def serialize_sim_results(sim_results: dict) -> dict:
    """sim_results → JSON 직렬화 가능 dict.

    반환 키: scenario_matrix · korea_gap_analysis
    (xai_results.json 스키마 — 키 이름·순서는 소비처 계약).
    """
    return {
        "scenario_matrix": (
            sim_results["scenario_matrix"].round(2).to_dict()
            if not sim_results["scenario_matrix"].empty else {}
        ),
        "korea_gap_analysis": (
            sim_results["kor_gap_analysis"].round(4).to_dict(orient="records")
            if not sim_results["kor_gap_analysis"].empty else []
        ),
    }
