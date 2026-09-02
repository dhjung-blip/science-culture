"""
shap_analyzer.py
================
SHAP (SHapley Additive exPlanations) 분석 모듈

- 전체 피처 중요도 (Global SHAP)
- 4대 카테고리별 순수 기여 가중치
- 국가별 개별 SHAP 분석 (Local Explainability)
- SHAP Dependence Plot 데이터 추출
"""

import warnings

import numpy as np
import pandas as pd
import shap
from config import CATEGORY_LABELS_KR, CATEGORY_MAP

# CATEGORY_MAP(피처→4범주)·CATEGORY_LABELS_KR 정의는 config.py 로 이동했다.
# replicate_paper.py 가 shap_analyzer.CATEGORY_MAP 을 monkey-patch 하므로
# 반드시 이름 바인딩(`from config import X`)으로 가져온다(함수 본문 `config.X` 참조 금지).

def _get_category(col_name: str) -> str:
    """lag 피처 포함 카테고리 매핑"""
    base = col_name.split("_lag")[0]
    return CATEGORY_MAP.get(base, CATEGORY_MAP.get(col_name, "기타"))


# ──────────────────────────────────────────────
# 1. SHAP 값 계산
# ──────────────────────────────────────────────
def compute_shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, shap.Explainer]:
    """
    TreeExplainer를 사용한 빠른 SHAP 계산

    Returns
    -------
    shap_values : ndarray (n_samples, n_features)
    explainer   : shap.TreeExplainer 객체
    """
    print("📐 SHAP 값 계산 중...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(f"   ✅ SHAP 계산 완료: {shap_values.shape}")
    return shap_values, explainer


# ──────────────────────────────────────────────
# 2. 글로벌 피처 중요도
# ──────────────────────────────────────────────
def global_feature_importance(shap_values: np.ndarray,
                                feature_names: list) -> pd.DataFrame:
    """|SHAP| 평균 기반 글로벌 피처 중요도 DataFrame 반환"""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": mean_abs,
        "category":   [_get_category(f) for f in feature_names],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    df["rank"] = df.index + 1
    df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
    return df


# ──────────────────────────────────────────────
# 3. 카테고리별 기여 집계
# ──────────────────────────────────────────────
def category_contribution(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    4대 카테고리별 순수 SHAP 기여 가중치 집계
    → 정책 우선순위 도출에 활용
    """
    cat_df = (
        importance_df.groupby("category")["importance"]
        .sum()
        .reset_index()
        .sort_values("importance", ascending=False)
    )
    cat_df["weight_pct"] = cat_df["importance"] / cat_df["importance"].sum() * 100

    cat_df["category_label"] = cat_df["category"].map(CATEGORY_LABELS_KR)
    return cat_df


# ──────────────────────────────────────────────
# 4. 국가별 Local SHAP 분석
# ──────────────────────────────────────────────
def country_shap_profile(shap_values: np.ndarray,
                          X: pd.DataFrame,
                          processed_df: pd.DataFrame,
                          target_country: str = "KOR") -> pd.DataFrame:
    """
    특정 국가의 최근 연도 SHAP waterfall 데이터 추출
    → 해당 국가의 강점/약점 병목 진단
    """
    if "country" not in processed_df.columns:
        return pd.DataFrame()

    # 해당 국가 최근 행 인덱스
    mask = processed_df.loc[X.index, "country"] == target_country
    if not mask.any():
        return pd.DataFrame()

    # 가장 최근 연도
    latest_idx = processed_df.loc[X.index[mask], "year"].idxmax()
    pos = X.index.get_loc(latest_idx)

    shap_row = shap_values[pos]
    feat_vals = X.iloc[pos]

    profile = pd.DataFrame({
        "feature":    X.columns.tolist(),
        "shap_value": shap_row,
        "feat_value": feat_vals.values,
        "category":   [_get_category(f) for f in X.columns],
    }).sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)

    profile["direction"] = profile["shap_value"].apply(
        lambda v: "▲ 기여" if v > 0 else "▼ 저해"
    )
    return profile


# ──────────────────────────────────────────────
# 5. SHAP Dependence 데이터 (상위 피처)
# ──────────────────────────────────────────────
def shap_dependence_data(shap_values: np.ndarray,
                          X: pd.DataFrame,
                          top_n: int = 6) -> dict:
    """상위 n개 피처의 의존성 데이터 반환 (x, shap_y 쌍)"""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    result = {}
    for i in top_idx:
        feat = X.columns[i]
        result[feat] = {
            "x":      X.iloc[:, i].values,
            "shap_y": shap_values[:, i],
        }
    return result


# ──────────────────────────────────────────────
# 6. 통합 SHAP 분석 파이프라인
# ──────────────────────────────────────────────
def run_shap_analysis(model, X: pd.DataFrame,
                       processed_df: pd.DataFrame,
                       focus_country: str = "KOR") -> dict:
    """
    Returns
    -------
    dict with keys:
        shap_values, explainer, global_importance,
        category_contribution, country_profile, dependence_data
    """
    # import 부수효과 방지: 경고 억제는 module import 시점이 아니라 실행 진입 시점에만.
    warnings.filterwarnings("ignore")

    print("🔍 SHAP 분석 실행 중...")

    shap_vals, explainer = compute_shap_values(model, X)

    global_imp = global_feature_importance(shap_vals, X.columns.tolist())
    print(f"   상위 5 피처:\n{global_imp[['feature','importance_pct','category']].head()}")

    cat_contrib = category_contribution(global_imp)
    print("\n   카테고리별 기여도:")
    for _, row in cat_contrib.iterrows():
        print(f"      {row['category']:10s}: {row['weight_pct']:.1f}%")

    country_prof = country_shap_profile(shap_vals, X, processed_df, focus_country)
    dep_data     = shap_dependence_data(shap_vals, X, top_n=6)

    print("✅ SHAP 분석 완료")
    return {
        "shap_values":          shap_vals,
        "explainer":            explainer,
        "global_importance":    global_imp,
        "category_contribution":cat_contrib,
        "country_profile":      country_prof,
        "dependence_data":      dep_data,
        "base_value":           explainer.expected_value,
    }


# ──────────────────────────────────────────────
# 7. 결과 직렬화 (xai_results.json 용)
# ──────────────────────────────────────────────
def serialize_shap_results(shap_results: dict) -> dict:
    """shap_results → JSON 직렬화 가능 dict.

    반환 키: category_contribution · top10_features · korea_bottlenecks
    (xai_results.json 스키마 — 키 이름·순서는 소비처 계약).
    """
    return {
        "category_contribution": (
            shap_results["category_contribution"]
            [["category", "weight_pct"]].round(2).to_dict(orient="records")
        ),
        "top10_features": (
            shap_results["global_importance"]
            [["rank", "feature", "importance_pct", "category"]]
            .head(10).round(2).to_dict(orient="records")
        ),
        "korea_bottlenecks": (
            shap_results["country_profile"]
            [["feature", "shap_value", "category", "direction"]]
            .head(10).round(4).to_dict(orient="records")
            if not shap_results["country_profile"].empty else []
        ),
    }
