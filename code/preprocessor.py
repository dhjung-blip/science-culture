"""
preprocessor.py
===============
데이터 전처리 및 피처 엔지니어링 모듈

- 결측치 처리 (국가 내 시계열 보간 = group-aware, fold 밖 / KNN·median = fold 안)
- Time-lag 피처 생성 (3·5·7년 시차 — config.LAG_YEARS)
- 복합 지수 산출 (과학문화 지수, 물적규모 지수 등)
- 학습용 X / y 분리 — **이중 경로**:
    · prepare_xy_raw : 스케일 전 X_raw (CV fold-내 fit 용, 누수성 변환 없음)
    · prepare_xy     : 전체 데이터로 누수성 파이프라인 적합 → 최종 산출용 X_final

전처리 누수 제거(2026-06): 과거 run_preprocessing 은 KNNImputer·StandardScaler·median fill 을
**전체 데이터에 fit** 한 뒤 그 결과를 GroupKFold CV 에 넘겨 CV 가 테스트 폴드 정보를 흡수했다
(낙관 편향). 이제 누수성 변환(KNN·median·StandardScaler)은 build_leakage_pipeline() 으로 묶여
모델 단계에서 fold-train 에만 fit 된다(CV 경로). 최종 산출용 X_final 만 전체 데이터로 fit 한다.
group-aware 변환(국가 내 보간·lag shift·복합지수)은 GroupKFold 가 국가 단위로 분할하므로
fold 밖에 유지해도 누수가 아니다(같은 국가가 train/val 에 동시 등장하지 않음).
"""

import pandas as pd
from config import ALL_FEATURES, FEATURE_GROUPS, LAG_YEARS, OUTCOME_COLS
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# 1. 피처 그룹 정의 — config.py 로 이동
# ──────────────────────────────────────────────
# FEATURE_GROUPS·ALL_FEATURES·OUTCOME_COLS·LAG_YEARS 의 정의·도메인 주석은 config.py 에 있다.
# replicate_paper.py 가 preprocessor.FEATURE_GROUPS / preprocessor.ALL_FEATURES 를
# monkey-patch 하므로 반드시 이름 바인딩(`from config import X`)으로 가져온다
# (함수 본문에서 `config.X` 직접 참조 금지 — v1 재현 계약).


# ──────────────────────────────────────────────
# 2. 결측치 처리 — group-aware 보간 (fold 밖) vs 누수성 변환 (fold 안, build_leakage_pipeline)
# ──────────────────────────────────────────────
def interpolate_within_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    국가 내 시계열 선형 보간 (group-aware → GroupKFold 국가 분할이라 누수 아님).

    과거 impute_missing 은 여기서 곧장 KNNImputer 까지 전체 데이터에 fit 했으나,
    KNN(교차국가 이웃 사용)은 폴드 간 정보를 섞어 CV 낙관 편향을 유발한다.
    KNN 잔여 대체·median·StandardScaler 는 build_leakage_pipeline() 으로 분리되어
    모델 단계의 fold-train 에만 적합된다. 이 함수는 국가 내부 보간만 수행한다.

    타깃(OUTCOME_COLS)은 보정에서 제외해 실관측만 사용한다
    (교차국가 KNN 타깃 날조 및 타깃→피처 누수 방지).
    """
    df = df.copy()
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    # 국가별 선형 보간 (group-aware — 같은 국가 시계열 내부에서만 채움)
    df[feature_cols] = df.groupby("country")[feature_cols].transform(
        lambda g: g.interpolate(method="linear", limit_direction="both")
    )

    return df


def build_leakage_pipeline(base_cols: list[str],
                           rest_cols: list[str]) -> Pipeline:
    """누수성 변환(KNN·median·StandardScaler)을 묶은 **미적합** sklearn Pipeline.

    이 한 객체를 fold-train 에만 fit 하면(model_trainer.timeseries_cv) CV 누수가
    사라지고, 전체 데이터에 fit 하면(prepare_xy) 기존과 동등한 최종 산출 X 를 얻는다.

    위상은 과거 코드를 최대한 복제한다(R1 동등성):
      · base_cols  → KNNImputer(n_neighbors=5)   (과거 H1: base 잔여결측 KNN 대체)
      · rest_cols  → SimpleImputer(median)        (과거 M1: lag head-NaN median, idx 는 무결성 통과)
      · 이후 전 컬럼 → StandardScaler              (과거 H2: 표준화)

    Parameters
    ----------
    base_cols : KNN 으로 채울 원천 피처 컬럼(ALL_FEATURES 교집합).
    rest_cols : median 으로 채울 파생 컬럼(idx_* · *_lag*).

    Notes
    -----
    ColumnTransformer 출력 컬럼 순서는 base_cols + rest_cols 이다. 호출부는
    이 순서로 DataFrame 을 만든 뒤 원래 feat_cols 순서로 재정렬한다.
    """
    impute = ColumnTransformer(
        transformers=[
            ("knn", KNNImputer(n_neighbors=5), base_cols),
            ("median", SimpleImputer(strategy="median"), rest_cols),
        ],
        remainder="drop",
    )
    return Pipeline([("impute", impute), ("scale", StandardScaler())])


# ──────────────────────────────────────────────
# 3. Time-lag 피처 생성
# ──────────────────────────────────────────────
def create_lag_features(df: pd.DataFrame,
                         lag_cols: list | None = None,
                         lag_years: list | None = None) -> pd.DataFrame:
    """
    과학문화, 보상 등 Input 지표의 n년 시차(lag) 버전 생성
    → 혁신 성과에 대한 선행 효과 포착
    """
    df = df.copy().sort_values(["country", "year"])
    lag_cols  = lag_cols  or FEATURE_GROUPS["sci_culture"] + FEATURE_GROUPS["economic"]
    lag_years = lag_years or LAG_YEARS

    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lag_years:
            df[f"{col}_lag{lag}"] = (
                df.groupby("country")[col]
                  .shift(lag)
            )
    return df


# ──────────────────────────────────────────────
# 4. 복합 지수 산출
# ──────────────────────────────────────────────
def build_composite_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    4대 분류별 단순 평균 복합 지수 산출 (0–1 스케일 정규화 후)
    → SHAP 요약 분석 및 시나리오 시뮬레이션에 활용
    """
    df = df.copy()

    # L3(전처리 누수 분류): 복합지수 min-max 정규화는 전체 컬럼 통계(min/max)를 쓰므로
    # 엄밀히는 fold 간 정보를 본다. 그러나 ① 단조변환(monotonic)이라 트리 분기 순서를
    # 바꾸지 않아 XGBoost 예측에 **무해**하고, ② 폴드 안으로 옮겨도 수치 이득이 0 에 가깝다.
    # → 비용(파이프라인 복잡도) 대비 이득이 없어 의도적으로 fold 밖에 유지한다.
    def _safe_norm(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + 1e-9)

    for group_name, cols in FEATURE_GROUPS.items():
        avail = [c for c in cols if c in df.columns]
        if not avail:
            continue
        normalized = df[avail].apply(_safe_norm)
        df[f"idx_{group_name}"] = normalized.mean(axis=1)

    return df


# ──────────────────────────────────────────────
# 5. 학습용 X / y 분리 — 이중 경로 (CV fold-내 fit 용 raw / 최종 산출용 scaled)
# ──────────────────────────────────────────────
def prepare_xy_raw(df: pd.DataFrame,
                   target: str = "innovation_score",
                   include_lag: bool = True,
                   drop_threshold: float = 0.3
                   ) -> tuple[pd.DataFrame, pd.Series, list, list, list]:
    """스케일 **전** X_raw 와 컬럼 분할을 반환한다(누수성 변환 없음).

    fillna(median)·StandardScaler·KNN 잔여대체는 일절 하지 않는다 — 이들은
    build_leakage_pipeline() 으로 묶여 model_trainer 에서 fold-train 에만 fit 된다.
    drop_threshold 컬럼 제거는 전역(전체 데이터 결측율) 1회만 수행한다(컬럼 선택은
    데이터 스키마 결정이라 fold 간 동일해야 하며, 값이 아니라 존재 여부만 본다).

    Parameters
    ----------
    df              : build_composite_indices 결과 DataFrame
    target          : 종속변수 컬럼명
    include_lag     : Time-lag 피처 포함 여부
    drop_threshold  : 결측 비율이 이 값 초과인 컬럼 제거(전역 1회)

    Returns
    -------
    X_raw, y, feat_cols, base_cols, rest_cols
        · X_raw     : 미충전·미스케일 피처 행렬 (feat_cols 순서)
        · base_cols : feat_cols 중 ALL_FEATURES 원천 피처 (KNN 대상)
        · rest_cols : feat_cols 중 idx_* · *_lag* 파생 피처 (median 대상)
    """
    df_model = df.dropna(subset=[target]).copy()

    # 사용할 피처 컬럼 선택
    base_feats = [c for c in ALL_FEATURES if c in df_model.columns]
    idx_feats  = [c for c in df_model.columns if c.startswith("idx_")]
    lag_feats  = [c for c in df_model.columns if "_lag" in c] if include_lag else []
    feat_cols  = base_feats + idx_feats + lag_feats

    # 결측 비율 높은 컬럼 제거 (전역 1회 — 컬럼 스키마 결정, fold 불변)
    missing_rate = df_model[feat_cols].isnull().mean()
    feat_cols = [c for c in feat_cols if missing_rate[c] <= drop_threshold]

    X_raw = df_model[feat_cols].copy()
    y = df_model[target]

    base_cols = [c for c in feat_cols if c in base_feats]
    rest_cols = [c for c in feat_cols if c not in base_feats]

    return X_raw, y, feat_cols, base_cols, rest_cols


def prepare_xy(df: pd.DataFrame,
               target: str = "innovation_score",
               include_lag: bool = True,
               drop_threshold: float = 0.3
               ) -> tuple[pd.DataFrame, pd.Series, list, Pipeline]:
    """최종 산출용 X_final 을 **전체 데이터로 fit** 한 누수성 파이프라인으로 만든다.

    CV 가 아닌 최종 모델 학습·SHAP·시나리오에 쓰일 X 다(전체 데이터 fit 은 최종
    산출에서는 정당 — 평가가 아니므로 누수 개념이 없다). 반환되는 fitted pipe 는
    재사용 목적이 아니라 계약 가시성을 위해 함께 돌려준다(scenario_simulator 는
    X 가 이미 스케일 공간이라 가정하고 σ-shift 하므로 pipe 를 다시 쓰지 않는다).

    Returns
    -------
    X_final, y, feat_cols, fitted_pipeline
    """
    X_raw, y, feat_cols, base_cols, rest_cols = prepare_xy_raw(
        df, target=target, include_lag=include_lag, drop_threshold=drop_threshold
    )

    pipe = build_leakage_pipeline(base_cols, rest_cols)
    transformed = pipe.fit_transform(X_raw)

    # ColumnTransformer 출력 순서 = base_cols + rest_cols → 원래 feat_cols 순서로 재정렬
    ct_order = base_cols + rest_cols
    X_final = pd.DataFrame(transformed, columns=ct_order, index=X_raw.index)[feat_cols]

    return X_final, y, feat_cols, pipe


# ──────────────────────────────────────────────
# 6. 전체 파이프라인 실행
# ──────────────────────────────────────────────
# 스테이지 경계 데이터 계약 (fail-loud 게이트용)
REQUIRED_META_COLS = ["country", "year", "country_name"]
REQUIRED_TARGET_COLS = OUTCOME_COLS  # 기본 2종 + 탈순환 2종(§7.7, 실 WB 전용 — --no-wb 경로엔 부재 가능)
MIN_FEATURE_COVERAGE = 0.5  # FEATURE_GROUPS 멤버 중 raw_df 에 있어야 하는 최소 비율


def _validate_input_contract(raw_df: pd.DataFrame, target: str) -> None:
    """
    스테이지 경계 데이터 계약 검증 (fail-loud).

    1) 메타 컬럼 (country, year, country_name) 전부 존재.
    2) 타깃 컬럼이 둘 다 없으면 실패. (개별 타깃 부재는 prepare_xy 가 dropna 로 처리)
    3) FEATURE_GROUPS 멤버 중 50% 미만이 raw_df 에 있으면 실패
       → graceful 폴백을 거치며 침묵 붕괴되는 상황 차단.

    누락 컬럼을 정확히 명시해 데이터 수집 단계 회귀를 즉시 드러낸다.
    """
    # (1) 메타 컬럼
    missing_meta = [c for c in REQUIRED_META_COLS if c not in raw_df.columns]
    if missing_meta:
        raise ValueError(
            f"raw_df 메타 컬럼 누락: {missing_meta}. "
            f"필수 메타={REQUIRED_META_COLS}. "
            "data_collector.build_full_dataset 출력 계약(docs/architecture.md §2)을 확인하세요."
        )

    # (2) 타깃 컬럼 — 적어도 하나
    present_targets = [c for c in REQUIRED_TARGET_COLS if c in raw_df.columns]
    if not present_targets:
        raise ValueError(
            f"raw_df 에 타깃 컬럼이 하나도 없음: 후보={REQUIRED_TARGET_COLS}. "
            "타깃 후보(OUTCOME_COLS) 중 최소 하나가 필요합니다."
        )
    if target not in raw_df.columns:
        raise ValueError(
            f"지정 타깃 '{target}' 이 raw_df 에 없음 (있는 타깃: {present_targets}). "
            "--target 인자 또는 data_collector 출력 확인. "
            "탈순환 타깃(articles_per_m_log·patents_per_m_log)은 WB population 기반 파생이라 "
            "실 WB 모드 필요 (--no-wb 비호환)."
        )

    # (3) FEATURE_GROUPS 커버리지
    total = len(ALL_FEATURES)
    present = [f for f in ALL_FEATURES if f in raw_df.columns]
    missing_feats = [f for f in ALL_FEATURES if f not in raw_df.columns]
    coverage = len(present) / total if total else 0.0
    if coverage < MIN_FEATURE_COVERAGE:
        raise ValueError(
            f"FEATURE_GROUPS 멤버 {len(present)}/{total} 만 존재 "
            f"(커버리지 {coverage:.1%} < {MIN_FEATURE_COVERAGE:.0%}) — "
            f"누락: {missing_feats}. 데이터 수집 단계 점검 필요."
        )


def run_preprocessing(raw_df: pd.DataFrame,
                      target: str = "innovation_score",
                      lag_years: list | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list, Pipeline, pd.DataFrame]:
    """
    raw_df → 전처리 → 피처 엔지니어링 → (X_final · X_raw) 반환.

    이중 경로(전처리 누수 제거):
      · X_final : 전체 데이터로 fit 한 누수성 파이프라인 출력 — 최종 모델·SHAP·시나리오용.
      · X_raw   : 미충전·미스케일 피처 — model_trainer 가 fold-train 에만 누수성
                  파이프라인을 fit 하도록 넘기는 CV 입력. final_pipe 는 그 미적합
                  원형(make_pipeline 으로 fold 마다 clone) 제공용으로 함께 반환.

    group-aware 변환(국가 내 보간·lag shift·복합지수)은 fold 밖에서 1회 수행한다
    (GroupKFold 국가 분할이라 누수 아님). 누수성 변환(KNN·median·StandardScaler)만
    경로가 갈린다(X_final=전체 fit, CV=fold-train fit).

    Returns
    -------
    processed_df, X_final, y, feature_names, final_pipe, X_raw
    """
    print("⚙️  전처리 파이프라인 시작...")

    # 스테이지 경계 데이터 계약 검증 (fail-loud)
    _validate_input_contract(raw_df, target)

    df = interpolate_within_country(raw_df)
    print("   [1/4] 국가 내 시계열 보간 완료 (KNN·median·scale 은 모델 fold 안에서)")

    df = create_lag_features(df, lag_years=lag_years or LAG_YEARS)
    print(f"   [2/4] Time-lag 피처 생성 완료 (lag={lag_years or LAG_YEARS})")

    df = build_composite_indices(df)
    print("   [3/4] 복합 지수 산출 완료")

    # 단일 prepare_xy_raw 호출로 X_raw·컬럼분할 확정 → 동일 index·컬럼순서로 X_final 도 생성
    X_raw, y, feat_names, base_cols, rest_cols = prepare_xy_raw(df, target=target)

    # 출력 계약 검증: 피처 0개면 모든 후속 단계가 무의미 → 즉시 실패
    if X_raw.shape[1] == 0:
        raise ValueError(
            "prepare_xy_raw 결과 X 컬럼 수가 0. "
            f"feat_cols 누락 — 입력 피처/결측 임계(drop_threshold)/타깃 dropna 결과 점검. "
            f"raw_df shape={raw_df.shape}, target='{target}'."
        )

    # 최종 산출용: 전체 데이터로 누수성 파이프라인 fit (평가 아님 → 누수 개념 없음)
    final_pipe = build_leakage_pipeline(base_cols, rest_cols)
    transformed = final_pipe.fit_transform(X_raw)
    ct_order = base_cols + rest_cols
    X_final = pd.DataFrame(transformed, columns=ct_order, index=X_raw.index)[feat_names]

    print(f"   [4/4] 학습 데이터 준비 완료: X={X_final.shape}, y={y.shape}")
    print(f"         피처 수: {len(feat_names)}")

    return df, X_final, y, feat_names, final_pipe, X_raw


if __name__ == "__main__":
    from data_collector import build_full_dataset
    raw = build_full_dataset(use_worldbank=False)
    processed, X, y, features, pipe, X_raw = run_preprocessing(raw)
    print(X.describe().round(3))
