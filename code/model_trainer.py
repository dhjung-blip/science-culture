"""
model_trainer.py
================
XGBoost 학습 모듈

- Optuna를 사용한 하이퍼파라미터 최적화
- Time-series 기반 교차검증 (GroupKFold)
- 모델 성능 평가 (RMSE, MAE, R²)
"""

import hashlib
import logging
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

# 전처리 누수 제거: 누수성 변환(KNN·median·StandardScaler)을 fold-train 에만 적합하기
# 위해, main 이 미적합 파이프라인 factory(() -> estimator)를 주입한다. model_trainer 는
# preprocessor 를 import 하지 않는다(스테이지 상호 독립 불변식 — architecture.md §3).
# clone 은 sklearn.base 에서 가져온다(third-party 허용). factory 의 출처는 모른다.
PipelineFactory = Callable[[], BaseEstimator]

# 모듈 로거 — 폴백·경고(⚠) 전용 (ADR-0007). 핸들러 미구성: logging.lastResort 가
# stderr 로 출력해 stdout(진행 리포트 print)/stderr(경고)가 분리된다.
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 1. 기본 XGBoost 모델
# ──────────────────────────────────────────────
DEFAULT_XGB_PARAMS = {
    "n_estimators":    300,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.70,
    "colsample_bytree":0.60,
    "min_child_weight":5,
    "reg_alpha":       1.0,
    "reg_lambda":      3.0,
    "gamma":           0.5,
    "random_state":    42,
    "n_jobs":          -1,
    "verbosity":       0,
}

# 조기중단(early stopping) 설정
# - 데이터가 --country-set 으로 228~1604행까지 변하므로 n_estimators 고정 대신
#   inner-val 성능이 정체되면 트리 추가를 멈춰 데이터 규모에 자동 적응한다.
EARLY_STOPPING_ROUNDS = 40
EARLY_STOPPING_MAX_TREES = 1500  # 조기중단 경로에서만 쓰는 높은 트리 상한


def build_xgboost(params: dict | None = None) -> xgb.XGBRegressor:
    """XGBoost 회귀 모델 생성"""
    p = {**DEFAULT_XGB_PARAMS, **(params or {})}
    return xgb.XGBRegressor(**p)


def _fit_early_stopping(params: dict, X_train: np.ndarray | pd.DataFrame,
                        y_train: pd.Series) -> xgb.XGBRegressor:
    """
    학습 폴드를 inner-split 하여 조기중단으로 트리 수를 자동 결정한다.

    ★누수 방지: eval_set 은 **학습 데이터 내부에서 다시 떼어낸 inner-val**(X_es)이며,
    CV 테스트 폴드는 절대 들어가지 않는다. 호출부(timeseries_cv·train_final_model)는
    이 함수에 자기 학습 분할만 넘긴다(timeseries_cv 는 fold-train 변환 후 ndarray 를 넘김).
    """
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    p = {**DEFAULT_XGB_PARAMS, **params, "n_estimators": EARLY_STOPPING_MAX_TREES}
    model = xgb.XGBRegressor(**p, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    return model


# ──────────────────────────────────────────────
# 2. 시계열 기반 교차검증
# ──────────────────────────────────────────────
def timeseries_cv(params: dict, X_raw: pd.DataFrame, y: pd.Series,
                   groups: pd.Series, make_pipeline: PipelineFactory,
                   n_splits: int = 5) -> dict:
    """
    GroupKFold(country) CV: 같은 국가가 train/valid 에 안 섞이게 분할
    → **처음 보는 국가로 일반화 검증**. (연도는 분리하지 않으므로 시간누수 방지가 아님.)

    ★전처리 누수 제거: 누수성 변환(KNN·median·StandardScaler)을 **fold-train 에만**
    fit 한다. 폴드마다 make_pipeline() 으로 미적합 파이프라인을 만들어 clone 후
    fit_transform(train) / transform(val) → 테스트 폴드 통계가 학습에 새지 않는다.
    (과거: 전체 데이터에 fit 한 X 를 CV 에 넘겨 낙관 편향 → 이 함수가 입력을 X_raw 로 받음.)

    각 폴드는 학습 폴드를 inner-split 한 조기중단(_fit_early_stopping)으로 적합한다.
    ★누수 방지: 조기중단 eval_set 은 학습 폴드(변환 후) 내부의 inner-val 이며, 테스트
    폴드(X_val)는 적합·조기중단에 일절 사용하지 않고 오직 예측 평가에만 쓴다.

    Parameters
    ----------
    X_raw         : 미충전·미스케일 피처 (prepare_xy_raw 출력). 변환은 fold 안에서.
    make_pipeline : () -> 미적합 누수성 파이프라인 (main 이 build_leakage_pipeline 주입).
    """
    # 패널 정합성 가드: X·y·groups 는 동일 길이·동일 index 정렬이어야 한다
    # (architecture.md §3 불변식 ④ — `y.index` ↔ `processed_df` 정렬).
    if not (len(X_raw) == len(y) == len(groups)):
        raise ValueError(
            f"패널 정합성 깨짐 — len(X)={len(X_raw)}, len(y)={len(y)}, "
            f"len(groups)={len(groups)}"
        )
    if not X_raw.index.equals(y.index):
        raise ValueError("패널 정합성 깨짐 — X.index 와 y.index 가 일치하지 않음")

    gkf = GroupKFold(n_splits=n_splits)
    rmse_list, mae_list, r2_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups)):
        X_tr_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 누수성 변환을 fold-train 에만 fit → val 은 transform 만 (테스트 폴드 통계 차단)
        pipe = clone(make_pipeline())
        X_tr = pipe.fit_transform(X_tr_raw)
        X_val = pipe.transform(X_val_raw)

        model = _fit_early_stopping(params, X_tr, y_tr)

        preds = model.predict(X_val)
        rmse  = np.sqrt(mean_squared_error(y_val, preds))
        mae   = mean_absolute_error(y_val, preds)
        r2    = r2_score(y_val, preds)

        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

        print(f"   Fold {fold+1}/{n_splits}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    return {
        "rmse": np.mean(rmse_list), "rmse_std": np.std(rmse_list),
        "mae":  np.mean(mae_list),  "mae_std":  np.std(mae_list),
        "r2":   np.mean(r2_list),   "r2_std":   np.std(r2_list),
    }


# ──────────────────────────────────────────────
# 3. Optuna 하이퍼파라미터 탐색 (선택)
# ──────────────────────────────────────────────
def optimize_hyperparams(X_raw: pd.DataFrame, y: pd.Series,
                          groups: pd.Series, make_pipeline: PipelineFactory,
                          n_trials: int = 30, n_splits: int = 4) -> dict:
    """Optuna 기반 하이퍼파라미터 자동 탐색.

    ★전처리 누수 제거: 탐색 목적함수의 CV 도 fold-train 에만 누수성 파이프라인을
    fit 한다(전체 fit X 를 쓰면 best params 가 누수 신호에 과적합). 입력은 X_raw.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("   ⚠ Optuna 미설치 – 기본 파라미터 사용")
        return DEFAULT_XGB_PARAMS

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 3.0),
            "random_state": 42, "n_jobs": -1, "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        gkf = GroupKFold(n_splits=n_splits)
        scores = []
        for tr_idx, val_idx in gkf.split(X_raw, y, groups):
            pipe = clone(make_pipeline())
            X_tr = pipe.fit_transform(X_raw.iloc[tr_idx])
            X_val = pipe.transform(X_raw.iloc[val_idx])
            model.fit(X_tr, y.iloc[tr_idx], verbose=False)
            preds = model.predict(X_val)
            scores.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = {**DEFAULT_XGB_PARAMS, **study.best_params}
    print(f"   ✅ 최적 RMSE: {study.best_value:.4f}")
    return best


# ──────────────────────────────────────────────
# 4. 최종 모델 학습
# ──────────────────────────────────────────────
def train_final_model(X: pd.DataFrame, y: pd.Series,
                       params: dict | None = None) -> xgb.XGBRegressor:
    """
    전체 데이터로 최종 모델 학습.

    조기중단(_fit_early_stopping)으로 트리 수를 데이터 규모에 맞춰 자동 결정한다.
    eval_set 은 전체 데이터를 inner-split 한 inner-val 이다(외부 테스트셋 미사용).
    """
    return _fit_early_stopping(params or {}, X, y)


# ──────────────────────────────────────────────
# 5. 통합 학습 파이프라인
# ──────────────────────────────────────────────
def _compute_input_hash(X: pd.DataFrame, y: pd.Series) -> str:
    """X·y 의 모양·앞 1KB 바이트로 SHA256 첫 16자를 만들어 재현성 추적용 지문을 반환한다.

    전체 바이트 해싱은 비싸고, 모양+선두 청크 조합으로 데이터셋 동일성을
    실용적으로 판별한다(동일 데이터→동일 해시·다른 데이터→거의 확실히 다른 해시).
    """
    data_str = (
        f"{X.shape}_{y.shape}_{X.values.tobytes()[:1000]}_{y.values.tobytes()[:1000]}"
    )
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]


def run_training(X_final: pd.DataFrame, X_raw: pd.DataFrame, y: pd.Series,
                  processed_df: pd.DataFrame,
                  make_pipeline: PipelineFactory,
                  optimize: bool = False,
                  n_trials: int = 30) -> tuple[xgb.XGBRegressor, dict]:
    """이중 경로 학습 — CV 는 fold-내 fit(누수 제거), 최종 모델은 전체 fit X_final.

    Parameters
    ----------
    X_final       : 전체 데이터로 fit 한 파이프라인 출력(prepare_xy) — 최종 모델·train_r2·
                    메타(input_hash·feature_names)의 기준. SHAP·시나리오도 이 X 를 쓴다.
    X_raw         : 미충전·미스케일 피처(prepare_xy_raw) — CV 의 fold-train fit 입력.
    make_pipeline : () -> 미적합 누수성 파이프라인 (main 이 build_leakage_pipeline 주입).

    Returns
    -------
    trained_model, cv_metrics
    """
    # import 부수효과 방지: 경고 억제는 module import 시점이 아니라 실행 진입 시점에만.
    warnings.filterwarnings("ignore")

    print("🤖 XGBoost 모델 학습 시작...")

    # 패널 정합성 가드: y 의 모든 index 가 processed_df.index 의 부분집합이어야 한다
    # (architecture.md §3 불변식 ④ — `groups = processed_df.loc[y.index, "country"]`
    #  는 이 정렬을 암묵적으로 전제한다).
    if not y.index.isin(processed_df.index).all():
        raise ValueError(
            "y.index 가 processed_df.index 의 부분집합이 아님 — 패널 정합성 깨짐"
        )
    # 이중 경로 정합성: X_final·X_raw 는 동일 표본(index)이어야 한다(같은 prepare_xy_raw 산물).
    if not X_final.index.equals(X_raw.index):
        raise ValueError("X_final.index 와 X_raw.index 가 불일치 — 이중 경로 정합성 깨짐")

    # 그룹: 국가 코드 (레이블 인코딩)
    groups = processed_df.loc[y.index, "country"].factorize()[0] \
             if "country" in processed_df.columns \
             else np.zeros(len(y), dtype=int)

    params = DEFAULT_XGB_PARAMS
    if optimize:
        print("   하이퍼파라미터 최적화 중 (Optuna)...")
        params = optimize_hyperparams(X_raw, y, pd.Series(groups),
                                       make_pipeline, n_trials=n_trials)

    # ── 경로① CV: 누수성 변환을 fold-train 에만 fit (X_raw + make_pipeline)
    print("   교차검증 중...")
    cv_metrics = timeseries_cv(params, X_raw, y, pd.Series(groups), make_pipeline)

    print("\n   📊 CV 결과 요약:")
    print(f"      RMSE: {cv_metrics['rmse']:.4f} ± {cv_metrics['rmse_std']:.4f}")
    print(f"      MAE : {cv_metrics['mae']:.4f}  ± {cv_metrics['mae_std']:.4f}")
    print(f"      R²  : {cv_metrics['r2']:.4f}  ± {cv_metrics['r2_std']:.4f}")

    # ── 경로② 최종 모델: 전체 데이터로 fit 한 X_final (산출용 — 평가 아님 → 누수 개념 없음)
    print("\n   최종 모델 학습 (전체 데이터)...")
    final_model = train_final_model(X_final, y, params)

    # 학습 데이터 내 성능
    train_preds = final_model.predict(X_final)
    train_r2 = r2_score(y, train_preds)
    print(f"   학습 R²: {train_r2:.4f}")

    cv_metrics["train_r2"] = train_r2

    # 과적합 gap: 학습 R² 와 CV R² 의 격차로 일반화 한계를 관측한다.
    overfit_gap = train_r2 - cv_metrics["r2"]
    cv_metrics["overfit_gap"] = overfit_gap
    if overfit_gap > 0.30:
        logger.warning(f"   ⚠ train–CV 격차 {overfit_gap:.2f}: 과적합/국가일반화 한계 "
                       "— 결과는 설명용, 처음 보는 국가 예측엔 주의")

    # 모델 영속화 메타: xai_results.json 에 자동 포함되어 재현성 추적에 쓰인다.
    # (X_final 기준 — 최종 모델·SHAP·시나리오가 보는 실제 피처 행렬.)
    cv_metrics["n_features"] = int(X_final.shape[1])
    cv_metrics["feature_names"] = list(X_final.columns)
    cv_metrics["n_samples"] = int(X_final.shape[0])
    cv_metrics["params_used"] = dict(params)
    cv_metrics["input_hash"] = _compute_input_hash(X_final, y)
    cv_metrics["random_state"] = int(params.get("random_state", 42))

    print("✅ 학습 완료")
    return final_model, cv_metrics


# ──────────────────────────────────────────────
# 6. 결과 직렬화 (xai_results.json 용)
# ──────────────────────────────────────────────
def serialize_cv_metrics(cv_metrics: dict) -> dict:
    """cv_metrics → JSON 직렬화 가능 dict (float 4자리 반올림)."""
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in cv_metrics.items()}


if __name__ == "__main__":
    from data_collector import build_full_dataset
    from preprocessor import build_leakage_pipeline, run_preprocessing
    raw = build_full_dataset(use_worldbank=False)
    processed, X_final, y, features, _pipe, X_raw = run_preprocessing(raw)
    # CV 의 fold-내 fit 용 factory 주입 — base/rest 컬럼 분할은 prepare_xy_raw 와 동일 규칙.
    base_cols = [c for c in X_raw.columns if not (c.startswith("idx_") or "_lag" in c)]
    rest_cols = [c for c in X_raw.columns if c.startswith("idx_") or "_lag" in c]
    model, metrics = run_training(
        X_final, X_raw, y, processed,
        make_pipeline=lambda: build_leakage_pipeline(base_cols, rest_cols),
    )
    print(metrics)
