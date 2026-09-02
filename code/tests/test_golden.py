"""
시드 고정 골든 회귀 테스트 (opt-in)
====================================
``--no-wb`` 결정적 경로(P1 재현성)에서 모델 핵심 수치가 *허용 오차 안*에 있는지 검증한다.
ML 스택(xgboost/shap/sklearn)이 무거우므로 ``@pytest.mark.golden`` 으로 옵트인한다 —
기본 ``pytest`` 실행에서는 빠지고(`addopts = -m "not golden"`), ``pytest -m golden`` 으로만 돈다.

목적:
  정본 스테이지(data_collector·preprocessor·model_trainer) 변경 시
  *결과 수치*가 실수로 변하는 회귀를 즉시 드러낸다.
  AST 구조 테스트(test_architecture.py)와 상보적이다(구조 vs 수치).

골든 값:
  build_full_dataset(use_worldbank=False, use_fred=False, seed=42)
  → run_preprocessing(target='innovation_score')
  → run_training(X, y, processed_df)  [random_state=42 고정]
  를 두 번 돌려 완전 결정적임을 확인한 뒤 박았다(허용 오차는 라이브러리
  마이너 업그레이드 정도의 수치 흔들림을 흡수하는 수준).

허용 오차 정책:
  * CV R²·train R² : ±0.02 (XGBoost 빌드/스케줄러 미세변동 흡수)
  * RMSE·MAE       : ±0.10 (R²·y std 스케일에서 약 1~3% 변동 허용)
  * 한국 baseline  : ±0.50 (GII 0~100 스케일에서 ~1% 변동)
  * 정수(피처 수/표본 수): 정확 일치(데이터 계약 변경 즉시 실패)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# code/ 를 모듈 경로에 추가 (pyproject.toml [tool.pytest.ini_options].pythonpath 와 동일).
# (collect 시점엔 ML 스택 없이도 import 가능해야 하므로, 정본 모듈은 테스트 함수 안에서 import 한다.)
_FILES_DIR = Path(__file__).resolve().parent.parent
if str(_FILES_DIR) not in sys.path:
    sys.path.insert(0, str(_FILES_DIR))


# ──────────────────────────────────────────────────────────────────────────
# 골든 상수 — 시드 고정 --no-wb 경로의 측정값
# ──────────────────────────────────────────────────────────────────────────
# 측정 환경: Python 3.12 / xgboost 3.2.0 / scikit-learn / numpy.default_rng(42).
# 두 번 실행해 비트단위 동일 확인 (RMSE/R²/train_r2 모두 일치).
# 2026-06 갱신 ①: WB 실 4피처 추가 (운영구조 +2, 보상수준 +1, 과학문화 +1) — 4범주 키 고정.
# 2026-06 갱신 ②: PISA 2022 앵커 추가 (OECD 공식 발표분, KOR 527/528/515 교차검증)
#   → 동결 5.2년 → 1.2년, --no-wb 경로도 PISA 로컬 파일 사용이라 수치 변동 (정당한 데이터 갱신).
# 2026-06 갱신 ③ (전처리 누수 제거): KNN·median·StandardScaler 를 build_leakage_pipeline 으로
#   묶어 CV 의 **fold-train 에만 fit** 하도록 전환(과거: 전체 데이터 fit 후 CV 에 전달 → 누수).
#   부수로 lag/idx 가 KNN-충전 base 가 아니라 국가내-보간 base 에서 파생되도록 위상이 바뀜
#   (단일 X_raw 공유 설계). 두 효과의 합으로 CV 수치가 소폭 변동했다:
#     · CV R²   0.2952 → 0.2980   · RMSE 3.5485 → 3.5542   · MAE 2.8716 → 2.8983
#   누수 격리 측정(동일 위상): 누수 CV 0.2928 vs fold-local CV 0.2980 → 순수 누수효과 -0.0052
#   (이 패널은 국가내 보간이 결측 698/722 를 먼저 채워 KNN 이 24셀만 건드리고, 228표본에서
#    StandardScaler 모먼트가 fold-train↔전체 거의 동일 → 누수는 **구조적 존재·수치적 미미**).
#   ★최종 경로(②)는 거의 불변 — train_r2 0.9585→0.9597, KOR baseline 56.78→56.78 (둘 다 허용 내).
# 20 base + 4 idx + 11 lag = 35 (lag 대상: sci_culture 7 + economic 4 = 11).
GOLDEN_N_FEATURES = 35            # 실데이터(WVS·PISA·GII) 병합 후 20 base + 4 idx + 11 lag
GOLDEN_N_SAMPLES = 228            # 240 cells − GII 미수록 12 (dropna(subset=[target]))
GOLDEN_CV_RMSE = 3.5542           # ±0.10 (전처리 누수 제거 후 fold-local CV)
GOLDEN_CV_R2 = 0.2980             # ±0.02 (fold-local CV — 누수 제거)
GOLDEN_CV_MAE = 2.8983            # ±0.10 (fold-local CV — 누수 제거)
GOLDEN_TRAIN_R2 = 0.9597          # ±0.02 (최종 경로 전체 fit — 거의 불변)
GOLDEN_KOR_BASELINE = 56.78       # ±0.50 (model.predict 평균, GII 스케일 — 누수 제거 후에도 불변)

# 허용 오차 (위 정책)
TOL_R2 = 0.02
TOL_ERR = 0.10
TOL_BASELINE = 0.50


def _import_stages() -> tuple:
    """ML 스택 부재 시 테스트를 skip 처리하며 정본 3 스테이지를 로드한다.

    ``pytest.importorskip`` 는 numpy/xgboost/sklearn 미설치 환경(예: 시스템
    python·post_edit 훅)에서도 collect 가 통과되도록 한다(opt-in 마크에
    더해 2중 안전망). 정본 스테이지를 모듈 상단이 아니라 여기서 import 하는
    이유는 동일하다 — 무거운 ML 스택을 collect 시 강제하지 않기 위함.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    import data_collector
    import model_trainer
    import preprocessor

    return data_collector, preprocessor, model_trainer


@pytest.mark.golden
def test_golden_v2_default_pipeline() -> None:
    """v2(확장 분석) --no-wb 결정적 경로의 모델 수치 골든 회귀.

    검증 대상:
      1) 데이터 계약 — 피처/표본 정수 일치(컬럼 추가/제거 즉시 실패)
      2) 모델 수치  — CV R²·RMSE·MAE·train R² 가 허용 오차 안
      3) 한국 baseline — model.predict(X[kor]) 평균이 골든 ±0.5
    """
    d, pp, m = _import_stages()

    # 시드 고정 데이터 (build_full_dataset 진입 시 np.random.default_rng(42) 새로 생성)
    raw = d.build_full_dataset(use_worldbank=False, use_fred=False, seed=42)
    processed, X, y, feat_names, _pipe, X_raw = pp.run_preprocessing(raw)
    # 전처리 누수 제거: CV 의 fold-내 fit 용 factory 주입 (main.py 배선과 동일 규칙).
    base_cols = [c for c in X_raw.columns if not (c.startswith("idx_") or "_lag" in c)]
    rest_cols = [c for c in X_raw.columns if c.startswith("idx_") or "_lag" in c]
    model, metrics = m.run_training(
        X, X_raw, y, processed,
        make_pipeline=lambda: pp.build_leakage_pipeline(base_cols, rest_cols),
    )

    # (1) 데이터 계약 — 정확 일치
    assert len(feat_names) == GOLDEN_N_FEATURES, (
        f"피처 수 회귀: {len(feat_names)} != {GOLDEN_N_FEATURES} "
        "(FEATURE_GROUPS 또는 lag/idx 생성 로직 변경 — 의도라면 골든값 갱신 필요)"
    )
    assert len(y) == GOLDEN_N_SAMPLES, (
        f"표본 수 회귀: {len(y)} != {GOLDEN_N_SAMPLES} "
        "(GII 실타깃 dropna 결과 변경 — 의도라면 골든값 갱신 필요)"
    )
    assert X.shape == (GOLDEN_N_SAMPLES, GOLDEN_N_FEATURES), (
        f"X.shape 회귀: {X.shape} != ({GOLDEN_N_SAMPLES}, {GOLDEN_N_FEATURES})"
    )

    # (2) 모델 수치 — 허용 오차
    assert abs(metrics["r2"] - GOLDEN_CV_R2) < TOL_R2, (
        f"CV R² 회귀: {metrics['r2']:.4f} (골든 {GOLDEN_CV_R2:.4f}, 허용 ±{TOL_R2})"
    )
    assert abs(metrics["rmse"] - GOLDEN_CV_RMSE) < TOL_ERR, (
        f"CV RMSE 회귀: {metrics['rmse']:.4f} (골든 {GOLDEN_CV_RMSE:.4f}, 허용 ±{TOL_ERR})"
    )
    assert abs(metrics["mae"] - GOLDEN_CV_MAE) < TOL_ERR, (
        f"CV MAE 회귀: {metrics['mae']:.4f} (골든 {GOLDEN_CV_MAE:.4f}, 허용 ±{TOL_ERR})"
    )
    assert abs(metrics["train_r2"] - GOLDEN_TRAIN_R2) < TOL_R2, (
        f"train R² 회귀: {metrics['train_r2']:.4f} (골든 {GOLDEN_TRAIN_R2:.4f}, 허용 ±{TOL_R2})"
    )

    # (3) 한국 baseline — SHAP/시나리오 미경유, 순수 모델 평균예측
    kor_mask = processed.loc[y.index, "country"] == "KOR"
    assert kor_mask.any(), "한국(KOR) 행이 학습셋에 없음 — COUNTRIES/GII 매칭 회귀"
    kor_pred = float(model.predict(X[kor_mask]).mean())
    assert abs(kor_pred - GOLDEN_KOR_BASELINE) < TOL_BASELINE, (
        f"한국 baseline 회귀: {kor_pred:.2f} (골든 {GOLDEN_KOR_BASELINE:.2f}, 허용 ±{TOL_BASELINE})"
    )


@pytest.mark.golden
def test_golden_determinism_two_runs() -> None:
    """동일 시드(42)·동일 입력에서 두 번 실행 시 비트단위 동일성 확인.

    P1 재현성의 기반 가드: build_full_dataset 이 호출별 로컬 rng 를 새로
    만들어야 두 번째 호출에서도 결정적이다. 합성 폴백이 모듈 전역 rng 를
    공유하면 이 테스트가 실패한다(회귀 즉시 감지).
    """
    d, pp, m = _import_stages()

    def _measure() -> tuple[float, float, float, int]:
        raw = d.build_full_dataset(use_worldbank=False, use_fred=False, seed=42)
        processed, X, y, feat_names, _pipe, X_raw = pp.run_preprocessing(raw)
        base_cols = [c for c in X_raw.columns if not (c.startswith("idx_") or "_lag" in c)]
        rest_cols = [c for c in X_raw.columns if c.startswith("idx_") or "_lag" in c]
        _model, metrics = m.run_training(
            X, X_raw, y, processed,
            make_pipeline=lambda: pp.build_leakage_pipeline(base_cols, rest_cols),
        )
        return metrics["r2"], metrics["rmse"], metrics["train_r2"], len(feat_names)

    r2_a, rmse_a, train_a, n_a = _measure()
    r2_b, rmse_b, train_b, n_b = _measure()

    # 비트단위 동일 (XGBoost random_state=42 + np.default_rng(seed=42) → 결정적)
    assert r2_a == r2_b, f"CV R² 비결정성: {r2_a} != {r2_b}"
    assert rmse_a == rmse_b, f"CV RMSE 비결정성: {rmse_a} != {rmse_b}"
    assert train_a == train_b, f"train R² 비결정성: {train_a} != {train_b}"
    assert n_a == n_b, f"피처 수 비결정성: {n_a} != {n_b}"
