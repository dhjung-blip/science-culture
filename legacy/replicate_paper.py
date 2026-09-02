"""
replicate_paper.py — 논문 v1 사양 재현 오케스트레이터
========================================================
초기 사양(합성 데이터·20개국·17피처·5시나리오) 그대로 파이프라인 실행.
정본 v2(code/main.py)와 별개 entry-point. 정본 스테이지 모듈·산출물 미변경.
※ 원고 본문 수치는 모두 main.py 경로(실측 44개국)에서 나온다 — 본 스크립트는 보조 재현용.

사용:
    cd code
    python replicate_paper.py [--output /tmp/sc_paper_v1] [--no-optimize] [--focus-country KOR]

설계
----
- main.py 와 같은 위치의 오케스트레이터(스테이지 모두 import 허용).
- v2 정본 스테이지 모듈(preprocessor / shap_analyzer / scenario_simulator)의
  공개 모듈 상수(FEATURE_GROUPS / CATEGORY_MAP / POLICY_SCENARIOS)를
  본 프로세스에서만 **monkey-patch** 하여 v1 사양으로 교체.
  (모듈 파일 자체는 정본 v2 그대로. import-time 부수효과 없음.)
- 데이터: data_collector.build_full_dataset(use_worldbank=False, use_fred=False) +
  실데이터 로더(PISA/WVS/GII/ILOSTAT/FRED) 무력화로 *합성* 데이터셋 100% 재현.

논문 사양 출처
--------------
- Table 1: 4대 카테고리 × 17변수 (sci_culture 4 + hard_rd 5 + structure 5 + economic 3).
- Table 6: 5개 정책 시나리오 — S1_과학문화_강화, S2_보상_개선, S3_과제기간_연장,
  S4_종합_혁신패키지, S5_물적규모_확대 (정본 baseline `b7322fe` git 보존본).
- 본 오케스트레이터는 위 git 보존 v1 정의를 *코드 내 인라인 상수*로 재현 (소스 코드는 정본 미변경).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any

import config
import data_collector
import pandas as pd
import preprocessor
import scenario_simulator
import shap_analyzer
from data_collector import build_full_dataset
from model_trainer import run_training, serialize_cv_metrics
from preprocessor import build_leakage_pipeline, run_preprocessing
from scenario_simulator import run_simulation, serialize_sim_results
from shap_analyzer import run_shap_analysis, serialize_shap_results
from visualizer import run_visualization

# ════════════════════════════════════════════════════════════════
# v1 사양 — 초기 사양 표 1·2·6 그대로
# ════════════════════════════════════════════════════════════════
# Table 1 (17변수, 4대 카테고리)
PAPER_V1_FEATURE_GROUPS: dict[str, list[str]] = {
    "sci_culture": [  # 과학문화 (Soft) — 4
        "sci_literacy",
        "self_efficacy",
        "sci_trust",
        "tech_acceptance",
    ],
    "hard_rd": [  # 물적 규모 (Hard) — 5
        "rd_gdp_pct",
        "researchers_per_m",
        "patent_apps",
        "journal_articles",
        "gdp_per_capita",
    ],
    "structure": [  # 운영 구조 — 5
        "avg_proj_duration",
        "admin_burden_ratio",
        "researcher_retire_age",
        "gov_edu_exp_pct",
        "internet_users_pct",
    ],
    "economic": [  # 보상 수준 — 3
        "relative_salary",
        "rd_budget_per_researcher",
        "hi_tech_export_pct",
    ],
}

# 17 base features + 4 idx_* 복합지수 → 21 카테고리 매핑.
# lag 피처는 shap_analyzer._get_category 가 split("_lag")[0] 로 기본 매핑.
PAPER_V1_CATEGORY_MAP: dict[str, str] = {
    # 과학문화 (Soft)
    "sci_literacy": "과학문화",
    "self_efficacy": "과학문화",
    "sci_trust": "과학문화",
    "tech_acceptance": "과학문화",
    "idx_sci_culture": "과학문화",
    # 물적 규모 (Hard)
    "rd_gdp_pct": "물적규모",
    "researchers_per_m": "물적규모",
    "patent_apps": "물적규모",
    "journal_articles": "물적규모",
    "gdp_per_capita": "물적규모",
    "idx_hard_rd": "물적규모",
    # 운영 구조 (Structure)
    "avg_proj_duration": "운영구조",
    "admin_burden_ratio": "운영구조",
    "researcher_retire_age": "운영구조",
    "gov_edu_exp_pct": "운영구조",
    "internet_users_pct": "운영구조",
    "idx_structure": "운영구조",
    # 보상 수준 (Economic)
    "relative_salary": "보상수준",
    "rd_budget_per_researcher": "보상수준",
    "hi_tech_export_pct": "보상수준",
    "idx_economic": "보상수준",
}

# Table 6: 5개 정책 시나리오 (정본 baseline `b7322fe` 보존본 그대로)
PAPER_V1_POLICY_SCENARIOS: dict[str, dict[str, Any]] = {
    "S1_과학문화_강화": {
        "description": "과학 문해력 및 자아효능감 집중 투자 (+20%)",
        "changes": {
            "sci_literacy": +20,
            "self_efficacy": +20,
            "tech_acceptance": +15,
        },
    },
    "S2_보상_개선": {
        "description": "연구원 상대 연봉 15% 인상 + 연구비 증액 20%",
        "changes": {
            "relative_salary": +15,
            "rd_budget_per_researcher": +20,
        },
    },
    "S3_과제기간_연장": {
        "description": "평균 과제 기간 2년 연장 (단기→장기 전환)",
        "changes": {
            "avg_proj_duration": +50,  # 4년 → 6년 ≈ +50%
            "admin_burden_ratio": -20,  # 행정 부담 경감
        },
    },
    "S4_종합_혁신패키지": {
        "description": "4대 분야 균형 투자 패키지",
        "changes": {
            "sci_literacy": +15,
            "relative_salary": +10,
            "avg_proj_duration": +30,
            "rd_gdp_pct": +10,
            "tech_acceptance": +10,
        },
    },
    "S5_물적규모_확대": {
        "description": "R&D 예산 GDP 대비 0.5%p 증액 + 연구원 수 10% 증가",
        "changes": {
            "rd_gdp_pct": +15,
            "researchers_per_m": +10,
        },
    },
}


# ════════════════════════════════════════════════════════════════
# 논문 보고 수치 (대조 출력용) — 별도 상수로 분리해 mypy 가 정확히 추론하도록.
# ════════════════════════════════════════════════════════════════
PAPER_CV_R2: float = 0.509
PAPER_RMSE: float = 2.29
PAPER_CATEGORY_PCT: dict[str, float] = {
    "보상수준": 30.4,
    "물적규모": 27.6,
    "과학문화": 24.2,
    "운영구조": 17.8,
}
PAPER_TOP5_FEATURES: list[str] = [
    "idx_economic",
    "idx_hard_rd",
    "journal_articles",
    "sci_literacy",
    "hi_tech_export",
]
PAPER_KOR_BASELINE: float = 51.18
PAPER_KOR_SCENARIO_DELTA_PCT: dict[str, float] = {
    "S1_과학문화_강화": -0.15,
    "S2_보상_개선": +0.06,
    "S3_과제기간_연장": 0.00,
    "S4_종합_혁신패키지": +0.21,
    "S5_물적규모_확대": +0.08,
}


# ════════════════════════════════════════════════════════════════
# v1 monkey-patch — 데이터 로더 무력화 + v1 상수 주입
# ════════════════════════════════════════════════════════════════
def _empty_loader(columns: list[str]):
    """data_collector 의 실데이터 로더를 무력화하는 빈 DataFrame factory."""

    def _loader(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(columns=columns)

    return _loader


def _apply_paper_v1_patches() -> None:
    """v2 모듈 → v1 사양 monkey-patch.

    - 데이터 로더(PISA/WVS/GII/ILOSTAT/FRED) 5종을 빈 DF 반환으로 무력화 →
      build_full_dataset 의 병합 블록이 graceful 스킵, 합성 데이터만 남는다.
    - preprocessor.FEATURE_GROUPS / shap_analyzer.CATEGORY_MAP /
      scenario_simulator.POLICY_SCENARIOS 을 v1 정의로 교체.
    - 본 프로세스 종료 시 자동 원복(파일 미변경, 다음 main.py 실행 영향 없음).
    - **fail-loud 계약**: 재할당은 setattr 의미라 대상 attribute 가 rename/중앙화로
      사라지면 조용히 *새* attribute 만 생기고 v1 이 v2 사양으로 돌게 된다
      (조용한 재현성 붕괴) → 패치 전 존재 검사로 즉시 AttributeError 를 던진다.
    """
    # 패치 대상 존재 검사 — 부재 시 setattr 전에 fail-loud
    patch_targets: list[tuple[Any, str]] = [
        (data_collector, "load_pisa_scores"),
        (data_collector, "load_wvs_indicators"),
        (data_collector, "load_gii_scores"),
        (data_collector, "load_ilostat_salary"),
        (data_collector, "collect_fred_data"),
        (preprocessor, "FEATURE_GROUPS"),
        (preprocessor, "ALL_FEATURES"),
        (shap_analyzer, "CATEGORY_MAP"),
        (scenario_simulator, "POLICY_SCENARIOS"),
    ]
    for module, attr in patch_targets:
        if not hasattr(module, attr):
            raise AttributeError(
                f"v1 재현 계약 깨짐: {module.__name__}.{attr} 부재 — "
                "중앙화/rename 여부 확인 "
                "(replicate_paper.py 가 이 attribute 를 monkey-patch 한다)"
            )

    # 실데이터 로더 5종 무력화 — 합성 100% 보장
    data_collector.load_pisa_scores = _empty_loader(
        ["country", "year", "sci_literacy", "pisa_math", "pisa_reading"]
    )
    data_collector.load_wvs_indicators = _empty_loader(
        ["country", "year", "sci_trust", "tech_acceptance"]
    )
    data_collector.load_gii_scores = _empty_loader(["country", "year", "gii_score"])
    data_collector.load_ilostat_salary = _empty_loader(["country", "year", "relative_salary"])
    data_collector.collect_fred_data = _empty_loader(["country", "year", "cpi_inflation"])

    # v1 상수 주입 — 모듈 attribute 재할당(파일 미변경)
    preprocessor.FEATURE_GROUPS = PAPER_V1_FEATURE_GROUPS
    preprocessor.ALL_FEATURES = [f for feats in PAPER_V1_FEATURE_GROUPS.values() for f in feats]
    shap_analyzer.CATEGORY_MAP = PAPER_V1_CATEGORY_MAP
    scenario_simulator.POLICY_SCENARIOS = PAPER_V1_POLICY_SCENARIOS

    # 패치 후 sanity — v1 17 base features 가 실제로 주입됐는지 확인
    assert len(preprocessor.ALL_FEATURES) == 17, "v1 사양 주입 실패"


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="논문 v1 사양 재현 오케스트레이터 — XGBoost + SHAP (합성·20국·17피처·5시나리오)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/sc_paper_v1",
        help="출력 디렉토리 (기본: /tmp/sc_paper_v1) — 정본 ./results 미덮어쓰기.",
    )
    parser.add_argument(
        "--focus-country",
        type=str,
        default="KOR",
        help="집중 분석 국가 ISO3 (기본: KOR — 논문 51.18 비교 대상)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optuna 하이퍼파라미터 최적화 (기본: 사용 안 함, 논문도 기본 파라미터)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna 시도 횟수 (--optimize 사용 시)",
    )
    parser.add_argument(
        "--lag-years",
        type=int,
        nargs="+",
        default=config.LAG_YEARS,
        help="Time-lag 범위 (기본 [3 5 7])",
    )
    return parser.parse_args()


def print_banner() -> None:
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║  📜 v1 사양 재현 모드 (합성 데이터 100%)                           ║
║     20개국·2012–2023 · 17변수 · 5 시나리오 · 합성 100%             ║
║     정본 v2 미변경 — monkey-patch 로 v1 사양만 본 프로세스 주입    ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


# ════════════════════════════════════════════════════════════════
# 결과 저장 — main.py 의 save_results_json 과 형식 일치
# ════════════════════════════════════════════════════════════════
def save_results_json(
    cv_metrics: dict,
    shap_results: dict,
    sim_results: dict,
    output_dir: str,
) -> str:
    """핵심 수치를 JSON 으로 저장(main.py 와 동일 스키마).

    직렬화 로직은 각 스테이지 소유(serialize_*) — 여기서는 v1 고유 키(mode·spec)를
    generated_at 직후에 끼워 넣고 나머지 키 순서는 main.py 와 동일하게 조립한다.
    """
    output = {
        "generated_at": datetime.now().isoformat(),
        "mode": "paper_v1_replication",
        "spec": {
            "n_countries": 20,
            "year_range": [2012, 2023],
            "n_base_features": 17,
            "n_scenarios": 5,
            "synthetic_only": True,
        },
        "model_performance": serialize_cv_metrics(cv_metrics),
        **serialize_shap_results(shap_results),
        **serialize_sim_results(sim_results),
    }

    path = os.path.join(output_dir, "xai_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"   ✅ JSON 저장: {path}")
    return path


# ════════════════════════════════════════════════════════════════
# 논문 보고 vs 실측 대조표
# ════════════════════════════════════════════════════════════════
def _diff_str(reported: float, actual: float, unit: str = "") -> str:
    diff = actual - reported
    sign = "+" if diff >= 0 else ""
    return f"{actual:.3f}{unit}  (Δ {sign}{diff:.3f}{unit})"


def print_comparison(
    cv_metrics: dict,
    shap_results: dict,
    sim_results: dict,
) -> None:
    """논문 보고 수치 vs 본 실행 수치 대조 — 콘솔 최종 표."""
    print("\n" + "═" * 70)
    print(" 📊 논문 v1 보고 수치 vs 본 실행 (근사 재현 — 완전 일치 아님)")
    print("═" * 70)

    # 1) 모델 성능
    rmse = cv_metrics.get("rmse", float("nan"))
    r2 = cv_metrics.get("r2", float("nan"))
    print("\n[ 모델 성능 ]")
    print(f"  CV R²    | 논문 {PAPER_CV_R2:.3f}  | 실측 {_diff_str(PAPER_CV_R2, r2)}")
    print(f"  RMSE     | 논문 {PAPER_RMSE:.3f}  | 실측 {_diff_str(PAPER_RMSE, rmse)}")

    # 2) 카테고리 SHAP %
    print("\n[ 카테고리별 SHAP 기여도 (%) ]")
    cat_df = shap_results["category_contribution"]
    actual_cat = {row["category"]: float(row["weight_pct"]) for _, row in cat_df.iterrows()}
    for cat, paper_val in PAPER_CATEGORY_PCT.items():
        act = actual_cat.get(cat, 0.0)
        print(f"  {cat:<8s} | 논문 {paper_val:5.1f}% | 실측 {_diff_str(paper_val, act, '%')}")

    # 3) Top 5 피처
    print("\n[ Top 5 피처 ]")
    top5 = shap_results["global_importance"].head(5)
    print(f"  논문 순서: {' > '.join(PAPER_TOP5_FEATURES)}")
    print("  실측 순서:")
    for _, row in top5.iterrows():
        print(f"     {int(row['rank']):2d}. {row['feature']:<35s} ({row['category']}) {row['importance_pct']:.2f}%")

    # 4) 한국 기준점수
    print("\n[ 한국(KOR) 기준 혁신점수 ]")
    sm = sim_results["scenario_matrix"]
    if not sm.empty and "KOR" in sm.index and "baseline" in sm.columns:
        kor_base = float(sm.loc["KOR", "baseline"])
        print(
            f"  논문 {PAPER_KOR_BASELINE:.2f} | "
            f"실측 {_diff_str(PAPER_KOR_BASELINE, kor_base)}"
        )
    else:
        print("  ⚠ scenario_matrix 에 KOR baseline 없음")

    # 5) 시나리오 5개 delta%
    print("\n[ 한국 시나리오 delta_pct (%) ]")
    if not sm.empty and "KOR" in sm.index:
        for scen, paper_val in PAPER_KOR_SCENARIO_DELTA_PCT.items():
            act = float(sm.loc["KOR", scen]) if scen in sm.columns else 0.0
            print(f"  {scen:<22s} | 논문 {paper_val:+.2f}% | 실측 {_diff_str(paper_val, act, '%')}")
    else:
        print("  ⚠ scenario_matrix 에 KOR 행 없음")

    print("\n" + "═" * 70)
    print(
        " ℹ 합성 데이터는 `build_full_dataset(seed=42)` 진입 시 로컬 rng 를 새로 만들어\n"
        "    호출별 결정적이다(P1 재현성). 논문 보고치와의 차이는 사양 재현 한계(근사 재현)."
    )
    print("═" * 70)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main() -> dict:
    args = parse_args()
    print_banner()

    # v1 사양 주입(모듈 attribute 재할당 — 파일 미변경)
    _apply_paper_v1_patches()
    print("🔧 v1 monkey-patch 적용 완료 — FEATURE_GROUPS·CATEGORY_MAP·POLICY_SCENARIOS")
    print("   sci_culture 4 · hard_rd 5 · structure 5 · economic 3 (총 17 base features) · 5 시나리오")

    os.makedirs(args.output, exist_ok=True)
    start = time.time()

    # ── STEP 1: 데이터 수집 — 합성 100%
    print("\n" + "=" * 70)
    print("STEP 1 / 5  데이터 수집 (합성 · 20국 · 2012–2023)")
    print("=" * 70)
    countries = list(data_collector.COUNTRIES.keys())  # 정본 20국 그대로
    print(f"🌍 국가셋: paper_v1 · {len(countries)}개국")
    raw_df = build_full_dataset(
        countries=countries,
        use_worldbank=False,  # WB API 미사용 → _synthesize_wb_features 합성
        use_fred=False,  # FRED 미사용 (실데이터 로더 monkey-patch 로도 안전)
    )
    raw_df.to_csv(os.path.join(args.output, "raw_dataset.csv"), index=False)

    # ── STEP 2: 전처리
    print("\n" + "=" * 70)
    print("STEP 2 / 5  전처리 및 피처 엔지니어링 (v1 17변수)")
    print("=" * 70)
    processed_df, X, y, feat_names, final_pipe, X_raw = run_preprocessing(
        raw_df, target="innovation_score", lag_years=args.lag_years
    )
    processed_df.to_csv(os.path.join(args.output, "processed_dataset.csv"), index=False)

    # 전처리 누수 제거(이중 경로): CV 의 fold-내 fit 용 factory 주입(main.py 와 동일 규칙).
    base_cols = [c for c in X_raw.columns if not (c.startswith("idx_") or "_lag" in c)]
    rest_cols = [c for c in X_raw.columns if c.startswith("idx_") or "_lag" in c]

    # ── STEP 3: 모델 학습
    print("\n" + "=" * 70)
    print("STEP 3 / 5  XGBoost 모델 학습")
    print("=" * 70)
    model, cv_metrics = run_training(
        X, X_raw, y, processed_df,
        make_pipeline=lambda: build_leakage_pipeline(base_cols, rest_cols),
        optimize=args.optimize, n_trials=args.n_trials,
    )

    # ── STEP 4: SHAP 분석
    print("\n" + "=" * 70)
    print("STEP 4 / 5  SHAP 분석 (v1 4범주 매핑)")
    print("=" * 70)
    shap_results = run_shap_analysis(model, X, processed_df, focus_country=args.focus_country)

    # ── STEP 5: 시나리오 시뮬레이션 + 시각화
    print("\n" + "=" * 70)
    print("STEP 5 / 5  시나리오 시뮬레이션 (v1 5개) + 시각화")
    print("=" * 70)
    # scenario_simulator 는 X 가 이미 스케일 공간이라 4번째 인자를 쓰지 않는다(del scaler).
    sim_results = run_simulation(model, X, processed_df, final_pipe, shap_results)
    vis_paths = run_visualization(shap_results, sim_results, processed_df, X, output_dir=args.output)

    # ── 결과 저장
    print("\n💾 결과 저장 중...")
    json_path = save_results_json(cv_metrics, shap_results, sim_results, args.output)

    # ── 논문 대조표
    print_comparison(cv_metrics, shap_results, sim_results)

    elapsed = time.time() - start
    print(f"\n⏱  전체 소요 시간: {elapsed:.1f}초")
    print(f"📁 결과 저장 위치: {os.path.abspath(args.output)}")

    return {
        "model": model,
        "processed_df": processed_df,
        "X": X,
        "X_raw": X_raw,
        "y": y,
        "final_pipe": final_pipe,
        "shap_results": shap_results,
        "sim_results": sim_results,
        "cv_metrics": cv_metrics,
        "vis_paths": vis_paths,
        "json_path": json_path,
    }


if __name__ == "__main__":
    main()
