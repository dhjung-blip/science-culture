"""
main.py
=======
과학문화 XAI 분석 파이프라인 메인 실행 파일

실행:
    python main.py                         # 기본 (WB API + 합성 데이터)
    python main.py --no-wb                 # WB API 미사용 (합성만)
    python main.py --optimize              # Optuna 하이퍼파라미터 최적화
    python main.py --target competitiveness  # 종속변수 변경
    python main.py --country-set pisa-gii --target patents_per_m_log
                                           # 탈순환 타깃(findings §7.7, 실 WB 모드 전용)
    python main.py --output ./results      # 출력 디렉토리 지정
"""

import argparse
import json
import os
import time
from datetime import datetime

# ── 로컬 모듈
import config
import data_collector
from data_collector import build_full_dataset
from model_trainer import run_training, serialize_cv_metrics
from preprocessor import build_leakage_pipeline, run_preprocessing
from scenario_simulator import run_simulation, serialize_sim_results
from shap_analyzer import run_shap_analysis, serialize_shap_results
from visualizer import run_visualization


def parse_args():
    parser = argparse.ArgumentParser(
        description="과학문화 XAI (XGBoost + SHAP) 분석 파이프라인"
    )
    parser.add_argument("--no-wb", action="store_true",
                        help="World Bank API 미사용 (합성 데이터만)")
    parser.add_argument("--country-set", type=str, default="default",
                        choices=["default", "pisa-gii", "gii"],
                        help="분석 국가셋 (default=20국, pisa-gii=PISA∩GII, gii=GII 전체)")
    parser.add_argument("--optimize", action="store_true",
                        help="Optuna 하이퍼파라미터 최적화 수행")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna 시도 횟수 (기본: 30)")
    parser.add_argument("--target", type=str, default="innovation_score",
                        choices=config.OUTCOME_COLS,
                        help="종속변수 (기본: innovation_score · "
                             "articles_per_m_log/patents_per_m_log 는 탈순환 타깃(findings §7.7) — "
                             "실 WB 모드 전용)")
    parser.add_argument("--focus-country", type=str, default="KOR",
                        help="집중 분석 국가 ISO3 코드 (기본: KOR)")
    parser.add_argument("--output", type=str, default="./results",
                        help="출력 디렉토리 (기본: ./results)")
    parser.add_argument("--lag-years", type=int, nargs="+", default=config.LAG_YEARS,
                        help="Time-lag 범위 (기본: 3 5 7)")
    args = parser.parse_args()
    # 빠른 가드: 탈순환 타깃은 WB population 파생 → --no-wb(합성 폴백) 경로엔 컬럼 부재.
    if args.no_wb and args.target in ("articles_per_m_log", "patents_per_m_log"):
        parser.error(f"--target {args.target}: 탈순환 타깃은 실 WB 모드 필요 (--no-wb 비호환)")
    return args


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     과학문화 × 국가혁신 XAI 분석 파이프라인 v1.0            ║
║     XGBoost + SHAP · Time-lag · What-if Simulation          ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def save_results_json(cv_metrics: dict,
                       shap_results: dict,
                       sim_results: dict,
                       output_dir: str):
    """핵심 수치 결과를 JSON으로 저장.

    직렬화 로직은 각 스테이지 소유(serialize_*) — main 은 키 순서만 조립한다:
    generated_at → model_performance → category_contribution → top10_features
    → korea_bottlenecks → scenario_matrix → korea_gap_analysis.
    """
    output = {
        "generated_at": datetime.now().isoformat(),
        "model_performance": serialize_cv_metrics(cv_metrics),
        **serialize_shap_results(shap_results),
        **serialize_sim_results(sim_results),
    }

    path = os.path.join(output_dir, "xai_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"   ✅ JSON 저장: {path}")
    return path


def print_summary(cv_metrics: dict, shap_results: dict, sim_results: dict):
    """콘솔 최종 요약 출력"""
    print("\n" + "═" * 62)
    print(" 📋 분석 결과 요약")
    print("═" * 62)

    print("\n[ 모델 성능 ]")
    print(f"  교차검증 RMSE : {cv_metrics['rmse']:.4f} ± {cv_metrics['rmse_std']:.4f}")
    print(f"  교차검증 R²   : {cv_metrics['r2']:.4f} ± {cv_metrics['r2_std']:.4f}")
    print(f"  학습 데이터 R²: {cv_metrics.get('train_r2', 0):.4f}")

    print("\n[ 카테고리별 혁신 성과 기여도 (SHAP) ]")
    cat = shap_results["category_contribution"]
    for _, row in cat.iterrows():
        bar = "█" * int(row["weight_pct"] / 2)
        print(f"  {row['category']:10s} │ {bar:<25} │ {row['weight_pct']:.1f}%")

    print("\n[ 상위 피처 Top 5 ]")
    top5 = shap_results["global_importance"].head(5)
    for _, row in top5.iterrows():
        print(f"  {row['rank']:2d}. {row['feature']:<35} ({row['category']}) "
              f"{row['importance_pct']:.2f}%")

    print("\n[ 정책 시나리오 – 한국 혁신 성과 변화 예측 ]")
    sm = sim_results["scenario_matrix"]
    if not sm.empty and "KOR" in sm.index:
        baseline_val = sm.loc["KOR", "baseline"] if "baseline" in sm.columns else "N/A"
        print(f"  기준 혁신 점수 (KOR): {baseline_val:.2f}")
        kor_row = sm.loc["KOR"].drop("baseline", errors="ignore")
        for scen, val in kor_row.items():
            arrow = "▲" if val > 0 else ("▼" if val < 0 else "━")
            print(f"  {arrow} {scen}: {val:+.2f}%")

    print("\n" + "═" * 62)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    print_banner()

    os.makedirs(args.output, exist_ok=True)
    start = time.time()

    # ── STEP 1: 데이터 수집
    print("=" * 62)
    print("STEP 1 / 5  데이터 수집")
    print("=" * 62)
    countries = data_collector.resolve_country_set(args.country_set, data_collector.YEARS)
    print(f"🌍 국가셋: {args.country_set} · {len(countries)}개국")
    # cpi_inflation 은 모델 피처가 아니므로 default 외 대규모 국가셋에선 FRED 호출을 생략.
    # --no-wb 는 오프라인 의미론: WB 뿐 아니라 FRED 네트워크 호출도 전부 생략한다
    # (cpi_inflation 은 비피처 → 결과 JSON 불변).
    raw_df = build_full_dataset(
        countries=countries,
        use_worldbank=not args.no_wb,
        use_fred=(args.country_set == "default") and not args.no_wb,
    )
    raw_df.to_csv(os.path.join(args.output, "raw_dataset.csv"), index=False)

    # ── STEP 2: 전처리
    print("\n" + "=" * 62)
    print("STEP 2 / 5  전처리 및 피처 엔지니어링")
    print("=" * 62)
    processed_df, X, y, feat_names, final_pipe, X_raw = run_preprocessing(
        raw_df, target=args.target, lag_years=args.lag_years
    )
    processed_df.to_csv(os.path.join(args.output, "processed_dataset.csv"), index=False)

    # 전처리 누수 제거: CV 가 fold-train 에만 누수성 변환을 fit 하도록 factory 를 주입한다.
    # model_trainer 는 preprocessor 를 import 하지 않으므로(스테이지 독립) main 이 콜러블로 넘긴다.
    base_cols = [c for c in X_raw.columns if not (c.startswith("idx_") or "_lag" in c)]
    rest_cols = [c for c in X_raw.columns if c.startswith("idx_") or "_lag" in c]

    # ── STEP 3: 모델 학습
    print("\n" + "=" * 62)
    print("STEP 3 / 5  XGBoost 모델 학습")
    print("=" * 62)
    model, cv_metrics = run_training(
        X, X_raw, y, processed_df,
        make_pipeline=lambda: build_leakage_pipeline(base_cols, rest_cols),
        optimize=args.optimize, n_trials=args.n_trials
    )

    # ── STEP 4: SHAP 분석
    print("\n" + "=" * 62)
    print("STEP 4 / 5  SHAP 분석")
    print("=" * 62)
    shap_results = run_shap_analysis(model, X, processed_df,
                                      focus_country=args.focus_country)

    # ── STEP 5: 시나리오 시뮬레이션
    print("\n" + "=" * 62)
    print("STEP 5 / 5  시나리오 시뮬레이션 + 시각화")
    print("=" * 62)
    # scenario_simulator 는 X 가 이미 스케일 공간이라 가정하고 σ-shift 하므로 4번째 인자를
    # 실제로 쓰지 않는다(del scaler). 호환을 위해 최종 파이프라인을 그 자리에 넘긴다.
    sim_results = run_simulation(model, X, processed_df, final_pipe, shap_results)

    # ── 시각화
    vis_paths = run_visualization(shap_results, sim_results,
                                   processed_df, X, output_dir=args.output)

    # ── 결과 저장
    print("\n💾 결과 저장 중...")
    json_path = save_results_json(cv_metrics, shap_results, sim_results, args.output)

    # ── 요약 출력
    print_summary(cv_metrics, shap_results, sim_results)

    elapsed = time.time() - start
    print(f"\n⏱  전체 소요 시간: {elapsed:.1f}초")
    print(f"📁 결과 저장 위치: {os.path.abspath(args.output)}")

    return {
        "model": model, "processed_df": processed_df,
        "X": X, "X_raw": X_raw, "y": y, "final_pipe": final_pipe,
        "shap_results": shap_results, "sim_results": sim_results,
        "cv_metrics": cv_metrics, "vis_paths": vis_paths,
        "json_path": json_path,
    }


if __name__ == "__main__":
    results = main()
