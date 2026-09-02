"""
visualizer.py
=============
분석 결과 시각화 모듈 (6종 그래프)

1. 카테고리별 SHAP 기여도 수평 막대
2. 글로벌 피처 중요도 Top 20 (Beeswarm 스타일)
3. 한국 병목 진단 Waterfall
4. 정책 시나리오 Heatmap (국가 × 시나리오)
5. Time-lag 상관관계 (과학문화 → 혁신 성과)
6. 한국 vs 선진국 갭 레이더
"""

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg") 는 import 시점 유지 — pyplot 첫 import 전에 백엔드를 고정해야
# 헤드리스 환경(서버/CI)에서 안전하다(지연 시 GUI 백엔드 초기화 부수효과 발생 가능).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe
import warnings

# ── 한글 폰트 설정
import matplotlib.font_manager as fm
import subprocess, os
import logging

# 모듈 로거 — 폴백·경고(⚠) 전용 (ADR-0007). 핸들러 미구성: logging.lastResort 가
# stderr 로 출력해 stdout(진행 리포트 print)/stderr(경고)가 분리된다.
logger = logging.getLogger(__name__)


def _setup_korean_font():
    """Ubuntu 환경 한글 폰트 자동 설정"""
    try:
        result = subprocess.run(["fc-list", ":lang=ko"], capture_output=True, text=True)
        fonts = result.stdout.strip().split("\n")
        for f in fonts:
            if "Noto" in f or "Malgun" in f or "나눔" in f or "Nanum" in f:
                path = f.split(":")[0].strip()
                if path.endswith(".ttf") or path.endswith(".otf"):
                    fm.fontManager.addfont(path)
                    prop = fm.FontProperties(fname=path)
                    plt.rcParams["font.family"] = prop.get_name()
                    return
    except Exception:
        pass
    plt.rcParams["font.family"] = "DejaVu Sans"
    logger.warning("   ⚠ 한글 폰트 미발견 – 영문 레이블 사용")


# ── 스타일 1회 적용 (idempotent) — import 부수효과 방지:
#    폰트 탐색(subprocess)·rcParams 변경은 module import 시점이 아니라
#    첫 플롯 호출 시점에 1회만 수행한다.
_STYLE_READY = False


def _ensure_style() -> None:
    """한글 폰트 + 공통 rcParams 를 1회만 적용한다(모듈 플래그 _STYLE_READY)."""
    global _STYLE_READY
    if _STYLE_READY:
        return
    _setup_korean_font()
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })
    _STYLE_READY = True

# 카테고리 색상 팔레트·영문 라벨·lag·레이더 기본국 — config.py 로 이동 (이름 바인딩 import)
from config import (
    CATEGORY_COLORS as CAT_COLORS,
)
from config import (
    CATEGORY_LABELS_EN,
    LAG_YEARS,
    RADAR_DEFAULT_COUNTRIES,
)


# ──────────────────────────────────────────────
# 1. 카테고리별 SHAP 기여도
# ──────────────────────────────────────────────
def plot_category_contribution(cat_contrib: pd.DataFrame,
                                save_path: str = "/tmp/01_category_shap.png"):
    _ensure_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))

    cats   = [CATEGORY_LABELS_EN.get(c, c) for c in cat_contrib["category"].tolist()]
    vals   = cat_contrib["weight_pct"].tolist()
    colors = [CAT_COLORS.get(c, "#8172B3") for c in cat_contrib["category"].tolist()]

    bars = ax.barh(cats, vals, color=colors, height=0.55, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10.5, fontweight="bold")

    ax.set_xlabel("SHAP Contribution Weight to Innovation Score (%)", fontsize=10)
    ax.set_title("Category-level Innovation Contribution\n(XGBoost + SHAP Analysis)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"   저장: {save_path}")


# ──────────────────────────────────────────────
# 2. 글로벌 피처 중요도 (Top 20)
# ──────────────────────────────────────────────
def plot_global_importance(global_imp: pd.DataFrame,
                            shap_values: np.ndarray,
                            X: pd.DataFrame,
                            top_n: int = 20,
                            save_path: str = "/tmp/02_global_importance.png"):
    _ensure_style()
    top_feats = global_imp.head(top_n)
    fig, ax = plt.subplots(figsize=(11, 7))

    colors = [CAT_COLORS.get(c, "#8172B3") for c in top_feats["category"]]
    y_pos  = range(len(top_feats) - 1, -1, -1)

    ax.barh(list(y_pos), top_feats["importance_pct"].values,
            color=colors, height=0.6, edgecolor="white", linewidth=0.8, alpha=0.88)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_feats["feature"].values, fontsize=8.5)
    ax.set_xlabel("Mean |SHAP| Contribution (%)", fontsize=10)
    ax.set_title(f"Global Feature Importance Top {top_n}\n(XGBoost + SHAP TreeExplainer)",
                 fontsize=12, fontweight="bold", pad=14)

    # 범례
    patches = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLORS.items()
               if k in top_feats["category"].values]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   저장: {save_path}")


# ──────────────────────────────────────────────
# 3. 한국 병목 진단 (Waterfall 스타일)
# ──────────────────────────────────────────────
def plot_korea_waterfall(country_profile: pd.DataFrame,
                          base_value: float,
                          top_n: int = 12,
                          save_path: str = "/tmp/03_korea_waterfall.png"):
    _ensure_style()
    if country_profile.empty:
        logger.warning("   ⚠ 한국 데이터 없음 – 스킵")
        return

    top = country_profile.head(top_n).copy()
    feats = top["feature"].tolist()
    vals  = top["shap_value"].tolist()
    colors = ["#4C72B0" if v > 0 else "#C44E52" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(feats) - 1, -1, -1)
    ax.barh(list(y_pos), vals, color=colors, height=0.6,
            edgecolor="white", linewidth=0.8, alpha=0.9)

    for pos, (feat, val) in zip(y_pos, zip(feats, vals)):
        lbl = f"{val:+.3f}"
        x   = val + (0.002 if val > 0 else -0.002)
        ha  = "left" if val > 0 else "right"
        ax.text(x, pos, lbl, va="center", ha=ha, fontsize=8.5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(feats, fontsize=8.5)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("SHAP Value  (+ contributes  /  - bottleneck)", fontsize=10)
    ax.set_title("Korea (KOR): Innovation Bottleneck Diagnosis\nSHAP Local Explanation – Most Recent Year",
                 fontsize=12, fontweight="bold", pad=14)

    pos_p = mpatches.Patch(color="#4C72B0", label="▲ 기여 (긍정 요인)")
    neg_p = mpatches.Patch(color="#C44E52", label="▼ 저해 (병목 요인)")
    ax.legend(handles=[pos_p, neg_p], loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   저장: {save_path}")


# ──────────────────────────────────────────────
# 4. 정책 시나리오 Heatmap
# ──────────────────────────────────────────────
def plot_scenario_heatmap(scenario_matrix: pd.DataFrame,
                           save_path: str = "/tmp/04_scenario_heatmap.png"):
    _ensure_style()
    if scenario_matrix.empty:
        return

    # baseline 컬럼 제거
    plot_df = scenario_matrix.drop(columns=["baseline"], errors="ignore")
    if plot_df.empty:
        return

    # 컬럼명 간소화
    col_renames = {
        "S1_과학문화_강화": "S1\nSci-Culture",
        "S2_보상_개선":    "S2\nCompensation",
        "S3_과제기간_연장": "S3\nProject Dur.",
        "S4_종합_혁신패키지":"S4\nPackage",
        "S5_물적규모_확대": "S5\nR&D Scale",
    }
    plot_df = plot_df.rename(columns=col_renames)

    fig, ax = plt.subplots(figsize=(11, 6))
    vmin, vmax = plot_df.values.min(), plot_df.values.max()
    if abs(vmax - vmin) < 0.01:
        vmin, vmax = -1.0, 1.0
    vcenter = 0 if vmin < 0 < vmax else (vmin + vmax) / 2
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    im = ax.imshow(plot_df.values, aspect="auto", cmap="RdYlGn", norm=norm)

    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, fontsize=9)
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=9)

    # 셀 수치 표기
    for i in range(len(plot_df.index)):
        for j in range(len(plot_df.columns)):
            val = plot_df.values[i, j]
            txt = f"{val:+.1f}%"
            color = "black" if abs(val) < plot_df.values.max() * 0.6 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Innovation Score Change (%)", fontsize=9)
    ax.set_title("What-if Policy Scenario Simulation\nCountry x Scenario Matrix – Innovation Score Change (%)",
                 fontsize=12, fontweight="bold", pad=14)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   저장: {save_path}")


# ──────────────────────────────────────────────
# 5. Time-lag 상관관계 분석
# ──────────────────────────────────────────────
def plot_timelag_correlation(processed_df: pd.DataFrame,
                              lag_feature: str = "sci_literacy",
                              outcome: str = "innovation_score",
                              lag_years: list | None = None,
                              save_path: str = "/tmp/05_timelag.png"):
    _ensure_style()
    lag_years = lag_years or [0, *LAG_YEARS]
    corrs = []

    df_sorted = processed_df.sort_values(["country", "year"])

    for lag in lag_years:
        lag_col = f"{lag_feature}_lag{lag}" if lag > 0 else lag_feature
        if lag_col not in df_sorted.columns:
            if lag > 0:
                continue
            else:
                lag_col = lag_feature

        valid = df_sorted[[lag_col, outcome]].dropna()
        if len(valid) < 10:
            continue
        corr = valid.corr().iloc[0, 1]
        corrs.append({"lag": lag, "correlation": corr})

    if not corrs:
        return
    corr_df = pd.DataFrame(corrs)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#C44E52" if c < 0 else "#4C72B0" for c in corr_df["correlation"]]
    bars = ax.bar(corr_df["lag"].astype(str) + "yr",
                  corr_df["correlation"], color=colors,
                  width=0.5, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, corr_df["correlation"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * np.sign(val),
                f"{val:.3f}", ha="center", va="bottom" if val > 0 else "top",
                fontsize=9.5, fontweight="bold")

    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xlabel("Time Lag (years)", fontsize=10)
    ax.set_ylabel("Pearson Correlation", fontsize=10)
    ax.set_title(f"Time-Lag Analysis: {lag_feature} → {outcome}\n(Correlation by Lag Year)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(-0.2, 1.1)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   저장: {save_path}")


# ──────────────────────────────────────────────
# 6. 한국 vs 선진국 레이더 차트
# ──────────────────────────────────────────────
def plot_radar_comparison(processed_df: pd.DataFrame,
                           countries: list | None = None,
                           save_path: str = "/tmp/06_radar.png"):
    from math import pi
    _ensure_style()
    countries = countries or RADAR_DEFAULT_COUNTRIES

    idx_cols = [c for c in processed_df.columns if c.startswith("idx_")]
    if not idx_cols:
        return

    labels = {
        "idx_sci_culture": "Sci-Culture",
        "idx_hard_rd":     "R&D Scale",
        "idx_structure":   "Structure",
        "idx_economic":    "Compensation",
    }
    idx_cols = [c for c in idx_cols if c in labels]
    if not idx_cols:
        return

    radar_labels = [labels[c] for c in idx_cols]
    n = len(idx_cols)
    angles = [pi / 2 + 2 * pi * i / n for i in range(n)] + [pi / 2]

    # 각 국가의 최근 연도 복합 지수 평균
    country_data = {}
    for cc in countries:
        sub = processed_df[processed_df["country"] == cc][idx_cols]
        if sub.empty:
            continue
        country_data[cc] = sub.tail(3).mean().tolist()   # 최근 3년 평균

    if not country_data:
        return

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    pal = ["#C44E52", "#4C72B0", "#DD8452", "#55A868", "#8172B3"]

    for (cc, vals), color in zip(country_data.items(), pal):
        v = vals + [vals[0]]
        ax.plot(angles, v, color=color, linewidth=2,
                label=cc, marker="o", markersize=5)
        ax.fill(angles, v, color=color, alpha=0.10)

    ax.set_theta_offset(0)
    ax.set_rlabel_position(45)
    ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], radar_labels, fontsize=10.5)
    ax.set_ylim(0, 1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title("Country Radar: 4-Index Comparison\n(3-Year Average)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   저장: {save_path}")


# ──────────────────────────────────────────────
# 통합 시각화 실행
# ──────────────────────────────────────────────
def run_visualization(shap_results: dict,
                       sim_results: dict,
                       processed_df: pd.DataFrame,
                       X: pd.DataFrame,
                       output_dir: str = "/tmp") -> list:
    """
    모든 시각화 생성 후 저장 경로 목록 반환
    """
    # import 부수효과 방지: 경고 억제·폰트/스타일 적용은 실행 진입 시점에만.
    warnings.filterwarnings("ignore")
    _ensure_style()

    import os
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    print("🎨 시각화 생성 중...")

    p = f"{output_dir}/01_category_shap.png"
    plot_category_contribution(shap_results["category_contribution"], p)
    paths.append(p)

    p = f"{output_dir}/02_global_importance.png"
    plot_global_importance(shap_results["global_importance"],
                           shap_results["shap_values"], X, save_path=p)
    paths.append(p)

    p = f"{output_dir}/03_korea_waterfall.png"
    plot_korea_waterfall(shap_results["country_profile"],
                         shap_results["base_value"], save_path=p)
    paths.append(p)

    p = f"{output_dir}/04_scenario_heatmap.png"
    plot_scenario_heatmap(sim_results["scenario_matrix"], save_path=p)
    paths.append(p)

    p = f"{output_dir}/05_timelag.png"
    plot_timelag_correlation(processed_df, save_path=p)
    paths.append(p)

    p = f"{output_dir}/06_radar.png"
    plot_radar_comparison(processed_df, save_path=p)
    paths.append(p)

    print(f"✅ 시각화 완료 ({len(paths)}개 파일)")
    return paths
