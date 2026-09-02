# -*- coding: utf-8 -*-
"""논문 그림 6종 — 한글 라벨 버전. /tmp/figures_kr/. read-only."""
import warnings
warnings.filterwarnings("ignore")
import sys, os
import os as _os
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, _os.environ.get("SC_CODE_DIR") or str(_Path(__file__).resolve().parents[1] / "code"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/tmp/figures_kr"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "font.family": "AppleGothic",
                     "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold"})
CUL, HARD, OPS, ECON = "#7c3aed", "#0284c7", "#0d9488", "#c2740a"
ACC, POS, NEG = "#2563eb", "#15a24a", "#dc2626"

# ── 그림 1. 매개 경로도 ──
fig, ax = plt.subplots(figsize=(8.4, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc="white", ec=color, lw=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=11, weight="bold", color=color)
box(0.3, 2, 2.4, 1, "과학소양\n(PISA)", CUL)
box(3.8, 3.2, 2.4, 1, "연구원 밀도\n(M)", HARD)
box(7.3, 2, 2.4, 1, "혁신\n(GII)", ACC)
def arrow(x1, y1, x2, y2, label, col="#444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=col))
    ax.text((x1+x2)/2, (y1+y2)/2 + 0.25, label, ha="center", fontsize=10, color=col, weight="bold")
arrow(2.7, 2.8, 3.8, 3.5, "a = +0.750***", POS)
arrow(6.2, 3.5, 7.3, 2.8, "b = +5.151***", POS)
arrow(2.7, 2.3, 7.3, 2.3, "c' = +3.532*** (직접효과)", "#666")
ax.text(5, 0.7, "총효과 c = +7.395***   |   간접효과(a·b) = +3.863   |   매개비율 52.2%  [부트스트랩 95% CI 32–67%]",
        ha="center", fontsize=9.5, style="italic", color="#333")
ax.set_title("그림 1. 인적자본 파이프라인: 과학소양 → 연구원 밀도 → 혁신", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_mediation.png", bbox_inches="tight"); plt.close()

# ── 그림 2. 탈순환 3구도 ──
fig, ax = plt.subplots(figsize=(7, 3.8))
targets = ["특허\n/백만명", "논문\n/백만명", "GII 산출\n서브지수"]; betas = [0.451, 0.543, 0.325]; within = [0.349, None, None]
x = np.arange(len(targets))
ax.bar(x, betas, 0.5, color=[CUL, CUL, HARD], alpha=0.85)
for i, (b, w) in enumerate(zip(betas, within)):
    ax.text(i, b + 0.015, f"+{b}***", ha="center", fontsize=10, weight="bold")
    if w: ax.text(i, 0.05, f"국가내\n+{w}***", ha="center", fontsize=8.5, color=POS, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(targets); ax.set_ylabel("표준화 β (R&D·GDP 통제)")
ax.set_ylim(0, 0.62); ax.axhline(0, color="#999", lw=0.8)
ax.set_title("그림 2. 탈순환: 과학소양은 비순환 3개 타깃 모두에서 유의")
ax.text(0.5, -0.14, "모두 p<0.001. 특허 타깃은 국가 내(within) 효과도 유의.",
        transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#555")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_decircle.png", bbox_inches="tight"); plt.close()

# ── 그림 3. 예측 vs 실제 ──
import data_collector as d, preprocessor as ppp, model_trainer as mt
import xgboost as xgb
from sklearn.model_selection import GroupKFold
CC = d.resolve_country_set("pisa-gii")
raw = d.build_full_dataset(countries=CC, years=d.YEARS, use_worldbank=True, use_fred=False, seed=42)
proc, X, y, feats, pipe, Xr = ppp.run_preprocessing(raw, target="innovation_score")
groups = proc.loc[y.index, "country"].values
xp = dict(mt.DEFAULT_XGB_PARAMS) if hasattr(mt, "DEFAULT_XGB_PARAMS") else {}
for _k in ("n_estimators","random_state","verbosity","early_stopping_rounds","n_jobs"): xp.pop(_k, None)
oof = np.full(len(y), np.nan)
for tr, va in GroupKFold(5).split(X, y, groups):
    m = xgb.XGBRegressor(n_estimators=300, random_state=42, verbosity=0, **xp)
    m.fit(X.iloc[tr], y.iloc[tr]); oof[va] = m.predict(X.iloc[va])
agg = pd.DataFrame({"c": groups, "a": y.values, "p": oof}).groupby("c").agg(
    actual=("a","mean"), pred=("p","mean")).reset_index()
fig, ax = plt.subplots(figsize=(5.6, 5.4))
ax.scatter(agg["actual"], agg["pred"], s=38, color=ACC, alpha=0.6, edgecolor="white", lw=0.5)
lo, hi = 20, 70; ax.plot([lo, hi], [lo, hi], "--", color="#999", lw=1.2, label="완벽 예측 (y=x)")
for _, r in agg.iterrows():
    if r["c"] in ["KOR","USA","SWE","CHN","FIN","DEU","SGP","IDN"]:
        ax.annotate(r["c"], (r["actual"], r["pred"]), fontsize=8, xytext=(3,3), textcoords="offset points")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("실제 GII (국가 평균)")
ax.set_ylabel("예측 GII (out-of-fold)"); ax.legend(loc="upper left", fontsize=9)
ax.set_title("그림 3. 예측 vs 실제 — 선진국 압축")
ax.text(0.97, 0.04, "상위국(SWE, FIN) 과소예측 →\n모델이 선두를 ~55로 압축",
        transform=ax.transAxes, ha="right", fontsize=8.5, style="italic", color="#555")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_pred_vs_actual.png", bbox_inches="tight"); plt.close()

# ── 그림 4. SHAP 카테고리 ──
fig, ax = plt.subplots(figsize=(8, 4))
cats = ["과학문화", "물적규모", "운영구조", "보상수준"]
data = {"20개국": [20.9, 13.6, 25.0, 40.5], "44개국(본분석)": [28.6, 50.6, 7.7, 13.0],
        "149개국": [10.6, 55.9, 17.3, 16.2], "44개국 통제": [38.4, 28.8, 9.2, 23.6]}
x = np.arange(len(cats)); w = 0.2; colors = ["#c7b3f0", "#7c3aed", "#b3d9f0", "#0284c7"]
for i, (lab, vals) in enumerate(data.items()):
    ax.bar(x + (i-1.5)*w, vals, w, label=lab, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("SHAP 기여도 (%)")
ax.set_title("그림 4. 카테고리 기여는 표본에 의존 (통제 후 과학문화 1위)")
ax.legend(fontsize=8.5, ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_shap_categories.png", bbox_inches="tight"); plt.close()

# ── 그림 5. R²–RMSE 착시 ──
fig, ax1 = plt.subplots(figsize=(6.6, 4))
samples = ["20개국", "44개국", "149개국"]; r2 = [-0.305, 0.770, 0.850]; rmse = [4.69, 4.08, 4.70]
x = np.arange(len(samples))
ax1.bar(x, r2, 0.45, color=ACC, alpha=0.8); ax1.set_ylabel("교차검증 R²", color=ACC); ax1.axhline(0, color="#999", lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(samples); ax1.set_ylim(-0.5, 1)
for i, v in enumerate(r2): ax1.text(i, v + (0.03 if v>0 else -0.08), f"{v:+.2f}", ha="center", fontsize=10, weight="bold")
ax2 = ax1.twinx(); ax2.plot(x, rmse, "o-", color=NEG, lw=2, ms=8); ax2.set_ylabel("RMSE (GII 점)", color=NEG); ax2.set_ylim(3.5, 5.2)
for i, v in enumerate(rmse): ax2.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=9, color=NEG)
ax1.set_title("그림 5. R²–RMSE 착시: RMSE는 ~일정, R²는 표본 분산으로 부풀려짐", fontsize=10.5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_r2_rmse.png", bbox_inches="tight"); plt.close()

# ── 그림 6. 모델 비교 ──
fig, ax = plt.subplots(figsize=(6.8, 3.8))
models = ["XGBoost", "ElasticNet", "랜덤포레스트", "Lasso", "Ridge", "선형(OLS)"]
cvr2 = [0.759, 0.725, 0.707, 0.697, 0.668, 0.640]; sd = [0.094, 0.124, 0.099, 0.150, 0.169, 0.190]
cols = [POS] + ["#9aa7b8"]*5
ax.barh(range(len(models))[::-1], cvr2, xerr=sd, color=cols, alpha=0.85, capsize=3)
for i, v in enumerate(cvr2): ax.text(v + 0.02, len(models)-1-i, f"{v:.3f}", va="center", fontsize=9.5)
ax.set_yticks(range(len(models))[::-1]); ax.set_yticklabels(models)
ax.set_xlabel("교차검증 R² (평균±SD, GroupKFold)"); ax.set_xlim(0, 0.95)
ax.set_title("그림 6. 모델 비교 — XGBoost 최고이나 선형도 근접 (한계는 모델이 아닌 데이터)", fontsize=10.5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_model_comparison.png", bbox_inches="tight"); plt.close()

print("### 한글 그림 6종 생성 완료:")
for f in sorted(os.listdir(OUT)): print(f"  {OUT}/{f}")
