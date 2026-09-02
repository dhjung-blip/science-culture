# Reproducibility guide

Single reference for reproducing the data and results of the manuscript. Column-level
provenance is in `DATA_DICTIONARY.md`; the release-time column removal is documented in
`COLUMN_REMOVAL_LOG.md`.

---

## 1. Environment

| Item | Value |
|---|---|
| Runtime | Python 3.12, pinned with `uv` and `uv.lock` (≥ 3.10 works) |
| Core libraries | xgboost 3.x, scikit-learn, shap, statsmodels, pandas, numpy |
| Seed | **42** on every path. `build_full_dataset(seed=42)` and XGBoost `random_state=42`; the golden test proves two runs are bit-identical |
| OS dependency | macOS: `brew install libomp` (Apple Silicon: `arch -arm64`). Linux: `libgomp1` |
| Figures | matplotlib `Agg` backend |

```bash
uv sync
MPLBACKEND=Agg uv run --no-sync python code/main.py --country-set pisa-gii
```

If OpenMP loading fails for sklearn or xgboost on macOS, prefix with:

```bash
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs"
```

## 2. What is frozen and what is regenerated

Two kinds of reproduction are possible and they answer different questions.

**Frozen-data reproduction (recommended, no network access).** Every script under
`analysis/` reads the frozen CSVs in `analysis/canonical/` and reproduces the published
numbers exactly. Each script prints the SHA-256 prefix of its inputs, so a mismatch is
visible immediately. This is the path a reviewer should use.

**Full-pipeline reproduction (requires source access).** `code/main.py` rebuilds the dataset
from the upstream sources. It regenerates five columns that the released frozen files omit
because no reported specification uses them; `tools/strip_legacy_columns.py` performs that
removal and verifies the composites are unchanged, so the difference between a fresh run and
the released file is fully accounted for:

```bash
uv run --no-sync python tools/strip_legacy_columns.py \
    /path/to/pipeline_output.csv analysis/canonical/processed_dataset.csv
```

## 3. Source access (`.env`, not committed)

No API keys are required; all sources are public. Downloaded file paths are supplied via
`.env`:

| Variable | Source |
|---|---|
| `PISA_DATA_PATH` | OECD PISA CSV, 2000–2022 combined |
| `WVS_WAVE6_PATH`, `WVS_WAVE7_PATH` | World Values Survey wave files |
| `GII_DATA_PATH` | WIPO GII long-format TSV |
| `GII_DIB_XLSX` | Mendeley GII compilation `gii_dataset_DIB.xlsx` (CC BY 4.0), used only by `gii_output_experiment.py` |

If a path is unset the collector skips that source gracefully. **World Values Survey
microdata are not redistributed here** under the survey programme's access terms; only the
derived national aggregates appear in the released files. Every other column rebuilds without
registration.

Data vintages follow the manuscript's data-availability statement. Scripts that read
`analysis/canonical/` reproduce independently of remote API state.

## 4. Script map

### 4.1 Manuscript tables and figures

| Script | Produces |
|---|---|
| `analysis/paper_tables_T2_T4.py` | Descriptive statistics; model-class comparison across six classes (Table 3a) |
| `analysis/paper_tableB_stability.py` | SHAP category stability over 20 seeds (Table 4); mediation bootstrap CIs; multi-seed CV |
| `analysis/paper_tables_A_all.py` | Supporting tables: SHAP top-10, leave-one-out marginal contribution, VIF, panel fixed effects, circularity correlations, within decomposition, per-country prediction, lags |
| `analysis/paper_figures.py` | Figures 1–4, English labels |
| `analysis/paper_figures_kr.py` | Figures 1–4, Korean labels |

### 4.2 De-circularization and the outcome hierarchy

| Script | Produces |
|---|---|
| `analysis/decircle_experiment.py` | Replaces the GII composite target with non-composite direct outputs (Table 2, tier a) |
| `analysis/gii_output_experiment.py` | GII Innovation Output Sub-Index target; manufacturing value-added discriminant check (Section 4.5) |
| `analysis/canonical/pooled_beta_canonical.py` | Fixes and reproduces the headline pooled β (Table 6) |
| `analysis/canonical/direct_output_robustness.py` | Oster δ, LOCO, and wild cluster bootstrap on the direct-output targets (Section 4.6) |

### 4.3 Source-separated corroborations

| Script | Produces |
|---|---|
| `analysis/outcome_extension_pct.py` | WIPO PCT applications by origin, via OECD SDMX |
| `analysis/outcome_extension_openalex.py` | OpenAlex top-10% cited publications |
| `analysis/outcome_extension_merge.py` | Merges external outcomes into the frozen panel and builds the five transformations (Table 7) |
| `analysis/outcome_extension_models.py` | Regression specifications: country-mean primary, pooled-clustered secondary |
| `analysis/run_external_outcome_analysis.py` | Runs the above end to end |
| `analysis/external_outcomes_template.py` | Ingestion contract for `data/external/*.csv` |
| `analysis/canonical/external_reference_tfp.py` | Penn World Table TFP, the fully independent macro check (Table 6, productivity rows) |

### 4.4 Sensitivity and identification boundary

| Script | Produces |
|---|---|
| `analysis/oster_sensitivity.py`, `analysis/canonical/oster_confounder_only.py` | Oster (2019) δ and β* at δ = 1 |
| `analysis/canonical/oster_rmax_curve.py` | R²max sensitivity curve and the critical R²max at which δ falls to 1 (Table 6a) |
| `analysis/wild_cluster_bootstrap.py` | Wild cluster bootstrap-t, G = 44 |
| `analysis/canonical/loco_jackknife.py` | Leave-one-country-out jackknife |
| `analysis/canonical/robustness_fdr_mediation_region.py` | BH-FDR correction, direct-output mediation CIs, leave-region, between-only effective sample (Tables 3, 10) |
| `analysis/canonical/robustness_shap_allocation_anchor_rho.py` | SHAP allocation robustness, PISA anchor-year within re-estimation, mediation ρ sensitivity |
| `analysis/patent_within_denominator_recheck.py` | Within-β after removing shared population-denominator confounding (Section 4.9) |
| `analysis/effective_sample_audit.py` | Observed-anchor versus interpolated shares; between:within variance ratio (Table 3, Section 3.2) |

### 4.5 Construct diagnostics

| Script | Produces |
|---|---|
| `analysis/sci_culture_axis_split.py` | Cognitive versus attitudinal axis SHAP split on the GII target (Table 5) |
| `analysis/canonical/direct_output_axis_split.py` | Same split on the de-circularized direct-output targets |

### 4.6 Temporal precedence

| Script | Produces |
|---|---|
| `analysis/pisa_cohort_lag.py` | Cohort-lag forward test using observed PISA anchors only (Section 4.8) |
| `analysis/future_pisa_placebo.py` | Future-PISA placebo (Table 8) |
| `analysis/placebo_robustness.py` | Sample-matched forward versus placebo, same window and country set (Table B1) |
| `analysis/lagged_mediation.py` | Time-ordered lagged decomposition, exploratory (Section 4.7) |

## 5. Determinism and integrity guarantees

- Two runs of the deterministic `--no-wb` path at seed 42 are bit-identical
  (`code/tests/test_golden.py::test_golden_determinism_two_runs`).
- Preprocessing leakage is blocked: imputation and standardization are fitted only on the
  training split of each CV fold. The measured leakage effect is −0.005 in out-of-fold R².
- Input SHA-256 prefixes are recorded in the metadata of every frozen result JSON.
- `code/tests/test_no_fallback_in_reported_series.py` asserts that no fallback-generated
  value reaches any reported series, and that the observed anchor shares match the 30.7% and
  6.25% stated in Section 3.2.

```bash
uv run --no-sync pytest code/tests -m "not golden"   # structural and containment checks
uv run --no-sync pytest code/tests -m golden         # determinism, slower
```

## 6. Excluded specifications

`legacy/` contains an earlier 20-country prototype, single-country SHAP diagnostics, category
counterfactual simulations, and a results dashboard. None of this is reported in the
manuscript. Section 5.4, item 11 explains the exclusion: a cross-sectional model learns
differences between countries and cannot be used to extrapolate the effect of an intervention
within one country. The directory is retained for provenance. See `legacy/README.md`.
