# `analysis/` — manuscript tables, figures, and robustness scripts

These scripts import the pipeline stages in `code/` **read-only** and recompute the tables,
figures, and robustness results of the manuscript. They never overwrite pipeline output; every
script writes to stdout, to a PNG, or to a JSON file under `results/`.

Run them from the package root. Each script adds `code/` to the module path automatically; if
you have moved it, set `SC_CODE_DIR`.

```bash
MPLBACKEND=Agg python analysis/decircle_experiment.py
```

## Frozen data

`analysis/canonical/` holds the frozen data the published numbers were computed from:

| File | Contents |
|---|---|
| `raw_dataset.csv` | Pre-interpolation source file, 528 × 27. The reference for the observed-anchor shares in Section 3.2 |
| `processed_dataset.csv` | Analysis file, 528 × 64: within-country interpolation, lag features, category composites |
| `external_tfp_pwt110.csv` | Penn World Table 11.0 extract, `ctfp` and `rtfpna` |
| `external_triadic_oecd.csv` | OECD triadic patent families, SDMX snapshot |

Scripts that read these files print the SHA-256 prefix of each input, so a substituted file is
visible immediately. Column-level provenance is in `DATA_DICTIONARY.md` at the package root.

The remaining two imputation stages — KNN with k = 5, then median fill — are applied inside
cross-validation folds and therefore appear in neither file.

## Script index

The full script-to-table map is in `REPRODUCIBILITY.md` §4. In brief:

| Group | Scripts |
|---|---|
| Tables and figures | `paper_tables_T2_T4.py`, `paper_tableB_stability.py`, `paper_tables_A_all.py`, `paper_figures.py`, `paper_figures_kr.py` |
| De-circularization | `decircle_experiment.py`, `gii_output_experiment.py`, `canonical/pooled_beta_canonical.py`, `canonical/direct_output_robustness.py` |
| Source-separated outcomes | `outcome_extension_*.py`, `run_external_outcome_analysis.py`, `external_outcomes_template.py`, `canonical/external_reference_tfp.py` |
| Sensitivity | `oster_sensitivity.py`, `canonical/oster_confounder_only.py`, `canonical/oster_rmax_curve.py`, `wild_cluster_bootstrap.py`, `canonical/loco_jackknife.py`, `canonical/robustness_*.py`, `patent_within_denominator_recheck.py`, `effective_sample_audit.py` |
| Construct diagnostics | `sci_culture_axis_split.py`, `canonical/direct_output_axis_split.py` |
| Temporal precedence | `pisa_cohort_lag.py`, `future_pisa_placebo.py`, `placebo_robustness.py`, `lagged_mediation.py` |

Scripts for specifications excluded from the manuscript are in `legacy/`, not here.
