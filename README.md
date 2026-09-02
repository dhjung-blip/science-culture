# Science Culture and National Innovation — Replication Package

Replication code and frozen data for:

> **Science Culture and National Innovation: The ECAB Framework and a De-Circularized
> Measurement Audit of Cognitive Human Capital.**

Every headline figure, table, and figure in the manuscript is reproduced by a script in
this package from the frozen data it contains. Nothing in the paper depends on a source
that must be downloaded first, with one documented exception noted below.

| | |
|---|---|
| DOI | [`10.5281/zenodo.22249253`](https://doi.org/10.5281/zenodo.22249253) — concept DOI, cite this one; it resolves to the latest version. Version 1.0.0 alone is [`10.5281/zenodo.22249254`](https://doi.org/10.5281/zenodo.22249254) |
| Runtime | Python 3.12 (pinned via `uv.lock`; ≥ 3.10 works) |
| Core libraries | xgboost, scikit-learn, shap, statsmodels, pandas, numpy |
| Seed | 42 throughout; the golden test verifies bit-identical output across two runs |
| Code licence | MIT (`LICENSE`) |
| Data licence | CC BY 4.0 for derived files produced here; upstream sources retain their own terms (see below) |

---

## 1. Start here

1. **`REPRODUCIBILITY.md`** — environment, exact commands, and the script-to-table map.
2. **`DATA_DICTIONARY.md`** — every column, its source, its role, and its observed density.
3. **`COLUMN_REMOVAL_LOG.md`** — five unused columns removed before release, with the
   verification that no reported estimate changes.

## 2. Layout

```
.
├── README.md                     this file
├── REPRODUCIBILITY.md            environment, commands, script → table/figure map
├── DATA_DICTIONARY.md / .csv     column reference and provenance notes
├── COLUMN_REMOVAL_LOG.md         what was removed at release and why
├── LICENSE                       MIT (code)
├── pyproject.toml / uv.lock / requirements.txt
│
├── code/                         analysis pipeline (six sequential stages)
│   ├── main.py                   orchestrator
│   ├── data_collector.py         source ingestion — READ ITS HEADER FIRST
│   ├── preprocessor.py           missingness, lags, composites (fitted inside CV folds)
│   ├── model_trainer.py          XGBoost with country-level GroupKFold
│   ├── shap_analyzer.py          global, category, and axis SHAP decomposition
│   ├── scenario_simulator.py     counterfactual scenario matrix — methodology
│   │                             demonstration, not a policy prescription
│   ├── visualizer.py             figure generation
│   ├── config.py                 country set, feature groups, category map
│   └── tests/                    architecture, config, golden, and containment tests
│
├── analysis/                     manuscript tables, figures, and robustness scripts
│   ├── canonical/                FROZEN DATA + the scripts that run on it
│   │   ├── raw_dataset.csv           pre-interpolation source file (528 × 27)
│   │   ├── processed_dataset.csv     analysis file (528 × 64)
│   │   ├── external_tfp_pwt110.csv   Penn World Table 11.0 extract
│   │   └── external_triadic_oecd.csv OECD triadic patent families
│   └── figures/{en,kr}/          generated figures, English and Korean labels
│
├── data/external/                source-separated outcome extracts + source manifest
├── results/                      frozen result JSON — the provenance of every reported number
├── tools/                        release utilities
└── legacy/                       NOT USED BY THE PAPER — see legacy/README.md
```

**On the location of the frozen data.** The canonical CSVs live in `analysis/canonical/`
rather than a top-level `data/` directory because roughly fifteen analysis scripts resolve
that path literally. Moving the files would have required editing the scripts that produced
the published numbers, which we judged a worse trade than an imperfect directory name.

## 3. Reproducing the paper

```bash
uv sync
uv run --no-sync pytest code/tests -m "not golden"          # fast structural checks
uv run --no-sync pytest code/tests -m golden                # determinism check (slow)
MPLBACKEND=Agg uv run --no-sync python analysis/paper_tables_A_all.py
MPLBACKEND=Agg uv run --no-sync python analysis/paper_figures.py
```

`REPRODUCIBILITY.md` §4 maps each manuscript table and figure to the script that produces it.

## 4. Two things a reader should check first

**The fallback generator.** `code/data_collector.py` contains a placeholder generator used
for constructs with no harmonized international source and as an offline fallback. Its module
docstring states exactly what it does and what it does not affect. The claim that it does not
touch any reported result is mechanically checkable:

```bash
uv run --no-sync pytest code/tests/test_no_fallback_in_reported_series.py -v
```

This asserts that the PISA series carry zero placeholder cells, that the observed anchor
shares are the 30.7% and 6.25% reported in Section 3.2, and that placeholder retention in the
two attitudinal columns is confined to exactly the nine countries disclosed there.

**The `legacy/` directory.** It contains an earlier specification — a 20-country prototype,
single-country SHAP diagnostics, category counterfactual simulations, and a results dashboard.
None of it is reported in the manuscript, and Section 5.4, item 11 explains why the
counterfactual simulations were excluded. It is retained for provenance, not for replication.
See `legacy/README.md`.

## 5. Data sources and redistribution

All primary sources are publicly available: World Bank Open Data (CC BY 4.0), OECD PISA,
World Values Survey, ILOSTAT, WIPO GII, OECD triadic patent families and WIPO PCT applications
via the OECD Data Explorer, the OpenAlex API, and the Penn World Table 11.0. Extracts used are
listed with retrieval dates and SHA-1 hashes in `data/external/external_outcome_sources.csv`.

**One source is not redistributed.** World Values Survey microdata are obtained under the
survey programme's own access terms. Only the derived national aggregates appear in the
released files. A user who wants to rebuild those two columns from source must register with
the WVS and point `.env` at the microdata; every other column rebuilds without registration.

## 6. Citation

If you use this package, please cite the manuscript and this deposit. The deposit's concept DOI is [`10.5281/zenodo.22249253`](https://doi.org/10.5281/zenodo.22249253), which always resolves to the latest version; cite it in preference to a version-specific DOI. Machine-readable metadata, including both DOIs, is in `CITATION.cff`.
