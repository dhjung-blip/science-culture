# Data dictionary — canonical analysis files

This dictionary documents the two canonical data files released with the manuscript
*Science Culture and National Innovation: The ECAB Framework and a De-Circularized
Measurement Audit of Cognitive Human Capital*.

| File | Rows | Columns | SHA-1 |
| --- | --- | --- | --- |
| `processed_dataset.csv` | 528 | 64 | `b59486517b790aff2b24dd61e17824252eeab6d4` |
| `raw_dataset.csv` | 528 | 27 | `bf00078a3260208fb1896e1433e99449bc11e38c` |

`raw_dataset.csv` is the merged source file **before** any interpolation or imputation, and is
the file against which the observed-anchor shares reported in Section 3.2 of the manuscript are
computed. `processed_dataset.csv` adds within-country linear interpolation, lag features and
category composites. The remaining two imputation stages (KNN with k = 5, then median fill) are
applied inside cross-validation folds and are therefore not present in either file.

Panel: 44 countries x 12 years (2012-2023) = 528 country-year cells.

## 1  Column reference

| Column | Category | Description | Source | Role | Observed pre-interpolation | Legacy placeholder cells | Non-null in analysis file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `country` | Identifier | ISO 3166-1 alpha-3 country code | — | Panel key | 528 (100.00%) | — | 528 |
| `year` | Identifier | Calendar year, 2012–2023 | — | Panel key | 528 (100.00%) | — | 528 |
| `sci_literacy` | Cognition | PISA science literacy, mean national score | OECD PISA | Treatment variable | 162 (30.68%) | — | 528 |
| `sci_trust` | Attitude | Confidence in science, national share (WVS) | World Values Survey W6/W7 | Predictor — construct diagnostic only | 33 (6.25%) | 108 | 384 |
| `tech_acceptance` | Attitude | Technology optimism, national share (WVS) | World Values Survey W6/W7 | Predictor — construct diagnostic only | 33 (6.25%) | 108 | 384 |
| `relative_salary` | Reward level | Professional wage ratio relative to the national mean | ILOSTAT | Predictor | 463 (87.69%) | — | 516 |
| `rd_budget_per_researcher` | Reward level | R&D resources per researcher, derived from R&D/GDP, GDP and researcher density | Derived in preprocessor.py | Predictor | 440 (83.33%) | — | 492 |
| `innovation_score` | Outcome | Global Innovation Index composite score (0–100) | WIPO GII | Contamination-bearing benchmark outcome | 528 (100.00%) | — | 528 |
| `rd_gdp_pct` | Physical scale | Gross domestic R&D expenditure as a share of GDP (%) | World Bank Open Data (WDI) | Control (confounder) | 496 (93.94%) | — | 528 |
| `researchers_per_m` | Physical scale | Researchers in R&D per million population | World Bank Open Data (WDI) | Mediator — excluded from primary control set | 440 (83.33%) | — | 492 |
| `patent_apps` | Physical scale | Patent applications, residents | World Bank Open Data (WDI) | Predictor and outcome numerator | 427 (80.87%) | — | 528 |
| `journal_articles` | Physical scale | Scientific and technical journal articles | World Bank Open Data (WDI) | Predictor and outcome numerator | 473 (89.58%) | — | 516 |
| `gdp_per_capita` | Physical scale | GDP per capita, current US$ | World Bank Open Data (WDI) | Control (confounder) | 528 (100.00%) | — | 528 |
| `gov_edu_exp_pct` | Operational structure | Government expenditure on education as a share of GDP (%) | World Bank Open Data (WDI) | Predictor | 451 (85.42%) | — | 528 |
| `internet_users_pct` | Operational structure | Individuals using the internet (% of population) | World Bank Open Data (WDI) | Predictor | 528 (100.00%) | — | 528 |
| `hi_tech_export_pct` | Reward level | High-technology exports as a share of manufactured exports (%) | World Bank Open Data (WDI) | Predictor | 526 (99.62%) | — | 528 |
| `tertiary_enroll` | Cognition | Gross tertiary enrolment ratio (%) | World Bank Open Data (WDI) | Predictor | 490 (92.80%) | — | 528 |
| `mobile_subs_per100` | Operational structure | Mobile cellular subscriptions per 100 people | World Bank Open Data (WDI) | Predictor | 528 (100.00%) | — | 528 |
| `secure_servers_per_m` | Operational structure | Secure internet servers per million people | World Bank Open Data (WDI) | Predictor | 528 (100.00%) | — | 528 |
| `youth_unemp_pct` | Reward level | Youth unemployment rate (%) | World Bank Open Data (WDI) | Predictor | 528 (100.00%) | — | 528 |
| `tertiary_attainment` | Cognition | Population 25+ with a bachelor's degree or higher (%) | World Bank Open Data (WDI) | Predictor | 381 (72.16%) | — | 504 |
| `population` | Physical scale | Total population | World Bank Open Data (WDI) | Denominator for per-capita outcomes | 528 (100.00%) | — | 528 |
| `articles_per_m_log` | Outcome | log1p of journal articles per million population | Derived in preprocessor.py | Primary de-circularized outcome | 473 (89.58%) | — | 473 |
| `patents_per_m_log` | Outcome | log1p of patent applications per million population | Derived in preprocessor.py | Primary de-circularized outcome | 427 (80.87%) | — | 427 |
| `pisa_math` | Cognition | PISA mathematics, mean national score | OECD PISA | Predictor | 163 (30.87%) | — | 528 |
| `pisa_reading` | Cognition | PISA reading, mean national score | OECD PISA | Predictor | 162 (30.68%) | — | 528 |
| `country_name` | Identifier | Country or economy name | — | Panel key | 528 (100.00%) | — | 528 |
| `sci_literacy_lag3` | Lag feature | 3-year within-country lag of sci_literacy | OECD PISA | Predictor (lag-3 only survived missingness pruning) | 162 (30.68%) | — | 396 |
| `sci_literacy_lag5` | Lag feature | 5-year within-country lag of sci_literacy | OECD PISA | Generated but pruned from the final feature set | 162 (30.68%) | — | 308 |
| `sci_literacy_lag7` | Lag feature | 7-year within-country lag of sci_literacy | OECD PISA | Generated but pruned from the final feature set | 162 (30.68%) | — | 220 |
| `sci_trust_lag3` | Lag feature | 3-year within-country lag of sci_trust | World Values Survey W6/W7 | Predictor (lag-3 only survived missingness pruning) | 33 (6.25%) | 108 | 288 |
| `sci_trust_lag5` | Lag feature | 5-year within-country lag of sci_trust | World Values Survey W6/W7 | Generated but pruned from the final feature set | 33 (6.25%) | 108 | 224 |
| `sci_trust_lag7` | Lag feature | 7-year within-country lag of sci_trust | World Values Survey W6/W7 | Generated but pruned from the final feature set | 33 (6.25%) | 108 | 160 |
| `tech_acceptance_lag3` | Lag feature | 3-year within-country lag of tech_acceptance | World Values Survey W6/W7 | Predictor (lag-3 only survived missingness pruning) | 33 (6.25%) | 108 | 288 |
| `tech_acceptance_lag5` | Lag feature | 5-year within-country lag of tech_acceptance | World Values Survey W6/W7 | Generated but pruned from the final feature set | 33 (6.25%) | 108 | 224 |
| `tech_acceptance_lag7` | Lag feature | 7-year within-country lag of tech_acceptance | World Values Survey W6/W7 | Generated but pruned from the final feature set | 33 (6.25%) | 108 | 160 |
| `pisa_math_lag3` | Lag feature | 3-year within-country lag of pisa_math | OECD PISA | Predictor (lag-3 only survived missingness pruning) | 163 (30.87%) | — | 396 |
| `pisa_math_lag5` | Lag feature | 5-year within-country lag of pisa_math | OECD PISA | Generated but pruned from the final feature set | 163 (30.87%) | — | 308 |
| `pisa_math_lag7` | Lag feature | 7-year within-country lag of pisa_math | OECD PISA | Generated but pruned from the final feature set | 163 (30.87%) | — | 220 |
| `pisa_reading_lag3` | Lag feature | 3-year within-country lag of pisa_reading | OECD PISA | Predictor (lag-3 only survived missingness pruning) | 162 (30.68%) | — | 396 |
| `pisa_reading_lag5` | Lag feature | 5-year within-country lag of pisa_reading | OECD PISA | Generated but pruned from the final feature set | 162 (30.68%) | — | 308 |
| `pisa_reading_lag7` | Lag feature | 7-year within-country lag of pisa_reading | OECD PISA | Generated but pruned from the final feature set | 162 (30.68%) | — | 220 |
| `tertiary_enroll_lag3` | Lag feature | 3-year within-country lag of tertiary_enroll | World Bank Open Data (WDI) | Predictor (lag-3 only survived missingness pruning) | 490 (92.80%) | — | 396 |
| `tertiary_enroll_lag5` | Lag feature | 5-year within-country lag of tertiary_enroll | World Bank Open Data (WDI) | Generated but pruned from the final feature set | 490 (92.80%) | — | 308 |
| `tertiary_enroll_lag7` | Lag feature | 7-year within-country lag of tertiary_enroll | World Bank Open Data (WDI) | Generated but pruned from the final feature set | 490 (92.80%) | — | 220 |
| `tertiary_attainment_lag3` | Lag feature | 3-year within-country lag of tertiary_attainment | World Bank Open Data (WDI) | Predictor (lag-3 only survived missingness pruning) | 381 (72.16%) | — | 378 |
| `tertiary_attainment_lag5` | Lag feature | 5-year within-country lag of tertiary_attainment | World Bank Open Data (WDI) | Generated but pruned from the final feature set | 381 (72.16%) | — | 294 |
| `tertiary_attainment_lag7` | Lag feature | 7-year within-country lag of tertiary_attainment | World Bank Open Data (WDI) | Generated but pruned from the final feature set | 381 (72.16%) | — | 210 |
| `hi_tech_export_pct_lag3` | Lag feature | 3-year within-country lag of hi_tech_export_pct | World Bank Open Data (WDI) | Predictor (lag-3 only survived missingness pruning) | 526 (99.62%) | — | 396 |
| `hi_tech_export_pct_lag5` | Lag feature | 5-year within-country lag of hi_tech_export_pct | World Bank Open Data (WDI) | Generated but pruned from the final feature set | 526 (99.62%) | — | 308 |
| `hi_tech_export_pct_lag7` | Lag feature | 7-year within-country lag of hi_tech_export_pct | World Bank Open Data (WDI) | Generated but pruned from the final feature set | 526 (99.62%) | — | 220 |
| `rd_budget_per_researcher_lag3` | Lag feature | 3-year within-country lag of rd_budget_per_researcher | Derived in preprocessor.py | Predictor (lag-3 only survived missingness pruning) | 440 (83.33%) | — | 369 |
| `rd_budget_per_researcher_lag5` | Lag feature | 5-year within-country lag of rd_budget_per_researcher | Derived in preprocessor.py | Generated but pruned from the final feature set | 440 (83.33%) | — | 287 |
| `rd_budget_per_researcher_lag7` | Lag feature | 7-year within-country lag of rd_budget_per_researcher | Derived in preprocessor.py | Generated but pruned from the final feature set | 440 (83.33%) | — | 205 |
| `relative_salary_lag3` | Lag feature | 3-year within-country lag of relative_salary | ILOSTAT | Predictor (lag-3 only survived missingness pruning) | 463 (87.69%) | — | 387 |
| `relative_salary_lag5` | Lag feature | 5-year within-country lag of relative_salary | ILOSTAT | Generated but pruned from the final feature set | 463 (87.69%) | — | 301 |
| `relative_salary_lag7` | Lag feature | 7-year within-country lag of relative_salary | ILOSTAT | Generated but pruned from the final feature set | 463 (87.69%) | — | 215 |
| `youth_unemp_pct_lag3` | Lag feature | 3-year within-country lag of youth_unemp_pct | World Bank Open Data (WDI) | Predictor (lag-3 only survived missingness pruning) | 528 (100.00%) | — | 396 |
| `youth_unemp_pct_lag5` | Lag feature | 5-year within-country lag of youth_unemp_pct | World Bank Open Data (WDI) | Generated but pruned from the final feature set | 528 (100.00%) | — | 308 |
| `youth_unemp_pct_lag7` | Lag feature | 7-year within-country lag of youth_unemp_pct | World Bank Open Data (WDI) | Generated but pruned from the final feature set | 528 (100.00%) | — | 220 |
| `idx_sci_culture` | Composite | Unweighted mean of min–max rescaled science-culture features (7) | Derived in preprocessor.py | SHAP category accounting only | n/a (n/a) | — | 528 |
| `idx_hard_rd` | Composite | Unweighted mean of min–max rescaled physical-scale features (5) | Derived in preprocessor.py | SHAP category accounting only | n/a (n/a) | — | 528 |
| `idx_structure` | Composite | Unweighted mean of min–max rescaled operational-structure features (4) | Derived in preprocessor.py | SHAP category accounting only | n/a (n/a) | — | 528 |
| `idx_economic` | Composite | Unweighted mean of min–max rescaled reward-level features (4) | Derived in preprocessor.py | SHAP category accounting only | n/a (n/a) | — | 528 |

## 2  Provenance notes

### P1  Legacy placeholder values in the attitudinal variables

An early prototype of the data-construction pipeline generated placeholder series for variables
that lacked a harmonized international source. The production pipeline overwrites these with
observed values for every country covered by the corresponding source, but for `sci_trust` and
`tech_acceptance` nine countries are absent from the World Values Survey extract and retain the
placeholder series:

> Austria, Belgium, Denmark, Finland, France, Israel, Norway, Sweden, Switzerland

Composition of the two attitudinal columns:

| Category | Countries | Cells | Share of the 528-cell grid |
| --- | --- | --- | --- |
| Observed WVS anchors | 23 | 33 | 6.25% |
| Interpolated between observed anchors | 23 | 243 | 46.02% |
| Legacy placeholder, not survey-derived | 9 | 108 | 20.45% |
| No value (excluded from models) | 12 | 144 | 27.27% |

Of the 384 attitudinal cells that enter the models, 108 (28.1%) are placeholder values. This is
disclosed in Section 3.2 of the manuscript. The attitudinal axis is reported throughout as a
construct diagnostic and never as a tested channel, and no claim in the paper rests on it.

### P2  PISA observed anchors

PISA is administered in 2012, 2015, 2018 and 2022, so a complete grid would contain 44 x 4 = 176
observed cells (33.3%). The actual count is 162 (30.68%) because six countries participated in
only two cycles and two in three. No country carries placeholder PISA values; all `sci_literacy`,
`pisa_math` and `pisa_reading` values originate from the OECD PISA source.

### P3  Undetermined provenance in two reward-level variables

The same overwrite logic that produced note P1 applies to `relative_salary` (ILOSTAT) and, through
it, to `rd_budget_per_researcher`. Because ILOSTAT reports annually, complete 12-year coverage is
also the expected pattern for genuine data, so the two cases cannot be distinguished from the
released files alone: 16 and 18 countries respectively outside the 18-country legacy set have
complete coverage, which is consistent with real data, while 14 and 11 countries inside the legacy
set do. Running `analysis/effective_sample_audit.py` with live source access resolves this
definitively; that script already separates observed anchors from retained placeholder cells.
Neither variable is a treatment, an outcome or a control in any reported specification.

### P3b  Variables generated but excluded from the final feature set

Five columns produced by earlier versions of the pipeline were removed from both released files.
See `COLUMN_REMOVAL_LOG.md`.

### P4  Lag features

Lag features were generated at 3, 5 and 7 years. Only the eight lag-3 features survived
missingness pruning and enter the 32-feature analysis set; the lag-5 and lag-7 columns are
retained in the file for transparency but are not used in any reported specification.

### P5  Redistribution

World Values Survey microdata are obtained under the survey programme's own access terms and are
not redistributed in this package; only the derived national aggregates appear in these files.
All other sources (World Bank WDI, OECD PISA, OECD triadic patent families, WIPO PCT, WIPO GII,
OpenAlex, ILOSTAT, Penn World Table 11.0) are publicly accessible, and the extracts used are
listed with hashes in `data/external/external_outcome_sources.csv`.

