# `data/external/` — source-separated outcome extracts

`analysis/external_outcomes_template.py` detects the CSVs in this folder automatically. If a
file is absent, the corresponding outcome is recorded as unavailable in
`results/results_identification_addendum.json` rather than silently skipped.

The design intent is that downloads are not forced inside the analysis scripts. The ingestion
contract below is fixed; anyone who supplies a conforming CSV gets the analysis immediately.
Retrieval dates and SHA-1 hashes for every extract used in the manuscript are in
`external_outcome_sources.csv`.

## Schemas

### `triadic_patents.csv` — OECD triadic patent families

| Column | Description |
|---|---|
| `country_iso3` | ISO 3166-1 alpha-3 code |
| `year` | Calendar year |
| `triadic_patent_families` | Families filed simultaneously at EPO, JPO and USPTO (fractional count) |

Provenance: reshaped from `analysis/canonical/external_triadic_oecd.csv`, a frozen OECD SDMX
snapshot (INVENTOR, PRIORITY, Patent Families, Total), 44 countries × 2012–2022.

**Interpretation caveat.** This indicator is conceptually adjacent to GII item 5.2.5 (patent
families filed in at least two offices). It is therefore not a fully independent external
criterion. The manuscript classifies it as tier (b) in Table 2: a patent-quality criterion
that is separated from the GII in measurement and source, not separated in construct. The only
fully independent macro outcome in the paper is Penn World Table total factor productivity.

### `pct_applications.csv` — WIPO PCT international applications by origin

| Column | Description |
|---|---|
| `country_iso3` | ISO 3166-1 alpha-3 code |
| `year` | Calendar year |
| `pct_applications` | PCT applications attributed to the applicant's country of origin |

Source: OECD `DSD_PATENTS@DF_PATENTS_WIPO` (9P50_1 / AP / APPLICANT). 44 countries,
2012–2023, n = 528.

### `openalex_top_cited.csv` — OpenAlex top-decile cited publications

| Column | Description |
|---|---|
| `country_iso3` | ISO 3166-1 alpha-3 code |
| `year` | Publication year |
| `top10_cited_publications` | Works of type article or review in the top citation decile of their publication year |
| `total_publications` | All works of type article or review |
| `counting_method` | `whole_multi` — whole counting with multiple country attribution |
| `field_normalized_citation_impact` | **Empty in this release.** Field normalization was pre-specified but not computed; see Table 2, tier (f) |

Source: OpenAlex API `group_by` country, snapshot 2026-06-30. 44 countries, 2012–2023,
n = 528.

**Two pre-specified adjustments are not present.** Field-normalized citation impact and
fractional author-country counting both require a work-level re-extraction rather than a
`group_by` query, and are formally deferred to future work. The third pre-specified
adjustment, a fixed citation window, was computed and is reported in Section 4.5. The unmet
items are left visible in Table 2 rather than removed from it.

### `outcome_extension_merged.csv`, `pct_applications_sensitivity.csv`

Derived files produced by `analysis/outcome_extension_merge.py`: the frozen panel joined to
the external outcomes, with the five scaling transformations of Table 7 precomputed.

### `external_outcome_sources.csv` — source manifest

One row per extract: outcome name, source identifier, retrieval date, file path, SHA-1, and a
coverage note. Update this file if any extract is replaced.
