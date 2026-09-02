"""results_external_outcomes.json 빌더 — PCT/OpenAlex 외부 outcome 분석을 동결.

입력(read-only): data/external/outcome_extension_merged.csv 의 재료(merge 모듈) + external CSV.
출력: results/results_external_outcomes.json, data/external/external_outcome_sources.csv.
분석: country-mean(주) + pooled-cluster(보조) × 5변환, cohort-lag/placebo/leave-region/denom-sep,
      외부 outcome family BH-FDR. 새 수치는 전부 실데이터+스크립트 산출(꾸밈 없음). seed=42.
재현: cd <package-root>; python analysis/run_external_outcome_analysis.py
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import warnings

warnings.filterwarnings("ignore")

import outcome_extension_models as M
from outcome_extension_merge import TRANSFORMS, build_merged

OUT_JSON = "results/results_external_outcomes.json"
SOURCES_CSV = "data/external/external_outcome_sources.csv"
PREFERRED = "log1p_per_million"  # triadic 와 동일 기준 변환

# 보고 대상 outcome: PCT(주) + OpenAlex top10(주) + OpenAlex total(맥락)
REPORT = [
    (
        "pct_applications_origin",
        "pct",
        "pct_applications_origin",
        "data/external/pct_applications.csv",
        "OECD DSD_PATENTS@DF_PATENTS_WIPO(9P50_1/AP/APPLICANT)",
        "Source-separated international-patenting criterion; patent-family/international patenting outcome; not fully independent macro outcome",
    ),
    (
        "openalex_top10_cited",
        "openalex_top10",
        "top10_cited_publications",
        "data/external/openalex_top_cited.csv",
        "OpenAlex API group_by country; article|review; percentile >= 90",
        "Open-bibliometric high-impact-publication criterion; whole-counted; not field-normalized citation impact",
    ),
    (
        "openalex_total_publications",
        "openalex_total",
        "total_publications",
        "data/external/openalex_top_cited.csv",
        "OpenAlex API group_by country; article|review (total)",
        "publication-quantity criterion (context only)",
    ),
]
# 주 외부준거 가족(FDR·승격 대상). total 은 맥락용.
MAIN_KEYS = ("pct_applications_origin", "openalex_top10_cited")


def file_hash(p: str) -> str | None:
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def coverage(merged, count_col: str) -> dict:
    nn = merged[count_col].notna()
    return {
        "n_obs": int(nn.sum()),
        "n_countries": int(merged.loc[nn, "country"].nunique()),
        "years": [int(merged.loc[nn, "year"].min()), int(merged.loc[nn, "year"].max())],
        "countries": sorted(merged.loc[nn, "country"].unique().tolist()),
    }


def analyze_outcome(merged, key: str, count_col: str) -> dict:
    pref_col = f"{key}__{PREFERRED}"
    cm = M.country_mean_between(merged, pref_col)
    pl = M.pooled_cluster(merged, pref_col)
    tsens = {}
    sig_signs = []
    sig_neg_transforms = []  # 유의한 부호반전(음의) 변환
    for t in TRANSFORMS:
        col = f"{key}__{t}"
        cmt = M.country_mean_between(merged, col)
        plt = M.pooled_cluster(merged, col)
        tsens[t] = {"country_mean": cmt, "pooled": plt}
        if cmt and cmt["p"] is not None:
            pos = (cmt["beta_std"] or 0) > 0
            sig = cmt["p"] < 0.05
            sig_signs.append((sig, pos))
            if sig and not pos:
                sig_neg_transforms.append({"transform": t, "beta": cmt["beta_std"], "p": cmt["p"]})
    n_sig_pos = sum(1 for s, pos in sig_signs if s and pos)
    return {
        "preferred_transformation": PREFERRED,
        "country_mean": cm,
        "pooled": pl,
        "transformation_sensitivity": tsens,
        "n_transforms_sig_positive": n_sig_pos,
        "n_transforms_total": len(sig_signs),
        "sig_negative_transforms": sig_neg_transforms,
        "transformation_sensitive": bool(
            cm and cm["p"] is not None and cm["p"] < 0.05 and n_sig_pos < max(4, len(sig_signs) - 1)
        ),
        "cohort_lag": M.cohort_lag(merged, pref_col),
        "placebo": M.placebo_future_anchor(merged, pref_col),
        "leave_region": M.leave_region(merged, pref_col),
        "denominator_separation": M.denominator_separation(merged, key),
    }


def judge(key: str, res: dict) -> str:
    cm = res["country_mean"]
    pref_sig = bool(cm and cm["p"] is not None and cm["p"] < 0.05 and (cm["beta_std"] or 0) > 0)
    n_sig, n_tot = res["n_transforms_sig_positive"], res["n_transforms_total"]
    robust = pref_sig and n_sig >= max(4, n_tot - 1)
    rev = res["sig_negative_transforms"]
    rev_note = ""
    if rev:
        names = ", ".join(r["transform"] for r in rev)
        rev_note = f" The sign reverses (significantly negative) under the {names} scale."
    base = " This remains a between-country observational association, not a causal effect."
    if "pct" in key:
        if robust:
            return (
                "International-patenting criterion provides additional corroboration."
                + rev_note
                + base
            )
        if pref_sig:
            return "PCT provides transformation-sensitive corroboration." + rev_note + base
        return (
            "The direct-output association does not extend to PCT in this sample." + rev_note + base
        )
    if "openalex_top10" in key:
        if robust:
            return (
                "The association extends to high-impact (top-10% cited) publication output."
                + rev_note
                + base
            )
        if pref_sig:
            return (
                "The high-impact-publication association appears only on the variance-stabilised "
                "log1p-per-million scale (transformation-sensitive)." + rev_note + base
            )
        return (
            "The association is limited to quantity/formal output and does not extend to "
            "citation-adjusted output." + rev_note + base
        )
    # total (context)
    return "Publication-quantity context outcome." + rev_note + base


def _ts_block(res: dict) -> dict:
    """transformation_sensitivity 블록(스펙 + per_transform 상세)."""
    neg = res["sig_negative_transforms"]
    block = {
        "significant_only_for": PREFERRED,
        "n_significant_out_of_5": f"{res['n_transforms_sig_positive']}/{res['n_transforms_total']}",
        "transformation_sensitive": res["transformation_sensitive"],
        "note": "Transformation-sensitive corroboration; not scale-independent",
        "by_transformation": res["transformation_sensitivity"],
    }
    if neg:
        block["per_researcher_beta"] = neg[0]["beta"]
        block["per_researcher_p"] = neg[0]["p"]
        block["per_researcher_note"] = "Per-researcher reversal appears denominator-driven"
    return block


def _placebo_block(res: dict) -> dict | None:
    fwd, pb = res["cohort_lag"], res["placebo"]
    if not (fwd and pb):
        return None
    return {
        "forward_beta": fwd["beta_std"],  # PISA(t)→outcome(t+6) (legitimate forward)
        "placebo_beta": pb["beta_std"],  # future PISA→past outcome (placebo)
        "forward_p": fwd["p"],
        "placebo_p": pb["p"],
        "n_obs": fwd["n_obs"],
        "interpretation": "No temporal-precedence advantage (placebo comparable to forward); time precedence unidentified",
    }


def main() -> int:
    merged, hashes = build_merged()
    computed = {}
    fdr_pairs = []
    sources_rows = []
    for jkey, key, ccol, path, source, interp in REPORT:
        res = analyze_outcome(merged, key, ccol)
        cov = coverage(merged, ccol)
        computed[jkey] = (res, cov, file_hash(path), source, interp, path, key)
        cmv = res["country_mean"]
        if jkey in MAIN_KEYS and cmv and cmv["p"] is not None:
            fdr_pairs.append((jkey, cmv["p"]))
        sources_rows.append(
            [
                jkey,
                source,
                dt.date.today().isoformat(),
                path,
                file_hash(path) or "NA",
                f"{cov['n_countries']}/44 countries, {cov['years'][0]}-{cov['years'][1]}, n_obs={cov['n_obs']}",
            ]
        )

    fdr = M.bh_fdr(fdr_pairs)
    fdr_survive = {m["label"]: m["reject_05"] for m in fdr}

    def outcome_dict(jkey: str) -> dict:
        res, cov, sha1, source, interp, path, key = computed[jkey]
        cm = dict(res["country_mean"]) if res["country_mean"] else None
        if cm is not None:
            cm["preferred_transformation"] = PREFERRED
            cm["fdr_survives"] = fdr_survive.get(jkey)
        d = {
            "status": "ACHIEVED" if cov["n_obs"] > 0 else "TODO",
            "source": source,
            "raw_sha1": sha1,
            "raw_file": path,
            "coverage": {
                "n_countries": cov["n_countries"],
                "n_obs": cov["n_obs"],
                "years": f"{cov['years'][0]}–{cov['years'][1]}",
                "country_list": cov["countries"],
            },
            "country_mean_between": cm,
            "pooled_between": res["pooled"],
            "transformation_sensitivity": _ts_block(res),
            "cohort_lag_forward": res["cohort_lag"],
            "placebo": _placebo_block(res),
            "leave_region": res["leave_region"],
            "denominator_separation": res["denominator_separation"],
            "_judgment": judge(key, res),
            "_interpretation": interp,
        }
        if "pct" in key:
            d["validation_note"] = "KOR 11,083→21,867 matches WIPO-published values"
        if key.startswith("openalex"):
            d["counting_method"] = "whole counting"
        return d

    combined_hash = hashlib.sha1(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    total = outcome_dict("openalex_total_publications")
    total["status"] = "ACHIEVED_CONTEXT_ONLY"
    total["note"] = (
        "Secured for context; not promoted to main outcome (results retained here, not a headline)"
    )
    out = {
        "_meta": {
            "purpose": "External outcome extension: WIPO PCT and OpenAlex high-impact publications",
            "main_estimand": "44-country between-country conditional association",
            "no_causal_claim": True,
            "no_temporal_precedence_claim": True,
            "data_coverage": "44 countries, 2012–2023, n=528 for achieved PCT and OpenAlex top10",
            "created_from": "official/statistical external outcome sources",
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "input_hashes": hashes,
            "data_snapshot_hash": combined_hash,
            "seed": M.SEED,
            "anchor": M.ANCHOR,
            "controls_confounder_only": M.CONTROLS,
            "preferred_transformation": PREFERRED,
            "_caveat": (
                "PCT = source-separated international-patenting criterion (not fully independent); "
                "OpenAlex = open bibliometric corroboration (whole counting; recent-year citation window short; "
                "FNCI not computed). Fully independent macro scope check remains TFP (null). GII = benchmark only."
            ),
        },
        "pct_applications_origin": outcome_dict("pct_applications_origin"),
        "openalex_top10_cited": outcome_dict("openalex_top10_cited"),
        "openalex_total_publications": total,
        "external_outcome_family_fdr": {
            "method": "Benjamini-Hochberg across new external-outcome family (PCT, OpenAlex top-10%)",
            "members": fdr,
        },
        "not_achieved": {
            "openalex_fnci": "TODO — field-year normalized citation impact not computed",
            "openalex_fractional_counting": "TODO — whole counting used",
            "fixed_citation_window": "TODO — current percentile uses publication-year percentile, recent-year citation window shorter",
        },
    }
    outcomes_json = {
        k: out[k]
        for k in ("pct_applications_origin", "openalex_top10_cited", "openalex_total_publications")
    }
    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "outcome",
                "source_url_or_source_name",
                "download_date",
                "raw_file",
                "sha1",
                "coverage_note",
            ]
        )
        w.writerows(sources_rows)

    # 콘솔 요약
    print("=" * 96)
    print(
        "외부 outcome (PCT·OpenAlex) — 주 사양: country-mean between-only"
    )
    print("=" * 96)
    for jkey, *_ in REPORT:
        o = outcomes_json[jkey]
        cm = o["country_mean_between"]
        pl = o["pooled_between"]
        cs = (
            f"β={cm['beta_std']:+.3f} p={cm['p']:.4f} n={cm['n_countries']}국 R²={cm['r2']}"
            if cm
            else "N/A"
        )
        ps = f"β={pl['beta_std']:+.3f} p={pl['p']:.4f}" if pl else "N/A"
        print(f"\n■ {jkey}  [{o['status']}]  ({PREFERRED})")
        print(f"   country-mean(주): {cs}")
        print(f"   pooled-cluster  : {ps}")
        print(f"   변환 민감: {o['transformation_sensitivity']['n_significant_out_of_5']}")
        print(f"   판정: {o['_judgment']}")
    print("\nFDR(BH, 신규 family):")
    for m in fdr:
        print(f"   {m['label']}: p={m['p']} q={m['q_fdr']} reject05={m['reject_05']}")
    print(f"\n[OK] 저장: {OUT_JSON} · {SOURCES_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
