"""WIPO PCT applications by origin — OECD SDMX ingestion (외부 준거 확장).

출처: OECD Data Explorer, dataflow OECD.STI.PIE,DSD_PATENTS@DF_PATENTS_WIPO
  PATENT_AUTHORITIES=9P50_1 (WIPO/PCT) · MEASURE=AP (applications) · UNIT=PATN
  DATE_TYPE=APPLICATION(주)·PRIORITY(민감도) · AGENT_ROLE=APPLICANT(origin, 주)·INVENTOR(민감도)
  WIPO=_T (기술 전체) · OECD_TECHNOLOGY_PATENT=_Z
국가귀속: OECD 의 분수계수(fractional count, 다출원인 분배) → 값은 소수.
명명: source-separated international-patenting criterion (NOT fully independent external outcome).
GII 와 완전 독립 아님(국제출원 채널은 일부 동일 활동 반영) — 한계로 명시.

산출: data/external/pct_applications.csv (필수 schema) + pct_applications_sensitivity.csv (date/role 변형).
실패 시: status=FAILED + 사유를 stdout/JSON 로 남기고 achieved 처럼 쓰지 않는다(빈 CSV 미작성).
재현: cd <package-root>; python analysis/outcome_extension_pct.py
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import sys
import urllib.request

OUT_MAIN = "data/external/pct_applications.csv"
OUT_SENS = "data/external/pct_applications_sensitivity.csv"
BASE = "https://sdmx.oecd.org/public/rest/data/OECD.STI.PIE,DSD_PATENTS@DF_PATENTS_WIPO,1.0"
SOURCE_NAME = "OECD Data Explorer / DSD_PATENTS@DF_PATENTS_WIPO (WIPO PCT, 9P50_1, MEASURE=AP)"

COUNTRIES = sorted(
    "AUS AUT BEL BRA CAN CHE CHL COL CRI CZE DEU DNK ESP EST FIN FRA GBR GRC HKG HUN "
    "IDN IRL ISL ISR ITA JPN KOR LTU LUX LVA MEX NLD NOR NZL PER POL PRT RUS SGP SVK "
    "SVN SWE TUR USA".split()
)
YEARS = list(range(2012, 2024))

# key dim order: PATENT_AUTHORITIES.FREQ.MEASURE.UNIT_MEASURE.DATE_TYPE.REF_AREA.
#                PARTNER_AREA.AGENT_ROLE.COOPERATION_TYPE.WIPO.OECD_TECHNOLOGY_PATENT
VARIANTS = {
    # (date_type, agent_role): column label
    ("APPLICATION", "APPLICANT"): "pct_app_applicant",  # 주 outcome
    ("PRIORITY", "APPLICANT"): "pct_prio_applicant",  # 날짜 민감도
    ("APPLICATION", "INVENTOR"): "pct_app_inventor",  # 원천역할 민감도
}


def _fetch_variant(date_type: str, agent_role: str) -> dict[tuple[str, int], float]:
    key = f"9P50_1.A.AP.PATN.{date_type}.._Z.{agent_role}._Z._T._Z"
    url = (
        f"{BASE}/{key}?startPeriod={YEARS[0]}&endPeriod={YEARS[-1]}"
        "&dimensionAtObservation=AllDimensions"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.sdmx.data+json",
            # OECD WAF 는 기본 Python-urllib UA 를 403 차단 → 브라우저형 UA 필요
            "User-Agent": "Mozilla/5.0 (research-replication; outcome_extension_pct.py)",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as r:  # noqa: S310 (trusted OECD host)
        d = json.loads(r.read().decode("utf-8"))
    ds = d["data"]["structures"][0]
    odims = ds["dimensions"]["observation"]
    codes = {dm["id"]: [v["id"] for v in dm["values"]] for dm in odims}
    idx = {dm["id"]: i for i, dm in enumerate(odims)}
    obs = d["data"]["dataSets"][0]["observations"]
    out: dict[tuple[str, int], float] = {}
    for k, v in obs.items():
        pos = [int(x) for x in k.split(":")]
        ref = codes["REF_AREA"][pos[idx["REF_AREA"]]]
        yr = int(codes["TIME_PERIOD"][pos[idx["TIME_PERIOD"]]])
        if ref in COUNTRIES and v and v[0] is not None:
            out[(ref, yr)] = float(v[0])
    return out


def main() -> int:
    os.makedirs("data/external", exist_ok=True)
    download_date = dt.date.today().isoformat()
    try:
        data = {lbl: _fetch_variant(*spec) for spec, lbl in VARIANTS.items()}
    except Exception as e:  # noqa: BLE001 — 실패는 FAILED 로 기록(꾸미지 않음)
        print(json.dumps({"status": "FAILED", "reason": repr(e), "source": SOURCE_NAME}))
        print(
            f"[FAILED] PCT 다운로드 실패: {e!r} — CSV 미작성, achieved 표시 금지", file=sys.stderr
        )
        return 1

    main_map = data["pct_app_applicant"]
    if not main_map:
        print(
            json.dumps({"status": "FAILED", "reason": "empty main variant", "source": SOURCE_NAME})
        )
        return 1

    # 필수 schema CSV (주 outcome = APPLICATION date · APPLICANT origin)
    with open(OUT_MAIN, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["country_iso3", "year", "pct_applications_origin", "source", "download_date", "notes"]
        )
        for c in COUNTRIES:
            for y in YEARS:
                v = main_map.get((c, y))
                if v is None:
                    continue
                w.writerow(
                    [
                        c,
                        y,
                        round(v, 4),
                        SOURCE_NAME,
                        download_date,
                        "OECD fractional count; WIPO/PCT 9P50_1; application date; applicant origin",
                    ]
                )

    # 민감도 CSV (날짜·역할 변형)
    with open(OUT_SENS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["country_iso3", "year", *VARIANTS.values(), "source", "download_date"])
        for c in COUNTRIES:
            for y in YEARS:
                row = [c, y]
                ok = False
                for lbl in VARIANTS.values():
                    v = data[lbl].get((c, y))
                    row.append(round(v, 4) if v is not None else "")
                    ok = ok or v is not None
                if ok:
                    w.writerow([*row, SOURCE_NAME, download_date])

    sha1 = hashlib.sha1(open(OUT_MAIN, "rb").read()).hexdigest()
    cov_countries = sorted({c for (c, _y) in main_map})
    summary = {
        "status": "ACHIEVED",
        "outcome": "pct_applications_origin",
        "source": SOURCE_NAME,
        "raw_sha1": sha1,
        "out_main": OUT_MAIN,
        "out_sensitivity": OUT_SENS,
        "download_date": download_date,
        "n_rows_main": len(main_map),
        "n_countries": len(cov_countries),
        "countries_missing_from_44": sorted(set(COUNTRIES) - set(cov_countries)),
        "years": [YEARS[0], YEARS[-1]],
        "_interpretation": "source-separated international-patenting criterion; not fully independent external outcome",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"\n[OK] PCT 저장: {OUT_MAIN} (n={len(main_map)}, {len(cov_countries)}국, sha1={sha1[:12]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
