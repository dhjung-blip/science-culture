"""OpenAlex high-impact publications — country-year ingestion (외부 준거 확장).

출처: OpenAlex REST API (api.openalex.org), works 엔드포인트 group_by=authorships.countries.
  total_publications     : filter publication_year:{y}, type:article|review
  top10_cited            : 위 + cited_by_percentile_year.min:90 (해당 출판연도 기준 상위10% 피인용)
국가귀속: group_by authorships.countries → whole/multi counting (다국적 논문은 각국에 1회 계수).
  fractional counting 은 본 집계에서 미구현 → whole counting 사용, 한계로 명시.
publication type: article|review 로 제한. citation window: OpenAlex 의 출판연도 정규화 백분위
  (publication-year normalized) — 단, 최근 연도는 피인용 누적창이 짧아 top10 비교가 불안정(한계).
field-normalized citation impact(FNCI): 본 스크립트는 산출하지 않음(컬럼 공란) — 상위10% 점유를
  high-impact criterion 으로 사용. 완전한 Scopus/WoS 대체 아님 → open bibliometric corroboration.

산출: data/external/openalex_top_cited.csv (필수 schema).
실패(rate limit·키·네트워크) 시: status=TODO/FAILED + 사유만 남기고 achieved 표시 금지.
재현: cd <package-root>; python analysis/outcome_extension_openalex.py
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

OUT = "data/external/openalex_top_cited.csv"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")  # polite pool (선택 — 미설정 시 익명 호출)
SOURCE_NAME = "OpenAlex API works group_by=authorships.countries (type:article|review; cited_by_percentile_year.min:90)"
YEARS = list(range(2012, 2024))

ISO2_TO_ISO3 = {
    "AU": "AUS",
    "AT": "AUT",
    "BE": "BEL",
    "BR": "BRA",
    "CA": "CAN",
    "CH": "CHE",
    "CL": "CHL",
    "CO": "COL",
    "CR": "CRI",
    "CZ": "CZE",
    "DE": "DEU",
    "DK": "DNK",
    "ES": "ESP",
    "EE": "EST",
    "FI": "FIN",
    "FR": "FRA",
    "GB": "GBR",
    "GR": "GRC",
    "HK": "HKG",
    "HU": "HUN",
    "ID": "IDN",
    "IE": "IRL",
    "IS": "ISL",
    "IL": "ISR",
    "IT": "ITA",
    "JP": "JPN",
    "KR": "KOR",
    "LT": "LTU",
    "LU": "LUX",
    "LV": "LVA",
    "MX": "MEX",
    "NL": "NLD",
    "NO": "NOR",
    "NZ": "NZL",
    "PE": "PER",
    "PL": "POL",
    "PT": "PRT",
    "RU": "RUS",
    "SG": "SGP",
    "SK": "SVK",
    "SI": "SVN",
    "SE": "SWE",
    "TR": "TUR",
    "US": "USA",
}
TARGET3 = set(ISO2_TO_ISO3.values())


def _group_by_country(year: int, top10: bool) -> dict[str, int]:
    filt = f"publication_year:{year},type:article|review"
    if top10:
        filt += ",cited_by_percentile_year.min:90"
    params = {
        "filter": filt,
        "group_by": "authorships.countries",
        "per_page": "200",
    }
    if MAILTO:
        params["mailto"] = MAILTO
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params, safe=":|")
    for attempt in range(4):
        try:
            _ua = f"mailto:{MAILTO}" if MAILTO else "science-culture-xai (replication)"
            req = urllib.request.Request(url, headers={"User-Agent": _ua})
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (trusted host)
                d = json.loads(r.read().decode("utf-8"))
            out: dict[str, int] = {}
            for g in d.get("group_by", []):
                iso2 = g["key"].rstrip("/").split("/")[-1].upper()
                iso3 = ISO2_TO_ISO3.get(iso2)
                if iso3 in TARGET3:
                    out[iso3] = int(g["count"])
            return out
        except Exception:  # noqa: BLE001 — rate limit 등은 재시도, 최종 실패는 호출부에서 처리
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    return {}


def main() -> int:
    os.makedirs("data/external", exist_ok=True)
    snapshot_date = dt.date.today().isoformat()
    rows: list[tuple[str, int, object, object]] = []
    try:
        for y in YEARS:
            tot = _group_by_country(y, top10=False)
            top = _group_by_country(y, top10=True)
            time.sleep(0.3)  # polite
            for iso3 in sorted(TARGET3):
                t = tot.get(iso3)
                k = top.get(iso3)
                if t is None and k is None:
                    continue
                rows.append((iso3, y, t if t is not None else "", k if k is not None else ""))
            print(f"  [{y}] countries total={len(tot)} top10={len(top)}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 — 데이터 확보 실패는 TODO 로 (꾸미지 않음)
        print(json.dumps({"status": "FAILED", "reason": repr(e), "source": SOURCE_NAME}))
        print(f"[FAILED] OpenAlex 다운로드 실패: {e!r} — CSV 미작성", file=sys.stderr)
        return 1

    if not rows:
        print(json.dumps({"status": "FAILED", "reason": "empty result", "source": SOURCE_NAME}))
        return 1

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "country_iso3",
                "year",
                "total_publications",
                "top10_cited_publications",
                "field_normalized_citation_impact",
                "counting_method",
                "source",
                "snapshot_date",
                "notes",
            ]
        )
        for r_iso3, r_year, r_total, r_top10 in rows:
            w.writerow(
                [
                    r_iso3,
                    r_year,
                    r_total,
                    r_top10,
                    "",  # FNCI 미산출(공란)
                    "whole_multi",
                    SOURCE_NAME,
                    snapshot_date,
                    "type=article|review; top10=cited_by_percentile_year.min:90; whole counting; recent-year citation window short",
                ]
            )

    sha1 = hashlib.sha1(open(OUT, "rb").read()).hexdigest()
    cov = sorted({str(r[0]) for r in rows})
    summary = {
        "status": "ACHIEVED",
        "outcome": "openalex_top10_cited_publications",
        "source": SOURCE_NAME,
        "raw_sha1": sha1,
        "out": OUT,
        "snapshot_date": snapshot_date,
        "n_rows": len(rows),
        "n_countries": len(cov),
        "countries_missing_from_44": sorted(TARGET3 - set(cov)),
        "years": [YEARS[0], YEARS[-1]],
        "counting_method": "whole_multi",
        "_interpretation": "open bibliometric high-impact-publication criterion; not a full Scopus/WoS substitute",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"\n[OK] OpenAlex 저장: {OUT} (n={len(rows)}, {len(cov)}국, sha1={sha1[:12]})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
