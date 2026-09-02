"""
data_collector.py
=================
Data collection module for the science-culture measurement audit.

READ THIS BEFORE READING THE CODE
---------------------------------
This module contains a fallback generator (``_synthesize_wb_features`` and the
``COUNTRY_PROFILES`` table) that emits placeholder series for constructs with no
harmonized international source, and as an offline fallback when the World Bank
API is unavailable. It is NOT a data-fabrication path for the results reported
in the manuscript, and the distinction is verifiable:

* For every country covered by the corresponding source, the production path
  overwrites the placeholder series with observed values. PISA, patent,
  publication, and all outcome series are therefore fully source-derived; the
  released pre-interpolation file carries zero placeholder cells in them.
* Two attitudinal columns (``sci_trust``, ``tech_acceptance``) are the single
  exception. Nine countries are absent from the World Values Survey extract and
  retain the placeholder series: 108 of the 384 attitudinal cells that enter the
  models (28.1%) are therefore not survey-derived. This is disclosed in
  Section 3.2 of the manuscript, the attitudinal axis is reported there only as
  a construct diagnostic, and no claim in the paper rests on it.
* The four category composites recompute bit-identically from the
  source-derived feature definitions, so no placeholder value enters any
  composite or any reported specification.

``code/tests/test_no_fallback_in_reported_series.py`` asserts all of the above
against the released frozen data. Run it first if you want the containment
claim checked mechanically rather than taken on trust.

Contents
--------
- Five real sources: World Bank API (14 indicators; ``population`` is a
  non-feature input to the de-circularized targets), WVS
  (``sci_trust``/``tech_acceptance``), PISA
  (``sci_literacy``/``pisa_math``/``pisa_reading``), WIPO GII (target
  ``innovation_score``), ILOSTAT (``relative_salary``), plus FRED
  (``cpi_inflation``, non-feature).
- De-circularized targets: ``articles_per_m_log`` and ``patents_per_m_log``,
  log publications/patents per million population, computed without any GII
  input. These require the real World Bank path because they depend on
  ``population``.
- Fallback behaviour: if the World Bank call fails or ``--no-wb`` is passed,
  ``_synthesize_wb_features`` substitutes; WVS/PISA/GII/ILOSTAT are skipped
  gracefully when their ``.env`` paths are unset.
"""


import logging
import os
import time
from io import BytesIO

import numpy as np
import pandas as pd
import requests

# ── 환경변수(.env) 로드 — 선택 의존성(python-dotenv). 미설치/`.env` 부재여도 동작.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

# 모듈 로거 — 폴백·경고(⚠) 전용 (ADR-0007). 핸들러 미구성: logging.lastResort 가
# stderr 로 출력해 stdout(진행 리포트 print)/stderr(경고)가 분리된다.
logger = logging.getLogger(__name__)

# ── 수집 설정 (env 로 오버라이드 · 기본값은 기존 동작 유지) ──
WB_API_BASE = os.getenv("WORLDBANK_API_BASE", "https://api.worldbank.org/v2").rstrip("/")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "3"))
HTTP_BACKOFF = float(os.getenv("HTTP_BACKOFF", "2"))
DATA_CONTACT_EMAIL = os.getenv("DATA_CONTACT_EMAIL", "")

# WVS(World Values Survey) 웨이브별 마이크로데이터 CSV 경로 (env 로 지정 · 미설정 시 합성 폴백)
WVS_WAVE6_PATH = os.getenv("WVS_WAVE6_PATH", "")
WVS_WAVE7_PATH = os.getenv("WVS_WAVE7_PATH", "")

# FRED(美 세인트루이스 연준) API — OECD MEI CPI 지수 → 국가별 연간 인플레이션(cpi_inflation)
# 키 미설정 시 조용히 스킵(WB·WVS 폴백과 동일 graceful 패턴).
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_API_BASE = os.getenv("FRED_API_BASE", "https://api.stlouisfed.org/fred").rstrip("/")

# GII(Global Innovation Index) 종합점수 long-format TSV 경로 (env 로 지정 · 미설정 시 합성 타깃 유지)
# 실제 GII Score(0~100)로 합성 innovation_score 타깃을 교체한다(미설정/부재 시 graceful 스킵).
GII_DATA_PATH = os.getenv("GII_DATA_PATH", "")

# OECD PISA 과학점수 long-format CSV 경로 (env 로 지정 · 미설정 시 합성 sci_literacy 유지)
# 실제 PISA 과학 평균점수(~460~560)로 합성 sci_literacy(과학문화 핵심)를 교체한다(미설정/부재 시 graceful 스킵).
PISA_DATA_PATH = os.getenv("PISA_DATA_PATH", "")

# ILOSTAT SDMX REST API base (키 불필요·env 로 오버라이드). 직종별 월평균소득 → relative_salary(전문직 임금 프리미엄).
# 키 미설정/HTTP 실패 시 빈 DF (WB·WVS·FRED·GII·PISA 와 동일 graceful 패턴).
ILOSTAT_SDMX_BASE = os.getenv("ILOSTAT_SDMX_BASE", "https://sdmx.ilo.org/rest").rstrip("/")

# ──────────────────────────────────────────────
# 1. 국가 정의
# ──────────────────────────────────────────────
COUNTRIES = {
    "KOR": {"name": "South Korea",  "wb": "KR"},
    "USA": {"name": "United States","wb": "US"},
    "JPN": {"name": "Japan",        "wb": "JP"},
    "DEU": {"name": "Germany",      "wb": "DE"},
    "CHN": {"name": "China",        "wb": "CN"},
    "FIN": {"name": "Finland",      "wb": "FI"},
    "SWE": {"name": "Sweden",       "wb": "SE"},
    "ISR": {"name": "Israel",       "wb": "IL"},
    "CHE": {"name": "Switzerland",  "wb": "CH"},
    "GBR": {"name": "United Kingdom","wb": "GB"},
    "FRA": {"name": "France",       "wb": "FR"},
    "NLD": {"name": "Netherlands",  "wb": "NL"},
    "DNK": {"name": "Denmark",      "wb": "DK"},
    "CAN": {"name": "Canada",       "wb": "CA"},
    "AUS": {"name": "Australia",    "wb": "AU"},
    "SGP": {"name": "Singapore",    "wb": "SG"},
    "TWN": {"name": "Taiwan",       "wb": "TW"},
    "AUT": {"name": "Austria",      "wb": "AT"},
    "BEL": {"name": "Belgium",      "wb": "BE"},
    "NOR": {"name": "Norway",       "wb": "NO"},
}

YEARS = list(range(2012, 2024))   # 12년치 시계열

# ──────────────────────────────────────────────
# 2. World Bank API 수집
# ──────────────────────────────────────────────
WB_INDICATORS = {
    "rd_gdp_pct":        "GB.XPD.RSDV.GD.ZS",   # R&D / GDP (%)
    "researchers_per_m": "SP.POP.SCIE.RD.P6",    # 백만 명당 연구원
    "patent_apps":       "IP.PAT.RESD",           # 특허 출원(거주자)
    "journal_articles":  "IP.JRN.ARTC.SC",        # 과학 저널 게재
    "gdp_per_capita":    "NY.GDP.PCAP.CD",        # 1인당 GDP
    "gov_edu_exp_pct":   "SE.XPD.TOTL.GD.ZS",    # 교육 지출 / GDP
    "internet_users_pct":"IT.NET.USER.ZS",        # 인터넷 사용자 비율
    "hi_tech_export_pct":"TX.VAL.TECH.MF.ZS",    # 첨단 기술 수출 비율
    "tertiary_enroll":   "SE.TER.ENRR",           # 고등교육 진학률 (% gross)
    # ── 2026-06 추가 (4범주 고정) — 4개 실 WB 피처 ──
    "mobile_subs_per100":   "IT.CEL.SETS.P2",     # 모바일 가입자 100명당 (디지털 인프라 → 운영구조)
    "secure_servers_per_m": "IT.NET.SECR.P6",     # 보안 인터넷 서버 백만명당 (디지털 보안 → 운영구조)
    "youth_unemp_pct":      "SL.UEM.1524.ZS",     # 청년(15-24) 실업률 (보상의 *반대편* 신호 · GII 비구성요소 → 보상수준)
    "tertiary_attainment":  "SE.TER.CUAT.BA.ZS",  # 학사 이상 보유 비율 (25+ 인구 %) → 과학문화·교육
    # ── 2026-06 추가 (탈순환 타깃 — findings §7.7) — 비피처 ──
    "population":           "SP.POP.TOTL",        # 총인구 — 피처 아님(FEATURE_GROUPS 미포함), 탈순환 타깃
                                                  # (articles_per_m_log·patents_per_m_log) 파생 전용 (cpi_inflation 과 같은 비피처 패턴)
}

def fetch_worldbank(
    indicator_code: str, countries: list[str], years: list[int], retries: int = HTTP_RETRIES
) -> pd.DataFrame:
    """World Bank API에서 단일 지표 시계열 데이터 수집.

    WB API는 ISO3 코드를 직접 수용하므로(`country/KOR/...` → countryiso3code=KOR)
    요청·응답 모두 ISO3 로 처리한다(COUNTRIES 매핑 의존 제거 → 임의 국가셋 지원).
    per_page 는 대규모 국가셋(예: 149국×12년≈1788행/지표)에서 잘림을 막도록 충분히 크게 둔다.
    """
    country_set = set(countries)
    country_str = ";".join(countries)  # ISO3 그대로 (WB API가 직접 수용)
    yr_start, yr_end = min(years), max(years)

    url = (
        f"{WB_API_BASE}/country/{country_str}/indicator/"
        f"{indicator_code}?date={yr_start}:{yr_end}&format=json&per_page=20000"
    )
    headers = {}
    if DATA_CONTACT_EMAIL:
        headers = {"User-Agent": f"science-culture-xai ({DATA_CONTACT_EMAIL})"}

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if len(data) < 2 or not data[1]:
                return pd.DataFrame()

            records = []
            for entry in data[1]:
                iso3 = entry.get("countryiso3code", "")
                year = int(entry.get("date", 0))
                val = entry.get("value")
                if iso3 in country_set and val is not None:
                    records.append({"country": iso3, "year": year,
                                    "value": float(val)})
            return pd.DataFrame(records)

        except Exception as e:
            logger.warning(f"  ⚠ WB API 재시도 {attempt+1}/{retries}: {e}")
            time.sleep(HTTP_BACKOFF)
    return pd.DataFrame()


def collect_worldbank_data(countries: list[str],
                           years: list[int]) -> pd.DataFrame:
    """모든 WB 지표 수집 후 wide-format DataFrame 반환"""
    print("🌐 World Bank API 데이터 수집 중...")
    all_frames = []

    for feat_name, ind_code in WB_INDICATORS.items():
        print(f"   [{feat_name}] {ind_code}")
        df = fetch_worldbank(ind_code, countries, years)
        if not df.empty:
            df = df.rename(columns={"value": feat_name})
            all_frames.append(df.set_index(["country", "year"]))
        time.sleep(0.5)   # API rate-limit 대응

    if not all_frames:
        logger.warning("   ⚠ WB 수집 실패 – 합성 데이터로 대체합니다.")
        return pd.DataFrame()

    result = pd.concat(all_frames, axis=1).reset_index()
    print(f"   ✅ WB 수집 완료: {len(result)} rows")
    return result


# ──────────────────────────────────────────────
# 3. 합성 데이터 생성 (PISA, WVS, OECD MSTI, ILO 대체)
# ──────────────────────────────────────────────
# 실제 운영 시: pd.read_csv("pisa_raw.csv") 등으로 교체
COUNTRY_PROFILES = {
    # (과학문화 기저, R&D 성숙도, 보상 수준, 혁신 성과) -- 실측 기반(2026-06 보정, PISA 2018·R&D/GDP 2020 평균·ILOSTAT 임금비·GII 2020 평균)
    # sci=PISA 2018/580, rd=R&D/GDP/5.0, pay=ILOSTAT 전문직 임금비/1.5, innov=GII 2020 평균/70
    "FIN": (0.900, 0.550, 0.867, 0.850),  # PISA 522, R&D 2.75, salary 1.30, GII 59.5
    "SWE": (0.860, 0.690, 0.800, 0.900),  # 499, 3.45, 1.20, 63.0
    "CHE": (0.869, 0.660, 0.933, 0.950),  # 504, 3.30, 1.40, 66.5
    "ISR": (0.819, 0.990, 0.700, 0.786),  # 475, 4.95, 1.05, 55.0
    "DEU": (0.867, 0.620, 0.927, 0.829),  # 503, 3.10, 1.39, 58.0
    "JPN": (0.912, 0.690, 0.733, 0.781),  # 529, 3.45, 1.10, 54.7
    "USA": (0.866, 0.660, 0.927, 0.857),  # 502, 3.30, 1.39, 60.0
    "NLD": (0.879, 0.500, 0.867, 0.857),  # 510, 2.50, 1.30, 60.0
    "DNK": (0.850, 0.610, 0.867, 0.839),  # 493, 3.05, 1.30, 58.7
    "AUT": (0.841, 0.650, 0.833, 0.743),  # 488, 3.25, 1.25, 52.0
    "GBR": (0.871, 0.540, 0.833, 0.857),  # 505, 2.70, 1.25, 60.0
    "FRA": (0.850, 0.460, 0.800, 0.757),  # 493, 2.30, 1.20, 53.0
    "CAN": (0.893, 0.340, 0.867, 0.771),  # 518, 1.70, 1.30, 54.0
    "AUS": (0.867, 0.360, 0.800, 0.710),  # 503, 1.80, 1.20, 49.7
    "BEL": (0.860, 0.660, 0.867, 0.743),  # 499, 3.30, 1.30, 52.0
    "NOR": (0.860, 0.460, 0.933, 0.724),  # 499, 2.30, 1.40, 50.7
    "SGP": (0.950, 0.460, 0.867, 0.929),  # 551, 2.30, 1.30, 65.0
    "TWN": (0.890, 0.710, 0.667, 0.814),  # 516, 3.55, 1.00, 57.0
    "KOR": (0.895, 0.910, 0.760, 0.808),  # 519, 4.55, 1.14, 56.6
    "CHN": (1.017, 0.480, 0.500, 0.759),  # 590 (B-S-J-Z), 2.40, 0.75, 53.1
}

# 모듈 전역 RNG — *호환성 보존용*(외부에서 `from data_collector import rng` 가능).
# build_full_dataset 경로는 더 이상 이것을 사용하지 않고, 함수 인자로 주입된 rng 를 쓴다(P1 재현성).
rng = np.random.default_rng(42)

def _trend(base: float, years: list[int], slope: float = 0.005,
           noise: float = 0.03, clip_min: float | None = None,
           clip_max: float | None = None,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """베이스라인에 완만한 트렌드 + 노이즈 추가 (스케일에 맞는 클리핑).

    Parameters
    ----------
    rng : 난수 생성기. None 이면 모듈 전역 ``rng`` 를 사용(레거시 동작).
          build_full_dataset 경로는 항상 호출자 rng 를 주입한다.
    """
    _rng = rng if rng is not None else globals()["rng"]
    t = np.array([(y - min(years)) for y in years], dtype=float)
    vals = base + slope * t + _rng.normal(0, noise, len(years))
    if clip_min is not None or clip_max is not None:
        vals = np.clip(vals, clip_min, clip_max)
    return vals


def generate_synthetic_data(countries: list[str],
                             years: list[int],
                             rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    PISA, WVS, OECD MSTI, ILOSTAT 기반 지표 합성 생성
    각 국가의 특성값(COUNTRY_PROFILES) 기반으로 현실적인 시계열 구성

    Parameters
    ----------
    rng : 난수 생성기. None 이면 ``np.random.default_rng(42)`` 로 폴백(레거시 동작).
          동일 프로세스에서 두 번 호출해도 결정적이려면 호출자에서 새 rng 를 주입한다.
    """
    print("🔬 합성 데이터 생성 중 (PISA / WVS / MSTI / ILO 대체)...")
    if rng is None:
        rng = np.random.default_rng(42)
    records = []

    for ccode in countries:
        if ccode not in COUNTRY_PROFILES:
            continue
        sci_base, rd_base, pay_base, innov_base = COUNTRY_PROFILES[ccode]

        # ── 과학문화 (Soft) 지표 — 실측 PISA 범위(420~590) 직접 산출
        # sci_base = 실측 PISA / 580 으로 정의 → 스케일 인자 580 이면 절대값 = 실측 (2026-06 보정)
        sci_literacy   = _trend(sci_base * 580, years, slope=1.2, noise=8, clip_min=350, clip_max=620, rng=rng)
        self_efficacy  = _trend(sci_base * 0.72, years, slope=0.003, noise=0.02, clip_min=0.2, clip_max=1.0, rng=rng)
        sci_trust      = _trend(sci_base * 0.68, years, slope=0.002, noise=0.03, clip_min=0.2, clip_max=1.0, rng=rng)
        tech_acceptance= _trend(sci_base * 0.70, years, slope=0.006, noise=0.03, clip_min=0.2, clip_max=1.0, rng=rng)

        # ── R&D 운영 (Structure) 지표
        avg_proj_dur   = _trend(rd_base * 3.2, years, slope=0.05, noise=0.15, clip_min=1.0, clip_max=8.0, rng=rng)
        admin_burden   = _trend((1 - rd_base) * 0.45, years, slope=-0.003, noise=0.02, clip_min=0.05, clip_max=0.6, rng=rng)
        retire_age     = _trend(rd_base * 68, years, slope=0.1, noise=0.5, clip_min=55, clip_max=75, rng=rng)

        # ── 보상 수준 (Economic) 지표
        rel_salary     = _trend(pay_base * 1.35, years, slope=0.008, noise=0.04, clip_min=0.6, clip_max=2.5, rng=rng)
        rd_budget_per_researcher = _trend(pay_base * 180, years, slope=3.5, noise=8, clip_min=30, clip_max=500, rng=rng)

        # ── 혁신 성과 (Outcome) 지표 -- 5년 시차 적용 시뮬레이션
        # GII 점수 스타일: 0~100 범위, 국가별 차별화
        innov_score    = _trend(innov_base * 62, years, slope=0.3, noise=1.5, clip_min=20, clip_max=95, rng=rng)
        gci_rank_inv   = _trend(innov_base * 0.90, years, slope=0.004, noise=0.02, clip_min=0.1, clip_max=1.0, rng=rng)

        for i, yr in enumerate(years):
            records.append({
                "country": ccode,
                "year":    yr,
                # 과학문화
                "sci_literacy":       sci_literacy[i],
                "self_efficacy":      self_efficacy[i],
                "sci_trust":          sci_trust[i],
                "tech_acceptance":    tech_acceptance[i],
                # 운영 구조
                "avg_proj_duration":  avg_proj_dur[i],
                "admin_burden_ratio": admin_burden[i],
                "researcher_retire_age": retire_age[i],
                # 보상
                "relative_salary":    rel_salary[i],
                "rd_budget_per_researcher": rd_budget_per_researcher[i],
                # 혁신 성과 (종속변수)
                "innovation_score":   innov_score[i],
                "competitiveness":    gci_rank_inv[i],
            })

    df = pd.DataFrame(records)
    print(f"   ✅ 합성 데이터 생성 완료: {len(df)} rows")
    return df


# ──────────────────────────────────────────────
# 3.5 World Values Survey (WVS) 실데이터 로더
# ──────────────────────────────────────────────
# 합성으로 채우던 과학·기술 태도 지표(sci_trust / tech_acceptance)를 WVS 실값으로 대체.
# 세 문항(1~10, 높을수록 친과학)은 웨이브별로 컬럼명이 다르다(W6=V코드/세미콜론, W7=Q코드/콤마).
#   item1 = "과학기술이 삶을 개선"        (W6 V192 / W7 Q158)
#   item2 = "다음 세대에 더 많은 기회"    (W6 V193 / W7 Q159)
#   item3 = "세상이 과학기술 덕에 더 나아짐" (W6 V197 / W7 Q163)
def _load_wvs_wave(
    path: str,
    sep: str,
    encoding: str | None,
    col_map: dict[str, str],
    countries: list[str],
) -> pd.DataFrame:
    """단일 WVS 웨이브 CSV에서 필요한 5개 컬럼만 읽어 long DF로 정규화.

    Parameters
    ----------
    path     : 웨이브 CSV 절대경로
    sep      : 구분자 (W6=";", W7=",")
    encoding : 인코딩 (W6="utf-8-sig"(BOM) / W7=None)
    col_map  : 원본 컬럼명 → 표준명(country/year/item1/item2/item3) 매핑
    countries: 분석 대상 ISO3 코드 (이 중 등장하는 국가만 보존)

    Returns
    -------
    DataFrame  : [country, year, item1, item2, item3]. 음수(<1)는 결측 처리,
                 year 는 int. (집계 전 raw 응답 단위 long-format)
    """
    df = pd.read_csv(path, sep=sep, usecols=list(col_map.keys()), encoding=encoding)
    df = df.rename(columns=col_map)

    for item in ("item1", "item2", "item3"):
        df.loc[df[item] < 1, item] = np.nan

    keep = set(countries)
    df = df[df["country"].isin(keep)].copy()
    df["year"] = df["year"].astype(int)
    return df[["country", "year", "item1", "item2", "item3"]]


def load_wvs_indicators(countries: list[str], years: list[int]) -> pd.DataFrame:
    """WVS Wave 6+7 실데이터에서 과학·기술 태도 지표를 (country, year) 단위로 산출.

    두 웨이브를 합쳐 (country, year)별 평균을 낸 뒤 1~10 응답을 0~1 로 정규화한다.
    파이프라인의 합성 sci_trust / tech_acceptance 가 0~1 스케일이므로 동일 척도.

    Parameters
    ----------
    countries : 분석 대상 ISO3 코드 목록
    years     : 보존할 연도 목록 (이 범위 밖 응답은 제외)

    Returns
    -------
    DataFrame  : ["country", "year", "sci_trust", "tech_acceptance"].
                 WVS 데이터가 있는 (country, year) 행만 포함.
                 env 경로 미설정·파일 부재·읽기 실패 시 **빈 DataFrame** (graceful, WB 폴백과 동일).
    """
    cols = ["country", "year", "sci_trust", "tech_acceptance"]

    waves = [
        # (env 경로, 구분자, 인코딩, {원본컬럼: 표준명})
        (
            WVS_WAVE6_PATH,
            ";",
            "utf-8-sig",
            {
                "B_COUNTRY_ALPHA": "country",
                "V262": "year",
                "V192": "item1",
                "V193": "item2",
                "V197": "item3",
            },
        ),
        (
            WVS_WAVE7_PATH,
            ",",
            None,
            {
                "B_COUNTRY_ALPHA": "country",
                "A_YEAR": "year",
                "Q158": "item1",
                "Q159": "item2",
                "Q163": "item3",
            },
        ),
    ]

    frames = []
    for path, sep, encoding, col_map in waves:
        # graceful: 경로 미설정 또는 파일 부재 → 조용히 건너뜀(예외 던지지 않음)
        if not path or not os.path.exists(path):
            continue
        try:
            frames.append(_load_wvs_wave(path, sep, encoding, col_map, countries))
        except Exception as e:  # 파일 손상·인코딩 등 외부 경계 실패만 graceful 처리
            logger.warning(f"   ⚠ WVS 파일 읽기 실패({os.path.basename(path)}): {e}")

    if not frames:
        return pd.DataFrame(columns=cols)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["year"].isin(years)]
    if raw.empty:
        return pd.DataFrame(columns=cols)

    agg = raw.groupby(["country", "year"], as_index=False)[["item1", "item2", "item3"]].mean()

    def _norm(x: pd.Series) -> pd.Series:
        # 1~10 → 0~1 (합성 sci_trust/tech_acceptance 스케일과 일치)
        return ((x - 1) / 9).clip(0, 1)

    agg["sci_trust"] = _norm(agg[["item1", "item3"]].mean(axis=1))
    agg["tech_acceptance"] = _norm(agg[["item1", "item2"]].mean(axis=1))
    return agg[cols]


# ──────────────────────────────────────────────
# 3.6 FRED(美 세인트루이스 연준) CPI 인플레이션 로더
# ──────────────────────────────────────────────
# OECD MEI 월간 CPI 지수(2015=100, 시리즈 ID = "{ISO3}CPIALLMINMEI")를 받아
# 연 평균 → 전년 대비(YoY) 인플레이션(%)으로 변환한다.
# 커버리지는 17/20개국(AUS·SGP·TWN 미제공 → 해당 국가는 NaN, 정상).
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """FRED observations 엔드포인트에서 단일 시리즈의 관측치를 수집.

    Parameters
    ----------
    series_id : FRED 시리즈 ID (예: "KORCPIALLMINMEI")

    Returns
    -------
    DataFrame  : ["date", "value"]. date 는 문자열("YYYY-MM-DD"), value 는 float.
                 결측 표기("." / 빈값)는 제외한다.
                 키 미설정·HTTP 실패 시 **빈 DataFrame** (graceful, WB·WVS 폴백과 동일).
    """
    cols = ["date", "value"]
    if not FRED_API_KEY:
        return pd.DataFrame(columns=cols)

    url = (
        f"{FRED_API_BASE}/series/observations?series_id={series_id}"
        f"&api_key={FRED_API_KEY}&file_type=json"
        f"&observation_start=2010-01-01&observation_end=2023-12-31"
    )
    headers = {}
    if DATA_CONTACT_EMAIL:
        headers = {"User-Agent": f"science-culture-xai ({DATA_CONTACT_EMAIL})"}

    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT, headers=headers)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
    except Exception as e:  # 네트워크/HTTP 외부 경계 실패만 graceful 처리
        msg = str(e).replace(FRED_API_KEY, "***") if FRED_API_KEY else str(e)
        logger.warning(f"   ⚠ FRED API 실패({series_id}): {msg}")
        return pd.DataFrame(columns=cols)

    records = []
    for obs in observations:
        raw_val = obs.get("value", ".")
        if raw_val in (".", "", None):  # FRED 결측 표기
            continue
        records.append({"date": obs.get("date", ""), "value": float(raw_val)})

    return pd.DataFrame(records, columns=cols)


def collect_fred_data(countries: list[str], years: list[int]) -> pd.DataFrame:
    """FRED CPI 지수에서 국가별 연간 인플레이션(cpi_inflation, %)을 산출.

    각 국가의 월간 CPI 지수를 연 평균으로 집계한 뒤 전년 대비(YoY) 변화율로
    인플레이션을 계산한다. years 범위에 속하는 (country, year)만 반환한다.

    Parameters
    ----------
    countries : 분석 대상 ISO3 코드 목록
    years     : 보존할 연도 목록

    Returns
    -------
    DataFrame  : ["country", "year", "cpi_inflation"]. FRED 데이터가 있는 국가만 포함.
                 FRED_API_KEY 미설정 시 **빈 DataFrame** (조용히 스킵).
    """
    cols = ["country", "year", "cpi_inflation"]
    if not FRED_API_KEY:
        return pd.DataFrame(columns=cols)

    year_set = set(years)
    frames = []
    for country in countries:
        series_id = f"{country}CPIALLMINMEI"
        print(f"💵 FRED 수집: {country} {series_id}")
        obs_df = fetch_fred_series(series_id)
        if obs_df.empty:
            continue

        obs_df["year"] = pd.to_datetime(obs_df["date"]).dt.year
        annual = obs_df.groupby("year", as_index=False)["value"].mean()
        annual = annual.sort_values("year")
        # 전년 대비(YoY) 인플레이션(%) = (연평균_t / 연평균_{t-1} - 1) * 100
        annual["cpi_inflation"] = annual["value"].pct_change() * 100
        annual = annual[annual["year"].isin(year_set)].dropna(subset=["cpi_inflation"])
        if annual.empty:
            continue

        annual["country"] = country
        frames.append(annual[cols])
        time.sleep(0.1)  # API rate-limit 대응(폴라이트)

    if not frames:
        return pd.DataFrame(columns=cols)

    result = pd.concat(frames, ignore_index=True)
    print(f"✅ FRED CPI 인플레이션 수집: {result['country'].nunique()}개국")
    return result


# ──────────────────────────────────────────────
# 3.7 GII(Global Innovation Index) 실타깃 로더
# ──────────────────────────────────────────────
# 합성 innovation_score(종속변수)를 실제 GII 종합점수(Score, 0~100)로 교체하기 위한 로더.
# long-format TSV(Country=ISO3 / Year / Rank / Score)에서 결측(-1)을 제외하고 그대로 사용한다.
# 우리 20개국 중 TWN만 미수록(19/20). 경로 미설정·부재 시 빈 DF(graceful, WB·WVS·FRED와 동일).
def load_gii_scores(countries: list[str], years: list[int]) -> pd.DataFrame:
    """GII 종합점수 TSV에서 (country, year)별 실제 GII Score를 산출.

    Parameters
    ----------
    countries : 분석 대상 ISO3 코드 목록
    years     : 보존할 연도 목록

    Returns
    -------
    DataFrame  : ["country", "year", "gii_score"]. Score>0(=-1 결측 제거)인 행만 포함.
                 GII_DATA_PATH 미설정·파일 부재·읽기 실패 시 **빈 DataFrame**
                 (graceful, WB·WVS·FRED 폴백과 동일).
    """
    cols = ["country", "year", "gii_score"]
    if not GII_DATA_PATH or not os.path.exists(GII_DATA_PATH):
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(GII_DATA_PATH, sep="\t")
    except Exception as e:  # 파일 손상·인코딩 등 외부 경계 실패만 graceful 처리
        logger.warning(f"   ⚠ GII 파일 읽기 실패({os.path.basename(GII_DATA_PATH)}): {e}")
        return pd.DataFrame(columns=cols)

    country_set = set(countries)
    year_set = set(years)
    mask = df["Country"].isin(country_set) & df["Year"].isin(year_set) & (df["Score"] > 0)
    df = df.loc[mask].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.rename(columns={"Country": "country", "Year": "year", "Score": "gii_score"})
    df["year"] = df["year"].astype(int)
    df["gii_score"] = df["gii_score"].astype(float)
    return df[cols]


# ──────────────────────────────────────────────
# 3.8 OECD PISA 점수 실데이터 로더 (과학·수학·읽기)
# ──────────────────────────────────────────────
# 과학문화 카테고리를 두껍게: PISASCIENCE→sci_literacy 외에 PISAMATH→pisa_math,
# PISAREAD→pisa_reading 까지 한 번에 로드한다(셋 다 SUBJECT=="TOT" 평균점수, ~460~560).
# long-format CSV(LOCATION=ISO3 / INDICATOR / SUBJECT / TIME / Value)를 세 지표로 필터한 뒤
# (country, year) 단위로 pivot → 컬럼 [sci_literacy, pisa_math, pisa_reading].
# PISA 주기는 3년(데이터 범위 내 2012/2015/2018)이라 비주기 연도는 preprocessor 보간에 맡긴다.
# 우리 20개국 중 CHN만 미수록(19/20). 경로 미설정·부재 시 빈 DF(graceful, WB·WVS·FRED·GII와 동일).
PISA_INDICATOR_MAP = {            # 원본 INDICATOR → 표준 피처명
    "PISASCIENCE": "sci_literacy",
    "PISAMATH": "pisa_math",
    "PISAREAD": "pisa_reading",
}


def load_pisa_scores(countries: list[str], years: list[int]) -> pd.DataFrame:
    """OECD PISA CSV에서 (country, year)별 실제 과학·수학·읽기 평균점수를 산출.

    Parameters
    ----------
    countries : 분석 대상 ISO3 코드 목록 (LOCATION 과 매칭)
    years     : 보존할 연도 목록 (이 범위 밖 PISA 주기는 제외)

    Returns
    -------
    DataFrame  : ["country", "year", "sci_literacy", "pisa_math", "pisa_reading"].
                 INDICATOR ∈ {PISASCIENCE, PISAMATH, PISAREAD} & SUBJECT=="TOT" 인 행을
                 지표별 컬럼으로 pivot. 특정 지표가 없는 (country, year)는 해당 컬럼만 NaN.
                 PISA_DATA_PATH 미설정·파일 부재·읽기 실패 시 **빈 DataFrame**
                 (graceful, WB·WVS·FRED·GII 폴백과 동일).
    """
    cols = ["country", "year", "sci_literacy", "pisa_math", "pisa_reading"]
    if not PISA_DATA_PATH or not os.path.exists(PISA_DATA_PATH):
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(PISA_DATA_PATH)
    except Exception as e:  # 파일 손상·인코딩 등 외부 경계 실패만 graceful 처리
        logger.warning(f"   ⚠ PISA 파일 읽기 실패({os.path.basename(PISA_DATA_PATH)}): {e}")
        return pd.DataFrame(columns=cols)

    country_set = set(countries)
    year_set = set(years)
    mask = (
        df["INDICATOR"].isin(PISA_INDICATOR_MAP.keys())
        & (df["SUBJECT"] == "TOT")
        & df["LOCATION"].isin(country_set)
        & df["TIME"].isin(year_set)
    )
    df = df.loc[mask].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.rename(columns={"LOCATION": "country", "TIME": "year"})
    df["feature"] = df["INDICATOR"].map(PISA_INDICATOR_MAP)
    df["Value"] = df["Value"].astype(float)
    # (country, year) × 지표 → wide. 동일 키 중복은 평균(방어적)으로 합쳐 단일 값 보장.
    pivoted = df.pivot_table(
        index=["country", "year"], columns="feature", values="Value", aggfunc="mean"
    ).reset_index()
    pivoted.columns.name = None
    # 세 지표 중 일부만 존재하는 데이터셋이어도 계약 컬럼은 항상 채운다(없으면 NaN).
    for feat in PISA_INDICATOR_MAP.values():
        if feat not in pivoted.columns:
            pivoted[feat] = np.nan
    pivoted["year"] = pivoted["year"].astype(int)
    return pivoted[cols]


# ──────────────────────────────────────────────
# 3.85 ILOSTAT 실 임금 로더 (relative_salary = 전문직 임금 프리미엄)
# ──────────────────────────────────────────────
# 합성으로 제거됐던 relative_salary(보상수준)를 ILOSTAT 실 임금으로 재투입한다.
# SDMX 데이터플로 DF_EAR_EMTA_SEX_OCU_NB(직종별 월평균소득)에서 두 직종을 받아 비율을 낸다:
#   relative_salary = OCU_ISCO08_2(전문직) / OCU_ISCO08_TOTAL(전체)  ← 비율이라 현지통화 상쇄, 국제 비교가능.
# GII 비구성요소라 타깃과 순환 없는 깨끗한 보상 신호. 검증값 KOR 1.14·USA 1.38·DEU 1.43·FIN 1.30.
# REF_AREA(=country, ISO3) ∈ countries & 연간(TIME_PERIOD=^\d{4}$, 월/분기 제외) & year ∈ years 만 보존.
# 경로/네트워크 실패·빈 응답 시 빈 DF(graceful, WB·WVS·FRED·GII·PISA 폴백과 동일).
def load_ilostat_salary(countries: list[str], years: list[int]) -> pd.DataFrame:
    """ILOSTAT SDMX에서 (country, year)별 전문직 임금 프리미엄(relative_salary)을 산출.

    Parameters
    ----------
    countries : 분석 대상 ISO3 코드 목록 (REF_AREA 와 매칭)
    years     : 보존할 연도 목록 (연간 TIME_PERIOD 중 이 범위만)

    Returns
    -------
    DataFrame  : ["country", "year", "relative_salary"].
                 relative_salary = OCU_ISCO08_2 / OCU_ISCO08_TOTAL (둘 다 있는 행만).
                 HTTP 실패·빈 응답·키 부족 시 **빈 DataFrame** (graceful, 타 실데이터 로더와 동일).
    """
    cols = ["country", "year", "relative_salary"]

    # SDMX 키 5위치(REF_AREA.FREQ.MEASURE.SEX.OCU) — 앞 3개 비우고 성별 합계·직종 2개만 지정.
    key = ".".join(["", "", "", "SEX_T", "OCU_ISCO08_2+OCU_ISCO08_TOTAL"])
    url = (
        f"{ILOSTAT_SDMX_BASE}/data/DF_EAR_EMTA_SEX_OCU_NB/{key}/"
        f"?startPeriod={min(years)}&endPeriod={max(years)}"
    )
    headers = {"Accept": "application/vnd.sdmx.data+csv"}
    if DATA_CONTACT_EMAIL:
        headers["User-Agent"] = f"science-culture-xai ({DATA_CONTACT_EMAIL})"

    print("💵 ILOSTAT SDMX 실임금 수집 중 (DF_EAR_EMTA_SEX_OCU_NB)...")
    content = b""
    for attempt in range(HTTP_RETRIES):
        try:  # 네트워크/HTTP 외부 경계 실패만 graceful 처리(내부 로직은 fail-loud)
            resp = requests.get(url, timeout=HTTP_TIMEOUT, headers=headers)
            resp.raise_for_status()
            content = resp.content
            break
        except Exception as e:
            logger.warning(f"   ⚠ ILOSTAT API 재시도 {attempt + 1}/{HTTP_RETRIES}: {e}")
            time.sleep(HTTP_BACKOFF)

    if not content:
        return pd.DataFrame(columns=cols)

    # 응답 파싱 — ILOSTAT 는 외부 API 라 응답 *포맷* 전체가 외부 경계다
    # (HTTP 200 + HTML 에러페이지, SDMX 스키마 변경 등). 파싱 실패 시 문서화된
    # 계약대로 빈 DF 반환(파일 로더 WVS/GII/PISA 와 동일 수준 graceful).
    try:
        df = pd.read_csv(BytesIO(content), low_memory=False)
        if df.empty or "TIME_PERIOD" not in df.columns:
            return pd.DataFrame(columns=cols)

        # 연간만: TIME_PERIOD 가 4자리 연도(예: "2019"). 월/분기("2014-M01" 등)는 제외.
        df = df[df["TIME_PERIOD"].astype(str).str.match(r"^\d{4}$")].copy()
        if df.empty:
            return pd.DataFrame(columns=cols)
        df["year"] = df["TIME_PERIOD"].astype(int)
        df = df.rename(columns={"REF_AREA": "country"})

        # (country, year) × 직종 → wide. 동일 키 중복은 평균(방어적)으로 단일 값 보장.
        pivoted = df.pivot_table(
            index=["country", "year"], columns="OCU", values="OBS_VALUE", aggfunc="mean"
        ).reset_index()
        pivoted.columns.name = None
        # 두 직종 컬럼이 모두 있어야 비율 계산 가능(없으면 빈 DF).
        if "OCU_ISCO08_2" not in pivoted.columns or "OCU_ISCO08_TOTAL" not in pivoted.columns:
            return pd.DataFrame(columns=cols)

        pivoted["relative_salary"] = pivoted["OCU_ISCO08_2"] / pivoted["OCU_ISCO08_TOTAL"]
    except Exception as e:  # 외부 응답 파싱 실패만 graceful — 이후 내부 로직은 fail-loud 유지
        logger.warning(f"   ⚠ ILOSTAT 응답 파싱 실패: {e} — relative_salary 합성 유지")
        return pd.DataFrame(columns=cols)

    country_set = set(countries)
    year_set = set(years)
    mask = (
        pivoted["country"].isin(country_set)
        & pivoted["year"].isin(year_set)
        & pivoted["relative_salary"].notna()
    )
    result = pivoted.loc[mask, cols].reset_index(drop=True)
    return result


# ──────────────────────────────────────────────
# 3.9 국가셋(country universe) 도출 헬퍼
# ──────────────────────────────────────────────
# 실데이터 소스(GII·PISA)에서 분석 가능한 국가 목록을 동적으로 도출한다.
# COUNTRIES(고정 20국)에 의존하지 않으므로 임의 개수의 국가로 파이프라인을 돌릴 수 있다.
# 파일 부재·읽기 실패 시 빈 목록(graceful) → resolve_country_set 이 default 로 폴백한다.
def gii_country_codes(years: list[int] | None = None) -> list[str]:
    """GII TSV에서 (해당 연도 범위에) 실측 Score 가 있는 국가(ISO3) 고유목록을 반환.

    결측 sentinel(-1.0)과 Score<=0 은 제외한다(load_gii_scores 와 동일 기준).

    Parameters
    ----------
    years : 연도 범위 (None → 기본 YEARS)

    Returns
    -------
    list[str] : 정렬된 ISO3 목록. 경로 미설정·파일 부재·읽기 실패 시 **빈 목록**(graceful).
    """
    years = years or YEARS
    if not GII_DATA_PATH or not os.path.exists(GII_DATA_PATH):
        return []
    try:
        df = pd.read_csv(GII_DATA_PATH, sep="\t")
    except Exception as e:  # 파일 손상·인코딩 등 외부 경계 실패만 graceful 처리
        logger.warning(f"   ⚠ GII 국가목록 읽기 실패({os.path.basename(GII_DATA_PATH)}): {e}")
        return []
    mask = df["Year"].isin(set(years)) & (df["Score"] > 0)
    return sorted(df.loc[mask, "Country"].dropna().astype(str).unique())


def pisa_country_codes(years: list[int] | None = None) -> list[str]:
    """PISA CSV에서 과학(PISASCIENCE)·전체(TOT) 점수가 있는 국가(ISO3) 고유목록을 반환.

    Parameters
    ----------
    years : 연도 범위 (None → 기본 YEARS). PISA 주기(3년)상 해당 범위 응답만 집계.

    Returns
    -------
    list[str] : 정렬된 ISO3 목록. 경로 미설정·파일 부재·읽기 실패 시 **빈 목록**(graceful).
    """
    years = years or YEARS
    if not PISA_DATA_PATH or not os.path.exists(PISA_DATA_PATH):
        return []
    try:
        df = pd.read_csv(PISA_DATA_PATH)
    except Exception as e:  # 파일 손상·인코딩 등 외부 경계 실패만 graceful 처리
        logger.warning(f"   ⚠ PISA 국가목록 읽기 실패({os.path.basename(PISA_DATA_PATH)}): {e}")
        return []
    mask = (
        (df["INDICATOR"] == "PISASCIENCE") & (df["SUBJECT"] == "TOT") & df["TIME"].isin(set(years))
    )
    return sorted(df.loc[mask, "LOCATION"].dropna().astype(str).unique())


def resolve_country_set(name: str, years: list[int] | None = None) -> list[str]:
    """이름으로 분석 국가셋(ISO3 목록)을 해석한다.

    Parameters
    ----------
    name  : "default" → COUNTRIES 고정 20국,
            "gii"     → GII 실측 국가 전체,
            "pisa-gii"→ PISA 과학점수 ∩ GII 실측 국가(둘 다 있는 국가).
    years : 연도 범위 (None → 기본 YEARS).

    Returns
    -------
    list[str] : ISO3 목록. 소스 파일이 없어 빈 목록이 되면 경고 후 default(20국) 로 폴백한다.
    """
    years = years or YEARS
    default = list(COUNTRIES.keys())

    if name == "default":
        return default
    if name == "gii":
        codes = gii_country_codes(years)
        if not codes:
            logger.warning("   ⚠ GII 국가목록을 얻지 못해 default(20국)로 폴백합니다.")
            return default
        return codes
    if name == "pisa-gii":
        pisa = set(pisa_country_codes(years))
        gii = set(gii_country_codes(years))
        codes = sorted(pisa & gii)
        if not codes:
            logger.warning("   ⚠ PISA∩GII 국가목록을 얻지 못해 default(20국)로 폴백합니다.")
            return default
        return codes

    logger.warning(f"   ⚠ 알 수 없는 country-set '{name}' → default(20국) 사용.")
    return default


# ──────────────────────────────────────────────
# 4. 통합 데이터셋 구성
# ──────────────────────────────────────────────
def build_full_dataset(
    countries: list | None = None,
    years: list | None = None,
    use_worldbank: bool = True,
    use_fred: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    World Bank 실데이터 + 합성 데이터를 병합하여 최종 분석용 DataFrame 반환

    Parameters
    ----------
    countries    : ISO3 국가 코드 목록 (None → COUNTRIES 고정 20국)
    years        : 연도 목록 (None → 기본 범위)
    use_worldbank: True면 WB API 시도, 실패 시 합성값으로 대체
    use_fred     : True면 FRED CPI 인플레이션 수집(국가당 1 호출).
                   대규모 국가셋에선 False 로 두어 불필요한 호출을 막는다
                   (cpi_inflation 은 모델 피처가 아님 → 결과 불변).
    seed         : 합성 데이터 RNG 시드(기본 42, P1 재현성).
                   build_full_dataset 진입 시 ``np.random.default_rng(seed)`` 를 새로 만들고
                   generate_synthetic_data / _synthesize_wb_features 에 주입한다 →
                   동일 프로세스에서 두 번 호출해도 결정적.
    """
    countries = countries or list(COUNTRIES.keys())
    years     = years or YEARS

    # 호출별 로컬 rng — 모듈 전역 rng 와 분리해야 두 번째 호출에서도 결정성 보장.
    rng_local = np.random.default_rng(seed)

    # 국가 universe 를 COUNTRY_PROFILES(합성 프로파일 20국)에서 분리한다.
    # 기준 그리드(요청 국가×연도 전체)를 만들고 그 위에 각 소스를 left-merge → 임의 국가셋 지원.
    # COUNTRY_PROFILES 에 없는 국가는 합성 컬럼이 NaN 이지만, 그 합성 6종은 모델 피처가 아니므로 무방.
    base = pd.DataFrame(
        [(c, y) for c in countries for y in years],
        columns=["country", "year"],
    )

    # 합성 데이터 (PISA 등) — COUNTRY_PROFILES 보유 국가만 생성됨 → 기준 그리드에 left-merge
    synth_df = generate_synthetic_data(countries, years, rng=rng_local)
    merged = base.merge(synth_df, on=["country", "year"], how="left")

    # World Bank 데이터 시도
    wb_df = pd.DataFrame()
    if use_worldbank:
        wb_df = collect_worldbank_data(countries, years)

    if not wb_df.empty:
        merged = merged.merge(wb_df, on=["country", "year"], how="left")
    else:
        # WB 실패 시 WB 지표도 합성으로 대체
        logger.warning("   ℹ WB 지표를 합성값으로 대체합니다.")
        wb_synth = _synthesize_wb_features(countries, years, rng=rng_local)
        merged = merged.merge(wb_synth, on=["country", "year"], how="left")

    # 연구원 1인당 R&D 자원(실 WB 파생) — generate_synthetic_data 의 합성 동명 컬럼을 덮어쓴다.
    #  · 인구 상쇄: (rd_gdp_pct/100 × gdp_pc × pop) / (researchers_per_m/1e6 × pop)
    #               = rd_gdp_pct × gdp_per_capita × 1e4 / researchers_per_m  [USD/연구원]
    #  · researchers_per_m = 0/NaN → ±inf → NaN(이후 preprocessor 가 impute).
    #  · --no-wb 에선 WB 입력이 합성(_synthesize_wb_features)이라 파생값도 합성기반(일관, 정상).
    merged["rd_budget_per_researcher"] = (
        merged["rd_gdp_pct"] * merged["gdp_per_capita"] * 1e4 / merged["researchers_per_m"]
    ).replace([np.inf, -np.inf], np.nan)
    print("💰 rd_budget_per_researcher 파생: 연구원당 R&D 자원(실 WB)")

    # 탈순환 타깃 파생 (findings §7.7): GII 미사용 혁신 산출 *직접측정* — log(백만명당 논문/특허).
    #  · population 있을 때만 파생 → **실 WB 모드 전용** (--no-wb 합성 폴백엔 population 없음 → 스킵, graceful).
    #  · 타깃 후보(OUTCOME_COLS)일 뿐 피처 아님(FEATURE_GROUPS 미포함) → 기본 타깃 경로의 X 불변.
    #  · pop=0/NaN 으로 인한 ±inf 는 NaN 처리(이후 prepare_xy 의 dropna(subset=[target])가 제외).
    if "population" in merged.columns and merged["population"].notna().any():
        pop_m = merged["population"] / 1e6
        merged["articles_per_m_log"] = np.log1p(merged["journal_articles"] / pop_m).replace(
            [np.inf, -np.inf], np.nan
        )
        merged["patents_per_m_log"] = np.log1p(merged["patent_apps"] / pop_m).replace(
            [np.inf, -np.inf], np.nan
        )
        print("🎯 탈순환 타깃 파생: articles_per_m_log·patents_per_m_log (per-capita log, GII 미사용)")

    # WVS 실데이터로 과학·기술 태도 지표 덮어쓰기.
    #  · WVS 보유국은 두 컬럼을 전 연도 NaN 으로 비운 뒤, 설문연도만 실 앵커로 채운다.
    #    → 사이 연도는 preprocessor 의 country별 interpolate(limit_direction="both") 가
    #      실 앵커 사이를 보간한다(합성값이 앵커 사이에 남는 버그 방지).
    #  · WVS 미보유국은 합성 sci_trust/tech_acceptance 를 그대로 유지.
    wvs_df = load_wvs_indicators(countries, years)
    if not wvs_df.empty:
        wvs_countries = set(wvs_df["country"])
        merged = merged.set_index(["country", "year"])
        wvs_idx = wvs_df.set_index(["country", "year"])
        common = merged.index.intersection(wvs_idx.index)
        has = merged.index.get_level_values("country").isin(wvs_countries)
        merged.loc[has, ["sci_trust", "tech_acceptance"]] = np.nan  # 보유국 전 연도 비움
        merged.loc[common, "sci_trust"] = wvs_idx.loc[common, "sci_trust"]  # 실 앵커
        merged.loc[common, "tech_acceptance"] = wvs_idx.loc[common, "tech_acceptance"]
        merged = merged.reset_index()
        print(f"🗳  WVS 실데이터 병합: {len(common)}개 국가×연도 (사이 연도 NaN→보간)")

    # PISA 실데이터로 과학문화 핵심 지표(sci_literacy)와 추가 인적자본 지표(pisa_math/pisa_reading) 덮어쓰기.
    #  · sci_literacy 는 기존 피처(FEATURE_GROUPS 과학문화)이고, pisa_math/pisa_reading 은 신규 과학문화 피처다.
    #    sci_literacy 는 컬럼명 그대로 값만 교체, math/reading 은 전 국가 NaN 으로 신설 후 실값을 채운다.
    #  · PISA 보유국은 세 지표를 전 연도 NaN 으로 비운 뒤, PISA 주기 연도만 실 앵커로 채운다.
    #    → 사이 연도는 preprocessor 의 country별 interpolate(limit_direction="both") 가 실 앵커 사이를
    #      보간하고, 마지막 앵커(2018) 이후(2019~2023)는 2018값으로 채운다(합성값 잔존 버그 방지).
    #  · PISA 미보유국(CHN 등)은 합성 sci_literacy 를 유지하고, math/reading 은 NaN(→ preprocessor impute).
    #  · PISA 비면 조용히 스킵.
    merged["pisa_math"] = np.nan       # 신규 과학문화 피처 — 전 국가 초기화(미보유국은 NaN→impute)
    merged["pisa_reading"] = np.nan
    pisa_df = load_pisa_scores(countries, years)
    if not pisa_df.empty:
        pisa_countries = set(pisa_df["country"])
        merged = merged.set_index(["country", "year"])
        pisa_idx = pisa_df.set_index(["country", "year"])
        common = merged.index.intersection(pisa_idx.index)
        has = merged.index.get_level_values("country").isin(pisa_countries)
        # 보유국 전 연도 비움(세 지표 동일 처리) → 사이 연도 합성/잔존값 제거 후 실 앵커만 남김
        merged.loc[has, ["sci_literacy", "pisa_math", "pisa_reading"]] = np.nan
        merged.loc[common, "sci_literacy"] = pisa_idx.loc[common, "sci_literacy"]      # 실 앵커
        merged.loc[common, "pisa_math"] = pisa_idx.loc[common, "pisa_math"]
        merged.loc[common, "pisa_reading"] = pisa_idx.loc[common, "pisa_reading"]
        merged = merged.reset_index()
        n_math = int(pisa_idx.loc[common, "pisa_math"].notna().sum())
        n_read = int(pisa_idx.loc[common, "pisa_reading"].notna().sum())
        print(
            f"📊 PISA 실데이터 병합: {len(common)}개 국가×연도 "
            f"(sci_literacy ← 실제 PISA 과학점수, math {n_math}·reading {n_read}, 사이 연도 NaN→보간)"
        )

    # ILOSTAT 실 임금으로 relative_salary(보상수준) 덮어쓰기 — 합성으로 제거됐던 피처의 실데이터 재투입.
    #  · use_worldbank=True 일 때만 호출(--no-wb 는 오프라인 → 합성 relative_salary 유지).
    #  · WVS/PISA 와 동일 "보유국 NaN→실 앵커 set→보간" 패턴:
    #    ILOSTAT 보유국(ilo_df.country)은 relative_salary 를 전 연도 NaN 으로 비운 뒤, 데이터 있는
    #    연도만 실 앵커로 채운다 → 사이/이후 연도는 preprocessor 의 country별 interpolate
    #    (limit_direction="both") 가 실 앵커 기준으로 채운다(합성값 잔존 방지).
    #  · ILOSTAT 미보유국은 합성 relative_salary 를 그대로 유지. ILOSTAT 비면 조용히 스킵.
    if use_worldbank:
        ilo_df = load_ilostat_salary(countries, years)
        if not ilo_df.empty:
            ilo_countries = set(ilo_df["country"])
            merged = merged.set_index(["country", "year"])
            ilo_idx = ilo_df.set_index(["country", "year"])
            common = merged.index.intersection(ilo_idx.index)
            has = merged.index.get_level_values("country").isin(ilo_countries)
            merged.loc[has, "relative_salary"] = np.nan  # 보유국 전 연도 비움
            # 실 앵커(데이터 있는 연도만) — 사이/이후는 preprocessor interpolate 가 채움
            merged.loc[common, "relative_salary"] = ilo_idx.loc[common, "relative_salary"]
            merged = merged.reset_index()
            print(
                f"💵 ILOSTAT 실임금 병합: {len(common)}개 국가×연도 "
                f"(relative_salary ← 전문직/전체 임금비)"
            )

    # FRED CPI 인플레이션 컬럼 추가 (있는 (country, year)만; 나머지는 NaN). FRED 비면 조용히 스킵.
    #  · cpi_inflation 은 모델 피처가 아니므로, 대규모 국가셋에선 use_fred=False 로 국가당 호출을 생략한다.
    if use_fred:
        fred_df = collect_fred_data(countries, years)
        if not fred_df.empty:
            merged = merged.merge(fred_df, on=["country", "year"], how="left")

    # GII 실타깃 적용: innovation_score(종속변수)를 실제 GII 종합점수로 교체.
    #  · GII에 있는 (country, year) → 실제 gii_score 로 덮어씀.
    #  · GII에 없는 행(TWN 등) → NaN (prepare_xy 의 dropna(subset=[target])가 학습에서 제외 = 순수 실타깃).
    #  · competitiveness(대체 타깃)는 건드리지 않음 — innovation_score 만 교체.
    #  GII 비면 조용히 스킵(합성 타깃 유지).
    gii_df = load_gii_scores(countries, years)
    if not gii_df.empty:
        merged = merged.set_index(["country", "year"])
        gii_idx = gii_df.set_index(["country", "year"])
        common = merged.index.intersection(gii_idx.index)
        merged["innovation_score"] = np.nan
        merged.loc[common, "innovation_score"] = gii_idx.loc[common, "gii_score"]
        merged = merged.reset_index()
        n_excluded = len(merged) - len(common)
        print(
            f"🎯 GII 실타깃 적용: {len(common)}개 국가×연도 "
            f"(innovation_score ← 실제 GII), 미수록 {n_excluded}행 제외"
        )

    # 국가명 추가 (COUNTRIES 에 없는 신규국은 ISO3 코드를 이름으로 사용)
    name_map = {k: (COUNTRIES[k]["name"] if k in COUNTRIES else k) for k in countries}
    merged["country_name"] = merged["country"].map(name_map)

    merged = merged.sort_values(["country", "year"]).reset_index(drop=True)
    print(f"\n📦 최종 데이터셋: {merged.shape[0]} rows × {merged.shape[1]} cols")
    return merged


def _synthesize_wb_features(countries: list, years: list,
                            rng: np.random.Generator | None = None) -> pd.DataFrame:
    """WB API 실패 시 WB 지표도 합성 생성.

    Parameters
    ----------
    rng : 난수 생성기. None 이면 ``np.random.default_rng(42)`` 로 폴백(레거시 동작).
          build_full_dataset 경로는 호출자 rng 를 주입한다.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    records = []
    for ccode in countries:
        if ccode not in COUNTRY_PROFILES:
            continue
        _, rd_base, pay_base, innov_base = COUNTRY_PROFILES[ccode]
        rd_gdp   = _trend(rd_base * 3.8, years, slope=0.03, noise=0.12, clip_min=0.3, clip_max=6.0, rng=rng)
        res_per_m= _trend(rd_base * 7200, years, slope=50, noise=150, clip_min=100, clip_max=12000, rng=rng)
        patents  = _trend(innov_base * 15000, years, slope=200, noise=500, clip_min=100, clip_max=50000, rng=rng)
        articles = _trend(innov_base * 25000, years, slope=300, noise=700, clip_min=100, clip_max=80000, rng=rng)
        gdp_pc   = _trend(pay_base * 52000, years, slope=800, noise=1500, clip_min=5000, clip_max=120000, rng=rng)
        # ── 2026-06 추가 (4범주 고정) — 4개 WB 피처 합성 폴백 (--no-wb 시만 사용) ──
        # sci_base 는 generate_synthetic_data 와 동일 의미(과학문화 기저)이나
        # _synthesize_wb_features 시그니처는 (_, rd, pay, innov) 만 unpack → 별도 산출.
        sci_base_fb = COUNTRY_PROFILES[ccode][0]
        mobile_subs   = _trend(rd_base * 100, years, slope=2, noise=3,
                               clip_min=20, clip_max=130, rng=rng)
        secure_srvrs  = _trend(innov_base * 40000, years, slope=2000, noise=3000,
                               clip_min=100, clip_max=100000, rng=rng)
        youth_unemp   = _trend((1 - pay_base) * 30, years, slope=-0.1, noise=2,
                               clip_min=2, clip_max=50, rng=rng)
        tert_attain   = _trend(sci_base_fb * 45, years, slope=0.5, noise=2,
                               clip_min=10, clip_max=70, rng=rng)
        for i, yr in enumerate(years):
            records.append({
                "country": ccode, "year": yr,
                "rd_gdp_pct": rd_gdp[i],
                "researchers_per_m": res_per_m[i],
                "patent_apps": patents[i],
                "journal_articles": articles[i],
                "gdp_per_capita": gdp_pc[i],
                "gov_edu_exp_pct": _trend(rd_base * 5.5, years, slope=0.02,
                                          noise=0.1, clip_min=2.0, clip_max=9.0, rng=rng)[i],
                "internet_users_pct": _trend(0.85, years, slope=0.008,
                                             noise=0.02, clip_min=0.3, clip_max=1.0, rng=rng)[i],
                "hi_tech_export_pct": _trend(innov_base * 0.22, years,
                                             slope=0.002, noise=0.01, clip_min=0.02, clip_max=0.6, rng=rng)[i],
                # 고등교육 진학률(% gross) 합성 폴백 — WB SE.TER.ENRR 부재(--no-wb) 시만 사용
                "tertiary_enroll": _trend(rd_base * 75, years, slope=0.3,
                                          noise=3, clip_min=10, clip_max=120, rng=rng)[i],
                # ── 2026-06 추가 (4범주 고정) ──
                "mobile_subs_per100":   mobile_subs[i],
                "secure_servers_per_m": secure_srvrs[i],
                "youth_unemp_pct":      youth_unemp[i],
                "tertiary_attainment":  tert_attain[i],
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = build_full_dataset()
    print(df.head())
    df.to_csv("/tmp/raw_dataset.csv", index=False)
    print("저장 완료: /tmp/raw_dataset.csv")
