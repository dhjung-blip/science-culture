"""config.py — 파이프라인 공용 상수 단일화 (스테이지 아님 · leaf 모듈)

- 스테이지 모듈이 import 해도 불변식 위반 아님 (architecture.md §3 예외).
- third-party import 금지 (ML 스택 없는 테스트/훅에서 import 됨).
- 4범주 키 고정(category-frame-locked): sci_culture/hard_rd/structure/economic 변경 금지.
- v1 재현 계약: preprocessor.FEATURE_GROUPS·ALL_FEATURES / shap_analyzer.CATEGORY_MAP 은
  replicate_paper.py 가 monkey-patch 하는 attribute 이름이다. 스테이지는 반드시
  `from config import X` (이름 바인딩) 로 가져와야 하며 함수 본문에서 `config.X` 직접 참조 금지.
"""

# ──────────────────────────────────────────────
# 1. 피처 그룹 정의 (← preprocessor.py)
# ──────────────────────────────────────────────
# 순수 합성 피처는 모델에서 제외(연구 신뢰성, 공개 실데이터 출처 없음):
#   self_efficacy · avg_proj_duration · admin_burden_ratio · researcher_retire_age
# → data_collector 는 여전히 생성하나 FEATURE_GROUPS 에서 빠져 모델·SHAP·시나리오 미사용.
#   결과적으로 운영구조는 WB 실 proxy 만 남는다.
# relative_salary 는 ILOSTAT 실 임금(전문직/전체 임금비, use_worldbank 시)으로 재투입되어
#   보상수준 그룹에 복귀했다(GII 비구성요소 → 순환 없는 보상 신호). --no-wb 에선 합성 폴백.
#
# 2026-06 추가 (4범주 고정 — category-frame-locked):
#   structure +2 — mobile_subs_per100 · secure_servers_per_m (디지털 인프라, WB 실 IT.* 지표)
#   economic  +1 — youth_unemp_pct (청년 실업률, 보상의 반대편 신호 · GII 비구성요소 = 비순환)
#   sci_culture +1 — tertiary_attainment (학사 보유율 25+ %, WB 실 SE.TER.CUAT.BA.ZS)
#   → 카테고리별 7·5·4·4 로 균형. 4범주 키(sci_culture/hard_rd/structure/economic) 변경 없음.
FEATURE_GROUPS = {
    "sci_culture": [           # 과학문화 (Soft) — 과학·교육 인적자본 포함
        "sci_literacy",
        "sci_trust",
        "tech_acceptance",
        "pisa_math",           # PISA 수학점수 (인적자본, sci_literacy 와 공선성 가능 — 측정 두껍게 목적)
        "pisa_reading",        # PISA 읽기점수 (인적자본)
        "tertiary_enroll",     # 고등교육 진학률 % gross (WB SE.TER.ENRR)
        "tertiary_attainment", # 2026-06 추가: 학사 이상 보유 비율 % (WB SE.TER.CUAT.BA.ZS, 25+ 인구)
    ],
    "hard_rd": [               # 물적 규모 (Hard)
        "rd_gdp_pct",
        "researchers_per_m",
        "patent_apps",
        "journal_articles",
        "gdp_per_capita",
    ],
    "structure": [             # R&D 운영 구조 (Structure) — 합성 3종 제외, WB proxy + 디지털 인프라
        "gov_edu_exp_pct",
        "internet_users_pct",
        "mobile_subs_per100",   # 2026-06 추가: 모바일 가입자 100명당 (WB IT.CEL.SETS.P2, 디지털 인프라)
        "secure_servers_per_m", # 2026-06 추가: 보안 인터넷 서버 백만명당 (WB IT.NET.SECR.P6, 디지털 보안 인프라)
    ],
    "economic": [              # 보상 수준 (Economic) — 첨단수출 + 연구원당 R&D 자원(WB 파생) + 전문직 임금비(ILOSTAT) + 청년 실업률
        "hi_tech_export_pct",
        "rd_budget_per_researcher",  # WB 파생: rd_gdp_pct×gdp_pc×1e4/researchers_per_m (USD/연구원)
        "relative_salary",           # ILOSTAT 실 임금: OCU_ISCO08_2(전문직)/TOTAL (use_worldbank 시 실값, --no-wb 합성)
        "youth_unemp_pct",           # 2026-06 추가: 청년(15-24) 실업률 % (WB SL.UEM.1524.ZS) — 보상의 반대편 신호·GII 비구성요소
    ],
}

ALL_FEATURES = [f for feats in FEATURE_GROUPS.values() for f in feats]
# 타깃 후보 — 뒤 2종은 탈순환 타깃(findings §7.7): log(백만명당 논문/특허), GII 미사용(순환 0).
#   population(WB SP.POP.TOTL) 기반 파생이라 **실 WB 모드 전용** — --no-wb 경로엔 컬럼 부재(graceful):
#   preprocessor._validate_input_contract 는 "적어도 하나" 존재만 요구, 지정 타깃 부재는 명확한 에러.
OUTCOME_COLS = ["innovation_score", "competitiveness", "articles_per_m_log", "patents_per_m_log"]

LAG_YEARS = [3, 5, 7]   # 분석할 Time-lag (년)


# ──────────────────────────────────────────────
# 2. 카테고리 매핑 (← shap_analyzer.py)
# ──────────────────────────────────────────────
# legacy 4키(self_efficacy·avg_proj_duration·admin_burden_ratio·researcher_retire_age)는
# v1 재현(replicate_paper.py)·과거 산출물 호환을 위해 유지한다 — 삭제 금지.
CATEGORY_MAP = {
    # 과학문화 (Soft)
    "sci_literacy": "과학문화", "self_efficacy": "과학문화",
    "sci_trust": "과학문화", "tech_acceptance": "과학문화",
    "pisa_math": "과학문화", "pisa_reading": "과학문화",
    "tertiary_enroll": "과학문화",
    "tertiary_attainment": "과학문화",  # 2026-06 추가 (학사 보유율 · WB SE.TER.CUAT.BA.ZS)
    "idx_sci_culture": "과학문화",

    # 물적 규모 (Hard)
    "rd_gdp_pct": "물적규모", "researchers_per_m": "물적규모",
    "patent_apps": "물적규모", "journal_articles": "물적규모",
    "gdp_per_capita": "물적규모", "idx_hard_rd": "물적규모",

    # 운영 구조 (Structure)
    "avg_proj_duration": "운영구조", "admin_burden_ratio": "운영구조",
    "researcher_retire_age": "운영구조", "gov_edu_exp_pct": "운영구조",
    "internet_users_pct": "운영구조", "idx_structure": "운영구조",
    "mobile_subs_per100": "운영구조",     # 2026-06 추가 (모바일 보급 · WB IT.CEL.SETS.P2)
    "secure_servers_per_m": "운영구조",   # 2026-06 추가 (보안 인터넷 서버 · WB IT.NET.SECR.P6)

    # 보상 수준 (Economic)
    "relative_salary": "보상수준", "rd_budget_per_researcher": "보상수준",
    "hi_tech_export_pct": "보상수준", "idx_economic": "보상수준",
    "youth_unemp_pct": "보상수준",        # 2026-06 추가 (청년 실업률 · WB SL.UEM.1524.ZS · 비순환)
}

# 카테고리 한글 표시 라벨 (← shap_analyzer.py 의 로컬 CATEGORY_KR)
CATEGORY_LABELS_KR = {
    "과학문화": "과학문화 (Soft)",
    "물적규모": "물적 규모 (Hard)",
    "운영구조": "R&D 운영 구조",
    "보상수준": "보상 수준 (Economic)",
    "기타":    "기타",
}

# 카테고리 영문 표시 라벨 (← visualizer.py 의 LABEL_MAP)
CATEGORY_LABELS_EN = {
    "과학문화": "Science Culture (Soft)",
    "물적규모": "R&D Scale (Hard)",
    "운영구조": "R&D Structure",
    "보상수준": "Compensation (Economic)",
    "기타":    "Other",
}

# 카테고리 색상 팔레트 (← visualizer.py 의 CAT_COLORS)
CATEGORY_COLORS = {
    "과학문화": "#4C72B0",
    "물적규모": "#DD8452",
    "운영구조": "#55A868",
    "보상수준": "#C44E52",
    "기타":    "#8172B3",
}


# ──────────────────────────────────────────────
# 3. 기본 국가 목록 (← scenario_simulator.py · visualizer.py)
# ──────────────────────────────────────────────
# 시나리오 시뮬레이션 기본 대상국 (run_policy_scenarios 의 countries 폴백)
SCENARIO_DEFAULT_COUNTRIES = ["KOR", "JPN", "USA", "DEU", "FIN", "CHN"]
# 한국 갭 분석 기준국 (korea_deep_dive 의 ref_countries)
GAP_REFERENCE_COUNTRIES = ["FIN", "USA", "DEU"]
# 레이더 차트 기본 비교국 (plot_radar_comparison 의 countries 폴백)
RADAR_DEFAULT_COUNTRIES = ["KOR", "FIN", "USA", "DEU", "JPN"]
