/* ============================================================
   DATA — 실제 파이프라인 산출물 기반. v1/v2 두 모드 지원.
   생성: scripts/build_data.py  (mock 아님)
   기본 모드: v2
   포함 모드: ['v1', 'v2']
   재생성:
     python main.py --country-set pisa-gii --output /tmp/sc_front      # v2
     python replicate_paper.py --output /tmp/sc_paper_v1               # v1
     python scripts/build_data.py /tmp/sc_front /tmp/sc_paper_v1       # 둘 다
   ============================================================ */
export const DATASETS = {
  "v2": {
    "kpis": [
      {
        "id": "culture",
        "label": "과학문화 기여도",
        "value": "28.6",
        "unit": "%",
        "note": "raw 28.6% · 통제 후 38%(발전수준·순환 제거)",
        "tone": "culture",
        "big": true
      },
      {
        "id": "r2",
        "label": "교차검증 R²",
        "value": "0.77",
        "unit": "",
        "note": "GroupKFold(국가) · 44국 일반화",
        "tone": "neutral"
      },
      {
        "id": "rmse",
        "label": "RMSE",
        "value": "4.08",
        "unit": "pt",
        "note": "GII 오차 · 선진국 미세차(2~5점)는 오차 내",
        "tone": "neutral"
      },
      {
        "id": "scope",
        "label": "분석 범위",
        "value": "44",
        "unit": "개국×12년",
        "note": "528 패널 표본 (PISA∩GII)",
        "tone": "neutral"
      },
      {
        "id": "feat",
        "label": "실 데이터 출처",
        "value": "5",
        "unit": "종",
        "note": "WB · PISA · WVS · GII · ILOSTAT (합성 4피처 모델 제외)",
        "tone": "neutral"
      }
    ],
    "categories": [
      {
        "id": "physical",
        "name": "물적규모",
        "value": 50.6,
        "color": "var(--c-physical)",
        "desc": "R&D 투자·연구원·특허·논문·1인당 GDP"
      },
      {
        "id": "culture",
        "name": "과학문화",
        "value": 28.6,
        "color": "var(--c-culture)",
        "desc": "PISA 과학·수학·읽기 + WVS 신뢰·기술수용 + 고등교육"
      },
      {
        "id": "reward",
        "name": "보상수준",
        "value": 13.0,
        "color": "var(--c-reward)",
        "desc": "전문직 임금비(ILOSTAT) + 연구원당 R&D 자원 + 첨단수출"
      },
      {
        "id": "ops",
        "name": "운영구조",
        "value": 7.7,
        "color": "var(--c-ops)",
        "desc": "교육지출·인터넷 인프라"
      }
    ],
    "features": [
      {
        "name": "1인당 GDP",
        "value": 0.258,
        "cat": "physical"
      },
      {
        "name": "물적규모 지수",
        "value": 0.169,
        "cat": "physical"
      },
      {
        "name": "PISA 읽기",
        "value": 0.072,
        "cat": "culture"
      },
      {
        "name": "PISA 수학",
        "value": 0.063,
        "cat": "culture"
      },
      {
        "name": "과학소양(PISA)",
        "value": 0.037,
        "cat": "culture"
      },
      {
        "name": "백만명당 연구원",
        "value": 0.033,
        "cat": "physical"
      },
      {
        "name": "특허 출원",
        "value": 0.028,
        "cat": "physical"
      },
      {
        "name": "GDP 대비 R&D",
        "value": 0.028,
        "cat": "physical"
      },
      {
        "name": "모바일 보급",
        "value": 0.027,
        "cat": "ops"
      },
      {
        "name": "연구원당 R&D 예산",
        "value": 0.026,
        "cat": "reward"
      }
    ],
    "catColor": {
      "culture": "var(--c-culture)",
      "physical": "var(--c-physical)",
      "ops": "var(--c-ops)",
      "reward": "var(--c-reward)"
    },
    "catLabel": {
      "reward": "보상",
      "physical": "물적",
      "culture": "과학문화",
      "ops": "운영"
    },
    "tornado": [
      {
        "name": "1인당 GDP",
        "value": 2.19,
        "cat": "physical"
      },
      {
        "name": "물적규모 지수",
        "value": 2.1,
        "cat": "physical"
      },
      {
        "name": "PISA 수학",
        "value": 0.9,
        "cat": "culture"
      },
      {
        "name": "PISA 읽기",
        "value": 0.74,
        "cat": "culture"
      },
      {
        "name": "GDP 대비 R&D",
        "value": 0.68,
        "cat": "physical"
      },
      {
        "name": "모바일 보급",
        "value": 0.57,
        "cat": "ops"
      },
      {
        "name": "과학소양(PISA)",
        "value": 0.56,
        "cat": "culture"
      },
      {
        "name": "백만명당 연구원",
        "value": 0.56,
        "cat": "physical"
      },
      {
        "name": "첨단기술 수출",
        "value": 0.37,
        "cat": "reward"
      }
    ],
    "countries": [
      {
        "id": "kr",
        "name": "한국",
        "flag": "🇰🇷",
        "base": 58.6
      },
      {
        "id": "jp",
        "name": "일본",
        "flag": "🇯🇵",
        "base": 54.3
      },
      {
        "id": "us",
        "name": "미국",
        "flag": "🇺🇸",
        "base": 61.6
      },
      {
        "id": "de",
        "name": "독일",
        "flag": "🇩🇪",
        "base": 58.5
      },
      {
        "id": "fi",
        "name": "핀란드",
        "flag": "🇫🇮",
        "base": 60.2
      }
    ],
    "policies": [
      {
        "id": "p1",
        "label": "R&D 투자 규모",
        "cat": "physical",
        "weight": 0.53
      },
      {
        "id": "p2",
        "label": "과학문화 확산",
        "cat": "culture",
        "weight": 0.274
      },
      {
        "id": "p3",
        "label": "첨단산업 보상",
        "cat": "reward",
        "weight": 0.127
      },
      {
        "id": "p4",
        "label": "교육·인프라 투자",
        "cat": "ops",
        "weight": 0.069
      }
    ],
    "timelag": {
      "lags": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
      ],
      "culture": [
        0.75,
        0.76,
        0.76,
        0.77,
        0.77,
        0.77,
        0.77,
        0.76
      ],
      "physical": [
        0.8,
        0.81,
        0.82,
        0.82,
        0.83,
        0.83,
        0.84,
        0.84
      ]
    },
    "cultureIndex": [
      {
        "id": "jp",
        "name": "일본",
        "value": 82,
        "korea": false
      },
      {
        "id": "fi",
        "name": "핀란드",
        "value": 78,
        "korea": false
      },
      {
        "id": "kr",
        "name": "한국",
        "value": 76,
        "korea": true
      },
      {
        "id": "de",
        "name": "독일",
        "value": 66,
        "korea": false
      },
      {
        "id": "us",
        "name": "미국",
        "value": 65,
        "korea": false
      },
      {
        "id": "se",
        "name": "스웨덴",
        "value": 64,
        "korea": false
      }
    ],
    "whatif": {
      "kr": {
        "culture": -0.11,
        "physical": 0.0,
        "ops": -0.01,
        "reward": 0.0
      },
      "jp": {
        "culture": 0.0,
        "physical": 0.13,
        "ops": -0.05,
        "reward": 0.16
      },
      "us": {
        "culture": 0.12,
        "physical": -0.14,
        "ops": -0.24,
        "reward": 0.04
      },
      "de": {
        "culture": -0.05,
        "physical": 0.03,
        "ops": -0.07,
        "reward": 0.66
      },
      "fi": {
        "culture": -0.08,
        "physical": 2.68,
        "ops": -0.12,
        "reward": -0.08
      }
    },
    "categoriesControlled": [
      {
        "id": "culture",
        "name": "과학문화",
        "value": 38.4,
        "color": "var(--c-culture)"
      },
      {
        "id": "physical",
        "name": "물적규모",
        "value": 28.8,
        "color": "var(--c-physical)"
      },
      {
        "id": "reward",
        "name": "보상수준",
        "value": 23.6,
        "color": "var(--c-reward)"
      },
      {
        "id": "ops",
        "name": "운영구조",
        "value": 9.2,
        "color": "var(--c-ops)"
      }
    ],
    "methods": [
      {
        "label": "XGBoost 한계기여",
        "value": "+0.086",
        "verdict": "예측 증분 중간",
        "ns": false,
        "detail": "R&D 통제 후 과학소양 단독 추가 효과"
      },
      {
        "label": "Pooled OLS β (R&D 통제)",
        "value": "+3.84***",
        "verdict": "횡단 연관 매우 견고",
        "ns": false,
        "detail": "원단위 GII점/1SD · 완전표준화 +0.394 · 국가 클러스터 SE · VIF 2.08"
      },
      {
        "label": "패널 FE β (within)",
        "value": "+1.30 n.s.",
        "verdict": "시간 내 효과 미검출",
        "ns": true,
        "detail": "국가 baseline 제거 시 사라짐 — PISA 느린 변동이 검정력 제한"
      }
    ],
    "methodsTakeaway": "연관은 실재하나 나라 간(횡단)이고, 인과적 within 효과는 이 데이터로 단정 못한다. 'X를 올리면 혁신이 오른다'는 인과 주장은 자제.",
    "mediation": {
      "x": "과학소양 (PISA)",
      "y": "혁신 (GII)",
      "pathways": [
        {
          "m": "researchers_per_m",
          "m_label": "연구원 밀도",
          "a": 0.75,
          "b": 5.151,
          "c_prime": 3.532,
          "indirect": 3.863,
          "sobel_p": "<0.001",
          "mediated_pct": 52.2,
          "highlight": true
        },
        {
          "m": "idx_hard_rd",
          "m_label": "물적규모 종합",
          "a": 0.634,
          "b": 5.138,
          "c_prime": 4.137,
          "indirect": 3.258,
          "sobel_p": "<0.001",
          "mediated_pct": 44.1,
          "highlight": false
        },
        {
          "m": "rd_gdp_pct",
          "m_label": "R&D 강도",
          "a": 0.525,
          "b": 4.251,
          "c_prime": 5.162,
          "indirect": 2.233,
          "sobel_p": "<0.001",
          "mediated_pct": 30.2,
          "highlight": false
        }
      ],
      "total_effect": 7.395,
      "note": "Baron-Kenny 4단계 + Sobel test, 44국 패널, 국가 클러스터 강건 SE. ★ 횡단 매개 — 시간 선행성 미보장(Imai-Tingley 후속 과제). 헤드라인 매개비율은 부트스트랩 49.1%[32.2,67.2](점추정 52.2%)."
    },
    "decircle": {
      "intro": "타깃이 GII(합성지수)일 때의 순환성('GII를 제 재료로 재구성') 반론을 검증하기 위해, GII 종합을 우회하는 세 가지 타깃(직접측정 2종 + 산출 서브지수)으로 동일 분석을 반복했다.",
      "rows": [
        {
          "target": "특허/백만명 (log)",
          "kind": "직접측정",
          "beta": "+0.410***",
          "within": "+0.349*** (p=0.0006)",
          "withinSig": true,
          "mediation": "−13.9% n.s.",
          "cvCulture": 0.402,
          "cvRnd": 0.302
        },
        {
          "target": "논문/백만명 (log)",
          "kind": "직접측정",
          "beta": "+0.564***",
          "within": "−0.037 n.s.",
          "withinSig": false,
          "mediation": "+21.0%*",
          "cvCulture": 0.656,
          "cvRnd": 0.513
        },
        {
          "target": "GII Output Sub-Index",
          "kind": "표준 지수(산출 절반)",
          "beta": "+0.325***",
          "within": "+0.385 n.s. (p=0.12)",
          "withinSig": false,
          "mediation": "—",
          "cvCulture": null,
          "cvRnd": null
        }
      ],
      "robustness": "제조업 부가가치(%GDP, 혁신과 무관한 외부 지표)와는 전 분석 무관(n.s.) — 신호가 혁신 산출에 특이적(분별 타당도).",
      "conclusion": "횡단 연관(β)은 세 구도 모두 생존 — '연관은 GII 순환 덕'이라는 대안 설명 기각. within(시간 내)은 특허 직접측정에서만 유의 — 합성지수의 sticky함이 within 검출을 막는다는 해석과 부합.",
      "caveats": "β는 표준화 계수(R&D 투입+GDP 통제, 국가 클러스터 SE). 문화 예측 우위(CV)는 직접측정 타깃에서만 성립. 인과 입증이 아니라 대안 설명 1개의 제거. GII Output은 Mendeley 정리본(CC BY 4.0, ISO3 결함을 국가명 기준 복구) 2013–2022. 매개비율은 동결 confounder-only 부트스트랩(국가 클러스터 2000회) 기준 — 논문 +21.0%[5.9,44.4] 유의, 특허 −13.9%[−61.6,36.0] 0 포함·불안정, GII Output 은 동결 사양에서 미보고(—)."
    },
    "caveats": [
      {
        "title": "GII 순환성 (2026-06 검증)",
        "body": "타깃 GII는 R&D지출·특허·논문 등 ~80지표의 합성 인덱스. 우리 물적규모 피처와 일부 구성요소 중복 → 예측 R²의 상당부분이 동어반복. → 탈순환 3구도(특허·논문·GII Output) 재검증에서 횡단 연관 전부 생존(§7.7·§7.8) — 순환이 연관의 *원천*이라는 대안 설명은 기각. 단 예측 R²의 절대값은 여전히 순환·발전수준으로 부풀려질 수 있음."
      },
      {
        "title": "PISA 3년 주기·보간",
        "body": "PISA 과학·수학·읽기는 2012/15/18 앵커만 실측, 사이 연도는 국가별 선형 보간. 느린 변동이 within 검정력을 제한."
      },
      {
        "title": "표본 한계",
        "body": "44국·528 패널(2012–2023). GroupKFold(국가) CV는 '처음 보는 국가' 일반화를 측정. 절대 오차 RMSE~5 GII점은 선진국 미세차(2~5점)를 가르기에 부족."
      },
      {
        "title": "통제 결과의 실행 민감성",
        "body": "발전수준·순환 통제 시 raw의 62 vs 28 격차가 ~42–49 vs ~42–49로 비등해지나, 순서는 실행·표본에 민감(다른 실행선 뒤집힘). '격차 축소'로 읽되 '역전 단정'은 금지."
      },
      {
        "title": "분류체계 한계",
        "body": "4대 카테고리(과학문화·물적규모·운영구조·보상수준)는 본 연구의 분석 프레임으로, 범주 내 피처 수 비대칭과 일부 이질성이 있다."
      },
      {
        "title": "SHAP ≠ 정책 진단",
        "body": "음의 SHAP은 '모델이 그 피처값으로 기대한 점수보다 실제가 낮음'을 뜻하지 현실의 부족이 아니다. 예: 한국 인터넷 보급 SHAP −0.44지만 실측은 97.4%로 44국 중 5위(세계 최상위권). 단일 국가 local SHAP을 정책 단정으로 읽으면 오해 — category·횡단 비교와 실측값을 함께 봐야 한다."
      },
      {
        "title": "데이터 출처(실데이터화)",
        "body": "World Bank(9지표) · WVS(W6/W7) · PISA(과학·수학·읽기) · GII(타깃) · ILOSTAT(전문직 임금비) · WB 파생(연구원당 R&D 자원). 합성 4피처는 모델 제외."
      },
      {
        "title": "두 사양의 관계",
        "body": "v2 는 실데이터 5종을 통합하고 국가를 44개국으로 확장한 본 분석이다. v1 은 합성 데이터·20개국의 초기 사양으로, 입력·피처·표본이 달라 두 모드의 수치는 직접 비교하지 않는다(상단 토글로 전환)."
      },
      {
        "title": "매개분석 — 횡단·시간 선행성 미보장",
        "body": "§7.5 매개분석은 Baron-Kenny 4단계+Sobel 횡단 추정으로, '과학소양 → 연구원밀도 → 혁신' 경로의 *연관 매개*를 보인다. 시간 선행성·반사실적 인과는 미보장(Imai-Tingley 시차 매개 후속 과제). '경로(pathway)'로 읽되 'X→Y 인과'로 단정 금지."
      },
      {
        "title": "피처 보강 2026-06 (4범주 고정)",
        "body": "운영구조 빈약(2→4) 해소 + 보상수준 비순환 청년실업 + 과학문화 학사보유율. ⚠ tertiary_attainment는 GII Human Capital 일부 가중 — 부분 순환."
      }
    ],
    "pisaRefYear": 2018,
    "label": "v2 — 확장 분석 (44국·실데이터·확장 피처)",
    "subtitle": "XGBoost + SHAP — 과학소양(PISA)이 국가혁신과 횡단 연관 (R&D 통제 후 견고; 특허 직접측정에서는 within +0.349*** 검출 — §탈순환) · 발전수준 통제 시 비등 (과학문화 ~38% vs 물적 ~29%)",
    "mode": "v2",
    "modePerf": "CV R²=0.770 · RMSE=4.08 · N=528"
  },
  "v1": {
    "kpis": [
      {
        "id": "culture",
        "label": "과학문화 기여도",
        "value": "12.9",
        "unit": "%",
        "note": "초기 사양 · 물적규모 45%와 함께 상위",
        "tone": "culture",
        "big": true
      },
      {
        "id": "r2",
        "label": "교차검증 R²",
        "value": "0.46",
        "unit": "",
        "note": "GroupKFold · 20국 (논문 0.509)",
        "tone": "neutral"
      },
      {
        "id": "rmse",
        "label": "RMSE",
        "value": "2.72",
        "unit": "pt",
        "note": "GII 오차 (논문 2.29)",
        "tone": "neutral"
      },
      {
        "id": "scope",
        "label": "분석 범위",
        "value": "20",
        "unit": "개국×12년",
        "note": "240 패널 (논문 사양)",
        "tone": "neutral"
      },
      {
        "id": "feat",
        "label": "데이터",
        "value": "합성",
        "unit": "",
        "note": "PISA·WVS·ILOSTAT 합성(논문 P111)",
        "tone": "neutral"
      }
    ],
    "categories": [
      {
        "id": "physical",
        "name": "물적규모",
        "value": 45.3,
        "color": "var(--c-physical)",
        "desc": "R&D 투자·연구원·특허·논문·1인당 GDP"
      },
      {
        "id": "reward",
        "name": "보상수준",
        "value": 30.5,
        "color": "var(--c-reward)",
        "desc": "상대임금·연구원당 R&D 예산·첨단수출 (합성 2종 포함)"
      },
      {
        "id": "culture",
        "name": "과학문화",
        "value": 12.9,
        "color": "var(--c-culture)",
        "desc": "PISA 과학소양·자아효능감·과학신뢰·기술수용 (합성)"
      },
      {
        "id": "ops",
        "name": "운영구조",
        "value": 11.2,
        "color": "var(--c-ops)",
        "desc": "과제기간·행정부담·은퇴연령·교육지출·인터넷 (합성 3종 포함)"
      }
    ],
    "features": [
      {
        "name": "과학 논문",
        "value": 0.267,
        "cat": "physical"
      },
      {
        "name": "첨단기술 수출",
        "value": 0.161,
        "cat": "reward"
      },
      {
        "name": "특허 출원",
        "value": 0.109,
        "cat": "physical"
      },
      {
        "name": "첨단기술 수출 (3년 시차)",
        "value": 0.051,
        "cat": "reward"
      },
      {
        "name": "과학소양(PISA)",
        "value": 0.044,
        "cat": "culture"
      },
      {
        "name": "인터넷 보급",
        "value": 0.035,
        "cat": "ops"
      },
      {
        "name": "운영구조 지수",
        "value": 0.031,
        "cat": "ops"
      },
      {
        "name": "물적규모 지수",
        "value": 0.029,
        "cat": "physical"
      },
      {
        "name": "보상수준 지수",
        "value": 0.028,
        "cat": "reward"
      },
      {
        "name": "백만명당 연구원",
        "value": 0.026,
        "cat": "physical"
      }
    ],
    "catColor": {
      "culture": "var(--c-culture)",
      "physical": "var(--c-physical)",
      "ops": "var(--c-ops)",
      "reward": "var(--c-reward)"
    },
    "catLabel": {
      "reward": "보상",
      "physical": "물적",
      "culture": "과학문화",
      "ops": "운영"
    },
    "tornado": [
      {
        "name": "첨단기술 수출",
        "value": 1.17,
        "cat": "reward"
      },
      {
        "name": "과학 논문",
        "value": 0.86,
        "cat": "physical"
      },
      {
        "name": "특허 출원",
        "value": 0.58,
        "cat": "physical"
      },
      {
        "name": "첨단기술 수출 (3년 시차)",
        "value": -0.55,
        "cat": "reward"
      },
      {
        "name": "운영구조 지수",
        "value": -0.45,
        "cat": "ops"
      },
      {
        "name": "상대 연봉 (3년 시차)",
        "value": -0.41,
        "cat": "reward"
      },
      {
        "name": "인터넷 보급",
        "value": -0.29,
        "cat": "ops"
      },
      {
        "name": "상대 연봉",
        "value": -0.25,
        "cat": "reward"
      },
      {
        "name": "물적규모 지수",
        "value": 0.13,
        "cat": "physical"
      }
    ],
    "countries": [
      {
        "id": "kr",
        "name": "한국",
        "flag": "🇰🇷",
        "base": 53.2
      },
      {
        "id": "jp",
        "name": "일본",
        "flag": "🇯🇵",
        "base": 50.1
      },
      {
        "id": "us",
        "name": "미국",
        "flag": "🇺🇸",
        "base": 57.5
      },
      {
        "id": "de",
        "name": "독일",
        "flag": "🇩🇪",
        "base": 55.5
      },
      {
        "id": "fi",
        "name": "핀란드",
        "flag": "🇫🇮",
        "base": 56.0
      },
      {
        "id": "cn",
        "name": "중국",
        "flag": "🇨🇳",
        "base": 48.6
      }
    ],
    "policies": [
      {
        "id": "p1",
        "label": "S1 과학문화 강화",
        "cat": "culture",
        "weight": 0.2,
        "scenario_key": "S1_과학문화_강화"
      },
      {
        "id": "p2",
        "label": "S2 보상 개선",
        "cat": "reward",
        "weight": 0.2,
        "scenario_key": "S2_보상_개선"
      },
      {
        "id": "p3",
        "label": "S3 과제기간 연장",
        "cat": "ops",
        "weight": 0.2,
        "scenario_key": "S3_과제기간_연장"
      },
      {
        "id": "p4",
        "label": "S4 종합 혁신패키지",
        "cat": "culture",
        "weight": 0.2,
        "scenario_key": "S4_종합_혁신패키지"
      },
      {
        "id": "p5",
        "label": "S5 물적규모 확대",
        "cat": "physical",
        "weight": 0.2,
        "scenario_key": "S5_물적규모_확대"
      }
    ],
    "timelag": {
      "lags": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
      ],
      "culture": [
        0.15,
        0.13,
        0.12,
        0.12,
        0.12,
        0.13,
        0.12,
        0.09
      ],
      "physical": [
        0.69,
        0.68,
        0.68,
        0.69,
        0.7,
        0.68,
        0.7,
        0.71
      ]
    },
    "cultureIndex": [
      {
        "id": "jp",
        "name": "일본",
        "value": 47,
        "korea": false
      },
      {
        "id": "kr",
        "name": "한국",
        "value": 44,
        "korea": true
      },
      {
        "id": "us",
        "name": "미국",
        "value": 42,
        "korea": false
      },
      {
        "id": "de",
        "name": "독일",
        "value": 34,
        "korea": false
      },
      {
        "id": "fi",
        "name": "핀란드",
        "value": 33,
        "korea": false
      },
      {
        "id": "se",
        "name": "스웨덴",
        "value": 31,
        "korea": false
      }
    ],
    "whatif": {
      "kr": {
        "culture_S1_과학문화_강화": 0.11,
        "reward_S2_보상_개선": 0.95,
        "ops_S3_과제기간_연장": 0.0,
        "culture_S4_종합_혁신패키지": 0.89,
        "physical_S5_물적규모_확대": 0.0,
        "S1_과학문화_강화": 0.11,
        "S2_보상_개선": 0.95,
        "S3_과제기간_연장": 0.0,
        "S4_종합_혁신패키지": 0.89,
        "S5_물적규모_확대": 0.0
      },
      "jp": {
        "culture_S1_과학문화_강화": 0.08,
        "reward_S2_보상_개선": -0.14,
        "ops_S3_과제기간_연장": 0.13,
        "culture_S4_종합_혁신패키지": 0.53,
        "physical_S5_물적규모_확대": 0.28,
        "S1_과학문화_강화": 0.08,
        "S2_보상_개선": -0.14,
        "S3_과제기간_연장": 0.13,
        "S4_종합_혁신패키지": 0.53,
        "S5_물적규모_확대": 0.28
      },
      "us": {
        "culture_S1_과학문화_강화": 0.05,
        "reward_S2_보상_개선": -0.57,
        "ops_S3_과제기간_연장": -0.07,
        "culture_S4_종합_혁신패키지": -0.03,
        "physical_S5_물적규모_확대": 0.04,
        "S1_과학문화_강화": 0.05,
        "S2_보상_개선": -0.57,
        "S3_과제기간_연장": -0.07,
        "S4_종합_혁신패키지": -0.03,
        "S5_물적규모_확대": 0.04
      },
      "de": {
        "culture_S1_과학문화_강화": -0.21,
        "reward_S2_보상_개선": -0.29,
        "ops_S3_과제기간_연장": -0.17,
        "culture_S4_종합_혁신패키지": -0.6,
        "physical_S5_물적규모_확대": 0.05,
        "S1_과학문화_강화": -0.21,
        "S2_보상_개선": -0.29,
        "S3_과제기간_연장": -0.17,
        "S4_종합_혁신패키지": -0.6,
        "S5_물적규모_확대": 0.05
      },
      "fi": {
        "culture_S1_과학문화_강화": 0.46,
        "reward_S2_보상_개선": 0.54,
        "ops_S3_과제기간_연장": 0.18,
        "culture_S4_종합_혁신패키지": 0.71,
        "physical_S5_물적규모_확대": 0.0,
        "S1_과학문화_강화": 0.46,
        "S2_보상_개선": 0.54,
        "S3_과제기간_연장": 0.18,
        "S4_종합_혁신패키지": 0.71,
        "S5_물적규모_확대": 0.0
      },
      "cn": {
        "culture_S1_과학문화_강화": 0.0,
        "reward_S2_보상_개선": 0.0,
        "ops_S3_과제기간_연장": 1.1,
        "culture_S4_종합_혁신패키지": 0.36,
        "physical_S5_물적규모_확대": 0.04,
        "S1_과학문화_강화": 0.0,
        "S2_보상_개선": 0.0,
        "S3_과제기간_연장": 1.1,
        "S4_종합_혁신패키지": 0.36,
        "S5_물적규모_확대": 0.04
      }
    },
    "categoriesControlled": [],
    "methods": [],
    "methodsTakeaway": "",
    "mediation": {},
    "decircle": null,
    "caveats": [
      {
        "title": "초기 사양 재현 모드",
        "body": "본 화면은 초기 사양(합성 데이터)으로 재현한 결과. 20개국·12년·240행, PISA/WVS/ILOSTAT 합성 시뮬레이션, 17변수·5시나리오."
      },
      {
        "title": "합성 데이터 한계",
        "body": "PISA·WVS 실데이터가 아닌 국가 프로파일 기반 합성. 본 원고의 수치는 모두 v2(실데이터 44개국) 모드에서 나온다 — v1 은 방법론 시연용이다."
      },
      {
        "title": "합성 데이터 — 실측 기반 보정 (2026-06)",
        "body": "COUNTRY_PROFILES 4-tuple 을 PISA 2018·WB R&D/GDP·ILOSTAT 임금비·GII 실측 기반으로 재산정해 합성 절대수치 왜곡을 해소(예: 한국 PISA 374→519 근방, 중국 PISA 350→590 근방). 함수 형태·시나리오 정의 등 논문 로직은 그대로 보존. 본 실행 CV R²·카테고리%는 논문 0.509·24.2%와 약간 다를 수 있다(데이터 보정의 결과)."
      },
      {
        "title": "표본 규모 (논문 §5.4)",
        "body": "20개국 × 12년 = 240 관측치는 비교적 소규모 패널. CV R² ≈ 0.51은 국가 간 혁신 격차의 약 절반을 설명. 모형 일반화 가능성에 제약."
      },
      {
        "title": "GII 순환성",
        "body": "타깃 GII는 R&D지출·특허·논문 등 ~80지표의 합성 인덱스. 우리 물적규모 피처와 구성요소 중복 → 예측 R²의 상당부분이 동어반복."
      },
      {
        "title": "근사 재현",
        "body": "본 실행 CV R² ≈ 0.49·RMSE ≈ 2.7. 카테고리 % / 시나리오 부호는 합성 입력(국가 프로파일)의 실측 보정 결과로 변동한다. 정량 비교는 자제하고 방법론 시연으로만 읽을 것."
      }
    ],
    "pisaRefYear": 2018,
    "label": "v1 — 초기 사양 (20국·합성·17피처·5시나리오)",
    "subtitle": "XGBoost + SHAP — 초기 사양 (PISA·WVS 합성)",
    "mode": "v1",
    "modePerf": "CV R²=0.462 · RMSE=2.72 · N=240"
  }
};
export const DEFAULT_MODE = "v2";
export const DATA = DATASETS["v2"];
