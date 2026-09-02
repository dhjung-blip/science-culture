"""
config.py 공용 상수 계약 테스트 (의존성 없는 가드)
====================================================
config.py 는 stdlib-only leaf 모듈이라 ML 스택 없이 항상 import 가능하다
→ 기본 pytest 실행(구조 테스트와 동일 게이트)에 포함된다.

검증 대상:
  1) 4범주 키 고정 (category-frame-locked — 재분류·재명명 금지)
  2) 모든 base 피처 + idx_* 4종이 CATEGORY_MAP 에 존재 (⊆ 방향)
  3) CATEGORY_MAP 값 ⊆ 4범주, 라벨(KR/EN)·색상 키가 4범주(+기타) 커버
  4) LAG_YEARS 양의 정수 오름차순
"""
from __future__ import annotations

import sys
from pathlib import Path

# code/ 를 모듈 경로에 추가 (pyproject.toml pythonpath 와 동일 — 단독 실행 대비).
_FILES_DIR = Path(__file__).resolve().parent.parent
if str(_FILES_DIR) not in sys.path:
    sys.path.insert(0, str(_FILES_DIR))

import config

FOUR_GROUP_KEYS = {"sci_culture", "hard_rd", "structure", "economic"}
FOUR_CATEGORIES_KR = {"과학문화", "물적규모", "운영구조", "보상수준"}


def test_feature_groups_four_keys_locked() -> None:
    """4범주 키 고정: sci_culture/hard_rd/structure/economic (정확 일치)."""
    assert set(config.FEATURE_GROUPS) == FOUR_GROUP_KEYS, (
        f"4범주 키 변경 감지: {sorted(config.FEATURE_GROUPS)} — "
        "category-frame-locked(논문 재현성) 위반. 재분류·재명명 금지."
    )


def test_all_base_features_and_indices_in_category_map() -> None:
    """모든 base 피처 + idx_<group> 4종이 CATEGORY_MAP 에 존재(⊆ 방향).

    CATEGORY_MAP 에는 legacy 키(self_efficacy 등 v1 전용)가 더 있을 수 있으므로
    역방향(CATEGORY_MAP ⊆ ALL_FEATURES)은 검사하지 않는다.
    """
    expected = set(config.ALL_FEATURES) | {f"idx_{g}" for g in config.FEATURE_GROUPS}
    missing = expected - set(config.CATEGORY_MAP)
    assert not missing, f"CATEGORY_MAP 누락 피처: {sorted(missing)}"


def test_category_map_values_are_four_categories() -> None:
    """CATEGORY_MAP 값은 4범주 한글명만 허용('기타'는 _get_category 폴백 전용)."""
    extra = set(config.CATEGORY_MAP.values()) - FOUR_CATEGORIES_KR
    assert not extra, f"CATEGORY_MAP 에 허용 외 카테고리 값: {sorted(extra)}"


def test_labels_and_colors_cover_categories() -> None:
    """라벨(KR/EN)·색상 dict 키가 4범주 + '기타' 를 모두 커버."""
    required = FOUR_CATEGORIES_KR | {"기타"}
    for name, mapping in [
        ("CATEGORY_LABELS_KR", config.CATEGORY_LABELS_KR),
        ("CATEGORY_LABELS_EN", config.CATEGORY_LABELS_EN),
        ("CATEGORY_COLORS", config.CATEGORY_COLORS),
    ]:
        missing = required - set(mapping)
        assert not missing, f"{name} 누락 키: {sorted(missing)}"


def test_lag_years_positive_ascending() -> None:
    """LAG_YEARS 는 양의 정수 · 중복 없음 · 오름차순."""
    assert all(isinstance(y, int) and y > 0 for y in config.LAG_YEARS), (
        f"LAG_YEARS 에 양의 정수가 아닌 값: {config.LAG_YEARS}"
    )
    assert config.LAG_YEARS == sorted(set(config.LAG_YEARS)), (
        f"LAG_YEARS 가 중복 없는 오름차순이 아님: {config.LAG_YEARS}"
    )
