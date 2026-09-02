"""
아키텍처 불변식 구조 테스트 (의존성 없는 가드)
================================================
docs/architecture.md §3 의 불변식을 import-linter 와 **독립적으로** 검증한다.
서드파티 스택(xgboost/shap/pandas 등) 없이 표준 라이브러리 AST 정적 분석만 사용
→ 빠르고 결정적이며, ML 환경이 없어도 CI/훅에서 항상 실행 가능.

import-linter(pyproject.toml [tool.importlinter])와 동일한 규칙을 인코딩하므로,
둘 중 하나가 환경 문제로 죽어도 다른 하나가 불변식을 지킨다(defense in depth).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODULES_DIR = Path(__file__).resolve().parent.parent  # = code/

# 6개 파이프라인 스테이지 (main 은 오케스트레이터라 제외)
STAGES = {
    "data_collector",
    "preprocessor",
    "model_trainer",
    "shap_analyzer",
    "scenario_simulator",
    "visualizer",
}

# __main__ 자가테스트 블록의 cross-import (런타임 아님 → 허용 baseline).
# pyproject.toml [tool.importlinter] 의 ignore_imports 와 반드시 일치시킨다.
TOLERATED_EDGES = {
    ("preprocessor", "data_collector"),
    ("model_trainer", "data_collector"),
    ("model_trainer", "preprocessor"),
}


def _imported_top_level_modules(path: Path) -> set[str]:
    """파일에서 import 되는 top-level 모듈명 전부 (중첩/함수 내 포함)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # 절대 import 만
                names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_stage_does_not_import_other_stage(stage: str) -> None:
    """불변식: 스테이지 모듈은 서로를 import 하지 않는다 (자가테스트 baseline 제외)."""
    path = MODULES_DIR / f"{stage}.py"
    assert path.exists(), f"스테이지 파일 없음: {path}"

    imported_stages = {
        m for m in _imported_top_level_modules(path) if m in STAGES and m != stage
    }
    offending = {(stage, target) for target in imported_stages} - TOLERATED_EDGES

    assert not offending, (
        f"스테이지 독립 불변식 위반: {sorted(offending)}\n"
        f"→ 스테이지 간 직접 import 금지. 데이터는 main.py 를 통해 인자로 전달하세요."
    )


def test_main_orchestrates_all_stages() -> None:
    """불변식: main.py 는 6개 스테이지를 모두 import 하는 오케스트레이터다."""
    imported = _imported_top_level_modules(MODULES_DIR / "main.py")
    missing = STAGES - imported
    assert not missing, f"main.py 가 import 하지 않는 스테이지: {sorted(missing)}"


def test_config_is_pure_leaf() -> None:
    """불변식: config.py 는 순수 leaf 상수 모듈 — 스테이지·main 을 import 하지 않는다.

    (스테이지 → config 방향은 architecture.md §3 예외로 허용. STAGES 집합에
    config 를 추가하지 말 것 — config 는 스테이지가 아니다.)
    """
    path = MODULES_DIR / "config.py"
    assert path.exists(), f"config.py 없음: {path}"

    imported = _imported_top_level_modules(path)
    forbidden = imported & (STAGES | {"main"})
    assert not forbidden, (
        f"config.py 가 import 하면 안 되는 모듈: {sorted(forbidden)}\n"
        f"→ config 는 leaf 모듈이어야 한다(순환 의존 방지)."
    )
