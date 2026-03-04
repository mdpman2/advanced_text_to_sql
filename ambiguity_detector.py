"""
Ambiguity Detector — 모호한 질문 감지 및 후속 질문 생성 모듈 (v3.1.1)

QueryWeaver 참조: 모호하거나 불명확한 자연어 질문을 감지하고,
사용자에게 명확한 후속 질문(follow-up questions)을 제안합니다.

기능:
- 다중 테이블 해석 가능성 감지 (ambiguous table reference)
- 불특정 컬럼 참조 감지 (e.g. "매출" → revenue? sales_amount?)
- 시간 범위 미지정 감지 (e.g. "최근" → 1주? 1개월? 1년?)
- 집계 기준 미지정 감지 (e.g. "평균" → 어떤 컬럼의 평균?)
- 비교 대상 미지정 감지 (e.g. "보다 높은" → 무엇보다?)
- 후속 질문 자동 생성으로 정확도 향상

Author: Azure OpenAI Sample
Date: 2026-06-15
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from text_to_sql_agent import DatabaseSchema

from schema_linker import _VAGUE_TIME, _VAGUE_REFERENCES

# 미리 컴파일된 패턴
_TIME_KEYWORDS = re.compile(r'최근|지난|이번|작년|올해|어제|오늘|이전|지금까지', re.IGNORECASE)
_AGGREGATE_KEYWORDS = re.compile(r'평균|합계|총|개수|최대|최소|합산', re.IGNORECASE)
_COMPARISON_KEYWORDS = re.compile(r'보다\s*(?:높은|낮은|많은|적은|큰|작은)|이상|이하|초과|미만', re.IGNORECASE)
_RANKING_KEYWORDS = re.compile(r'상위|하위|가장|제일|최고|최저|top\s*\d*|best|worst', re.IGNORECASE)


@dataclass
class AmbiguityResult:
    """모호성 분석 결과"""
    is_ambiguous: bool
    reason: Optional[str] = None
    ambiguity_type: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 1.0  # 모호하지 않을수록 1.0에 가까움


class AmbiguityDetector:
    """
    모호한 질문 감지기

    QueryWeaver의 follow-up question 패턴을 참조.
    자연어 질문의 모호성을 분석하고 구체적인 후속 질문을 제안합니다.
    """

    def __init__(self, schema: Optional['DatabaseSchema'] = None):
        self._schema = schema
        self._table_names: Set[str] = set()
        self._column_names: Dict[str, List[str]] = {}  # 컬럼명 → 포함 테이블
        if schema:
            self._index_schema(schema)

    def _index_schema(self, schema: 'DatabaseSchema') -> None:
        """스키마 인덱싱"""
        for table in schema.tables:
            self._table_names.add(table.name.lower())
            for col in table.columns:
                col_name = col["name"].lower()
                if col_name not in self._column_names:
                    self._column_names[col_name] = []
                self._column_names[col_name].append(table.name)

    def detect(self, question: str, schema: Optional['DatabaseSchema'] = None) -> Dict[str, Any]:
        """
        모호성 감지 (메인 메서드)

        Returns:
            {
                "is_ambiguous": bool,
                "reason": str | None,
                "suggestions": list[str],
                "ambiguity_type": str | None,
                "confidence": float
            }
        """
        if schema and schema != self._schema:
            self._schema = schema
            self._index_schema(schema)

        checks = [
            self._check_vague_time_reference,
            self._check_ambiguous_column,
            self._check_missing_aggregate_target,
            self._check_vague_comparison,
            self._check_pronoun_reference,
            self._check_missing_ranking_scope,
            self._check_too_short,
        ]

        for check in checks:
            result = check(question)
            if result.is_ambiguous:
                return {
                    "is_ambiguous": True,
                    "reason": result.reason,
                    "suggestions": result.suggestions,
                    "ambiguity_type": result.ambiguity_type,
                    "confidence": result.confidence,
                }

        return {
            "is_ambiguous": False,
            "reason": None,
            "suggestions": [],
            "ambiguity_type": None,
            "confidence": 1.0,
        }

    # ── 개별 모호성 체크 ──

    def _check_vague_time_reference(self, question: str) -> AmbiguityResult:
        """시간 범위 미지정 감지"""
        if _VAGUE_TIME.search(question):
            # "최근 3개월" 등 구체적 기간이 있는지 확인
            has_specific = bool(re.search(r'최근\s*\d+\s*(?:일|주|개월|달|년|분기)', question))
            if not has_specific:
                return AmbiguityResult(
                    is_ambiguous=True,
                    reason="시간 범위가 명확하지 않습니다.",
                    ambiguity_type="vague_time",
                    suggestions=[
                        "구체적인 기간을 지정해주세요. (예: 최근 3개월, 지난 1년)",
                        "시작/종료 날짜를 명시해주세요. (예: 2025-01-01부터 2025-12-31까지)",
                        "'올해' 또는 '이번 분기' 등으로 구체화할 수 있습니다.",
                    ],
                    confidence=0.5,
                )
        return AmbiguityResult(is_ambiguous=False)

    def _check_ambiguous_column(self, question: str) -> AmbiguityResult:
        """다중 테이블에 존재하는 컬럼 참조 감지"""
        if not self._column_names:
            return AmbiguityResult(is_ambiguous=False)

        # 질문에 언급된 키워드와 다중 테이블 컬럼 교차
        ambiguous_cols = []
        for col_name, tables in self._column_names.items():
            if len(tables) > 1 and col_name in question.lower():
                ambiguous_cols.append((col_name, tables))

        if ambiguous_cols:
            col_info = ambiguous_cols[0]
            return AmbiguityResult(
                is_ambiguous=True,
                reason=f"'{col_info[0]}' 컬럼이 여러 테이블에 존재합니다: {', '.join(col_info[1])}",
                ambiguity_type="ambiguous_column",
                suggestions=[
                    f"어떤 테이블의 '{col_info[0]}'을(를) 의미하시나요?",
                    *[f"  - {t}.{col_info[0]}" for t in col_info[1]],
                ],
                confidence=0.4,
            )
        return AmbiguityResult(is_ambiguous=False)

    def _check_missing_aggregate_target(self, question: str) -> AmbiguityResult:
        """집계 대상 미지정 감지"""
        if _AGGREGATE_KEYWORDS.search(question):
            # "평균 급여", "총 매출" 등 대상이 있는지 확인
            keywords = ["평균", "합계", "총", "개수", "최대", "최소"]
            for kw in keywords:
                if kw in question:
                    # 키워드 뒤에 명사가 있는지 간단히 확인
                    idx = question.find(kw)
                    after = question[idx + len(kw):idx + len(kw) + 10].strip()
                    if not after or after[0] in "을를의은는이가?":
                        return AmbiguityResult(
                            is_ambiguous=True,
                            reason=f"'{kw}'의 대상이 명확하지 않습니다.",
                            ambiguity_type="missing_aggregate_target",
                            suggestions=[
                                f"어떤 값의 {kw}을(를) 구하시나요?",
                                f"예: '{kw} 연봉', '{kw} 매출', '{kw} 주문 수' 등으로 구체화해주세요.",
                            ],
                            confidence=0.5,
                        )
        return AmbiguityResult(is_ambiguous=False)

    def _check_vague_comparison(self, question: str) -> AmbiguityResult:
        """비교 대상 미지정 감지"""
        if _COMPARISON_KEYWORDS.search(question):
            # "보다 높은" → 무엇보다?
            has_reference = bool(re.search(r'(?:평균|중앙값|전체|\d+)\s*보다', question))
            if not has_reference and "보다" in question:
                return AmbiguityResult(
                    is_ambiguous=True,
                    reason="비교 기준이 명확하지 않습니다.",
                    ambiguity_type="vague_comparison",
                    suggestions=[
                        "비교 기준을 명시해주세요.",
                        "예: '평균보다 높은', '50000보다 큰', '전체 중앙값 이상'",
                    ],
                    confidence=0.5,
                )
        return AmbiguityResult(is_ambiguous=False)

    def _check_pronoun_reference(self, question: str) -> AmbiguityResult:
        """대명사/불명확한 참조 감지"""
        if _VAGUE_REFERENCES.search(question):
            return AmbiguityResult(
                is_ambiguous=True,
                reason="불명확한 참조('그것', '이것', '해당' 등)가 포함되어 있습니다.",
                ambiguity_type="pronoun_reference",
                suggestions=[
                    "구체적인 이름으로 대체해주세요.",
                    "예: '그 부서' → '개발팀', '해당 프로젝트' → 'AI 챗봇 개발 프로젝트'",
                ],
                confidence=0.4,
            )
        return AmbiguityResult(is_ambiguous=False)

    def _check_missing_ranking_scope(self, question: str) -> AmbiguityResult:
        """순위 범위 미지정 감지"""
        if _RANKING_KEYWORDS.search(question):
            # "가장 많은 프로젝트" → 무엇을 기준으로?
            has_criteria = bool(re.search(
                r'(?:연봉|급여|매출|인원|예산|시간|비용)\s*(?:이|가)?\s*가장', question
            ))
            has_top_n = bool(re.search(r'(?:상위|하위|top)\s*\d+', question, re.IGNORECASE))
            has_specific_target = bool(re.search(
                r'가장\s*(?:많은|적은|높은|낮은|큰|작은)\s*\S+', question
            ))
            if not has_criteria and not has_top_n and not has_specific_target:
                return AmbiguityResult(
                    is_ambiguous=True,
                    reason="순위 기준이 명확하지 않습니다.",
                    ambiguity_type="missing_ranking_scope",
                    suggestions=[
                        "순위를 매길 기준을 명시해주세요.",
                        "예: '연봉이 가장 높은 직원', '매출 상위 10개 제품'",
                    ],
                    confidence=0.5,
                )
        return AmbiguityResult(is_ambiguous=False)

    @staticmethod
    def _check_too_short(question: str) -> AmbiguityResult:
        """너무 짧은 질문 감지"""
        stripped = question.strip()
        if len(stripped) < 5:
            return AmbiguityResult(
                is_ambiguous=True,
                reason="질문이 너무 짧습니다.",
                ambiguity_type="too_short",
                suggestions=[
                    "좀 더 구체적으로 질문해주세요.",
                    "예: '부서별 직원 수를 보여주세요', '올해 매출 상위 10개 제품'",
                ],
                confidence=0.3,
            )
        return AmbiguityResult(is_ambiguous=False)
