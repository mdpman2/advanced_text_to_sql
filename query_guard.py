"""
Query Guard — 파괴적 SQL 감지 및 확인 모듈 (v3.1.0)

QueryWeaver 참조: INSERT/UPDATE/DELETE/DROP/TRUNCATE 등 파괴적 SQL을
자동 감지하고 사용자 확인 절차를 거치도록 합니다.

기능:
- 파괴적 SQL 문 감지 (INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER)
- 영향 범위 분석 (WHERE 절 유무, 전체 테이블 영향 경고)
- 확인 요청 생성 및 관리 (TTL 5분)
- SQL 문 유형별 위험도 분류 (critical, high, medium)

Author: Azure OpenAI Sample
Date: 2026-06-15
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from text_to_sql_agent import TextToSQLAgent

# 미리 컴파일된 패턴
_DESTRUCTIVE_PATTERNS = {
    "DELETE": re.compile(r'\bDELETE\s+FROM\b', re.IGNORECASE),
    "UPDATE": re.compile(r'\bUPDATE\s+\w+\s+SET\b', re.IGNORECASE),
    "INSERT": re.compile(r'\bINSERT\s+INTO\b', re.IGNORECASE),
    "DROP_TABLE": re.compile(r'\bDROP\s+TABLE\b', re.IGNORECASE),
    "DROP_DATABASE": re.compile(r'\bDROP\s+DATABASE\b', re.IGNORECASE),
    "TRUNCATE": re.compile(r'\bTRUNCATE\s+TABLE?\b', re.IGNORECASE),
    "ALTER": re.compile(r'\bALTER\s+TABLE\b', re.IGNORECASE),
}

_WHERE_PATTERN = re.compile(r'\bWHERE\b', re.IGNORECASE)
_TABLE_NAME_PATTERN = re.compile(
    r'\b(?:DELETE\s+FROM|UPDATE|INSERT\s+INTO|DROP\s+TABLE|TRUNCATE\s+TABLE?|ALTER\s+TABLE)\s+(\w+)',
    re.IGNORECASE
)


class RiskLevel(Enum):
    """SQL 위험도"""
    SAFE = "safe"           # SELECT
    MEDIUM = "medium"       # INSERT
    HIGH = "high"           # UPDATE, DELETE (WITH WHERE)
    CRITICAL = "critical"   # DROP, TRUNCATE, DELETE/UPDATE (WITHOUT WHERE)


@dataclass
class DestructiveAnalysis:
    """파괴적 SQL 분석 결과"""
    is_destructive: bool
    risk_level: RiskLevel
    operation: str                    # DELETE, UPDATE, INSERT, DROP 등
    target_table: Optional[str]       # 영향받는 테이블
    has_where_clause: bool            # WHERE 절 유무
    warning_message: str              # 경고 메시지
    affected_scope: str               # 영향 범위 설명


class QueryGuard:
    """
    SQL 안전 가드

    파괴적 SQL을 감지하고 실행 전 확인을 요구합니다.
    QueryWeaver의 ConfirmRequest 패턴 참조.
    """

    def is_destructive(self, sql: str) -> bool:
        """파괴적 SQL 여부 판단"""
        return any(pattern.search(sql) for pattern in _DESTRUCTIVE_PATTERNS.values())

    def analyze(self, sql: str) -> DestructiveAnalysis:
        """파괴적 SQL 상세 분석"""
        sql_stripped = sql.strip()

        # SELECT는 안전
        if not self.is_destructive(sql_stripped):
            return DestructiveAnalysis(
                is_destructive=False,
                risk_level=RiskLevel.SAFE,
                operation="SELECT",
                target_table=None,
                has_where_clause=False,
                warning_message="",
                affected_scope="읽기 전용 쿼리",
            )

        # 대상 테이블 추출
        table_match = _TABLE_NAME_PATTERN.search(sql_stripped)
        target_table = table_match.group(1) if table_match else "unknown"

        # WHERE 절 유무
        has_where = bool(_WHERE_PATTERN.search(sql_stripped))

        # 연산 타입 및 위험도 결정
        operation, risk_level, scope = self._classify_operation(sql_stripped, has_where)

        # 경고 메시지 생성
        warning = self._build_warning(operation, target_table, has_where, risk_level)

        return DestructiveAnalysis(
            is_destructive=True,
            risk_level=risk_level,
            operation=operation,
            target_table=target_table,
            has_where_clause=has_where,
            warning_message=warning,
            affected_scope=scope,
        )

    @staticmethod
    def _classify_operation(sql: str, has_where: bool) -> tuple[str, RiskLevel, str]:
        """연산 분류"""
        if _DESTRUCTIVE_PATTERNS["DROP_TABLE"].search(sql) or _DESTRUCTIVE_PATTERNS["DROP_DATABASE"].search(sql):
            return "DROP", RiskLevel.CRITICAL, "테이블/데이터베이스 삭제 — 복구 불가"

        if _DESTRUCTIVE_PATTERNS["TRUNCATE"].search(sql):
            return "TRUNCATE", RiskLevel.CRITICAL, "테이블 전체 데이터 삭제 — 복구 불가"

        if _DESTRUCTIVE_PATTERNS["DELETE"].search(sql):
            if has_where:
                return "DELETE", RiskLevel.HIGH, "조건에 맞는 행 삭제"
            return "DELETE", RiskLevel.CRITICAL, "⚠️ WHERE 절 없음 — 전체 테이블 삭제"

        if _DESTRUCTIVE_PATTERNS["UPDATE"].search(sql):
            if has_where:
                return "UPDATE", RiskLevel.HIGH, "조건에 맞는 행 수정"
            return "UPDATE", RiskLevel.CRITICAL, "⚠️ WHERE 절 없음 — 전체 테이블 수정"

        if _DESTRUCTIVE_PATTERNS["ALTER"].search(sql):
            return "ALTER", RiskLevel.HIGH, "테이블 구조 변경"

        if _DESTRUCTIVE_PATTERNS["INSERT"].search(sql):
            return "INSERT", RiskLevel.MEDIUM, "새 행 삽입"

        return "UNKNOWN", RiskLevel.HIGH, "알 수 없는 파괴적 연산"

    @staticmethod
    def _build_warning(operation: str, table: str, has_where: bool, risk: RiskLevel) -> str:
        """경고 메시지 생성"""
        risk_emoji = {
            RiskLevel.CRITICAL: "🔴",
            RiskLevel.HIGH: "🟠",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.SAFE: "🟢",
        }

        emoji = risk_emoji.get(risk, "⚠️")
        msg = f"{emoji} [{risk.value.upper()}] {operation} 연산 감지 — 대상 테이블: {table}"

        if not has_where and operation in ("DELETE", "UPDATE"):
            msg += "\n   ⚠️ WARNING: WHERE 절이 없습니다! 전체 테이블에 영향을 줍니다."

        msg += "\n   이 SQL을 실행하시겠습니까? (확인 필요)"
        return msg

    def get_warning_message(self, sql: str) -> str:
        """편의 메서드: SQL에 대한 경고 메시지 반환"""
        analysis = self.analyze(sql)
        return analysis.warning_message


class ConfirmationStore:
    """
    파괴적 SQL 확인 요청 저장소

    TTL 기반으로 5분 내 확인하지 않으면 자동 만료.
    QueryWeaver의 ConfirmRequest 패턴 참조.
    """

    TTL_SECONDS = 300  # 5분

    def __init__(self):
        self._pending: Dict[str, Dict[str, Any]] = {}

    def store(self, sql: str, agent: 'TextToSQLAgent') -> str:
        """확인 대기 항목 저장 → confirmation_id 반환"""
        self._cleanup_expired()
        confirmation_id = str(uuid.uuid4())
        self._pending[confirmation_id] = {
            "sql": sql,
            "agent": agent,
            "created_at": time.time(),
        }
        return confirmation_id

    def get(self, confirmation_id: str) -> Optional[Dict[str, Any]]:
        """확인 대기 항목 조회"""
        self._cleanup_expired()
        return self._pending.get(confirmation_id)

    def remove(self, confirmation_id: str) -> None:
        """확인 대기 항목 제거"""
        self._pending.pop(confirmation_id, None)

    def _cleanup_expired(self) -> None:
        """만료된 항목 정리"""
        now = time.time()
        expired = [
            cid for cid, info in self._pending.items()
            if now - info["created_at"] > self.TTL_SECONDS
        ]
        for cid in expired:
            del self._pending[cid]
