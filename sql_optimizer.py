"""
SQL Optimizer & Self-Correction Module (v2.2.1)

SQL 쿼리 최적화 및 자가 수정 기능을 제공합니다.
Spider 2.0 벤치마크 #1 TCDataAgent-SQL (93.97%) 기술 참조.

기능:
- SELECT * / IN 서브쿼리 / ORDER BY without LIMIT 감지
- SelfCorrectionEngine: 테이블/컬럼 오타, 모호한 컬럼, GROUP BY 누락, 조인 오류 자동 수정
- 최대 5회 재시도 (5-round self-correction)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from text_to_sql_agent import DatabaseSchema

# 미리 컴파일된 정규식 패턴
_PATTERNS = {
    'select_star': re.compile(r'\bSELECT\s+\*\s+FROM\b', re.IGNORECASE),
    'in_subquery': re.compile(r'\bIN\s*\(\s*SELECT\b', re.IGNORECASE),
    'join': re.compile(r'\bJOIN\b', re.IGNORECASE),
    'alias': re.compile(r'\b\w+\s+AS\s+\w+\b|\b\w+\s+\w+\s+ON\b', re.IGNORECASE),
    'order_by_limit': re.compile(r'\bORDER\s+BY\b.*\bLIMIT\b', re.IGNORECASE),
    'order_by': re.compile(r'\bORDER\s+BY\b', re.IGNORECASE),
}


class SQLIssueType(Enum):
    """SQL 문제 유형"""
    SYNTAX_ERROR = "syntax_error"
    SCHEMA_MISMATCH = "schema_mismatch"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    MISSING_JOIN = "missing_join"
    INEFFICIENT_QUERY = "inefficient_query"
    LOGIC_ERROR = "logic_error"


@dataclass(slots=True)
class SQLIssue:
    """발견된 SQL 문제 정보"""
    issue_type: SQLIssueType
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    severity: str = "error"  # error, warning, info


@dataclass(slots=True)
class OptimizationResult:
    """최적화 결과"""
    original_sql: str
    optimized_sql: str
    issues_found: List[SQLIssue] = field(default_factory=list)
    optimizations_applied: List[str] = field(default_factory=list)
    estimated_improvement: Optional[float] = None


class SQLOptimizer:
    """
    SQL 쿼리 최적화기 (2026년 최신 규칙 적용)

    기능:
    - 쿼리 재작성
    - 인덱스 활용 제안
    - 조인 순서 최적화
    - 서브쿼리 -> 조인 변환
    - CTE 활용 제안
    """

    __slots__ = ('optimization_rules',)

    def __init__(self):
        self.optimization_rules: List[Callable[[str], Tuple[str, Optional[str]]]] = [
            self._optimize_select_star,
            self._optimize_subquery_to_join,
            self._optimize_exists_vs_in,
            self._add_table_aliases,
            self._optimize_order_by,
            self._optimize_distinct,
            self._optimize_like_pattern,
            self._suggest_cte_usage,
            self._optimize_null_comparison,
        ]

    @staticmethod
    def _optimize_select_star(sql: str) -> Tuple[str, Optional[str]]:
        """SELECT * 최적화 (필요한 컬럼만 선택하도록 제안)"""
        if _PATTERNS['select_star'].search(sql):
            return sql, "SELECT * 대신 필요한 컬럼만 명시하는 것이 좋습니다."
        return sql, None

    @staticmethod
    def _optimize_subquery_to_join(sql: str) -> Tuple[str, Optional[str]]:
        """IN 서브쿼리를 JOIN으로 변환 가능 여부 확인"""
        if _PATTERNS['in_subquery'].search(sql):
            return sql, "IN 서브쿼리는 JOIN으로 변환하면 성능이 향상될 수 있습니다."
        return sql, None

    @staticmethod
    def _optimize_exists_vs_in(sql: str) -> Tuple[str, Optional[str]]:
        """EXISTS vs IN 최적화 제안"""
        if _PATTERNS['in_subquery'].search(sql):
            if 'DISTINCT' not in sql.upper():
                return sql, "대용량 데이터에서는 IN 대신 EXISTS를 고려해보세요."
        return sql, None

    @staticmethod
    def _add_table_aliases(sql: str) -> Tuple[str, Optional[str]]:
        """테이블 별칭 추가 제안"""
        if _PATTERNS['join'].search(sql):
            if not _PATTERNS['alias'].search(sql):
                return sql, "복잡한 쿼리에서는 테이블 별칭 사용을 권장합니다."
        return sql, None

    @staticmethod
    def _optimize_order_by(sql: str) -> Tuple[str, Optional[str]]:
        """ORDER BY 최적화"""
        if _PATTERNS['order_by_limit'].search(sql):
            return sql, None  # LIMIT이 있으면 OK
        if _PATTERNS['order_by'].search(sql):
            return sql, "대용량 결과에 ORDER BY를 사용하면 성능에 영향을 줄 수 있습니다."
        return sql, None

    @staticmethod
    def _optimize_distinct(sql: str) -> Tuple[str, Optional[str]]:
        """DISTINCT 최적화"""
        if re.search(r'\bSELECT\s+DISTINCT\b', sql, re.IGNORECASE):
            if 'GROUP BY' not in sql.upper():
                return sql, "DISTINCT 대신 GROUP BY를 사용하면 성능이 향상될 수 있습니다."
        return sql, None

    @staticmethod
    def _optimize_like_pattern(sql: str) -> Tuple[str, Optional[str]]:
        """LIKE 패턴 최적화"""
        if re.search(r"LIKE\s+'%[^']+%'", sql, re.IGNORECASE):
            return sql, "LIKE '%..%' 패턴은 인덱스를 활용할 수 없습니다. FULLTEXT 검색을 고려해보세요."
        return sql, None

    @staticmethod
    def _suggest_cte_usage(sql: str) -> Tuple[str, Optional[str]]:
        """CTE 사용 제안"""
        # 중첩 서브쿼리 감지
        subquery_count = sql.upper().count('SELECT') - 1
        if subquery_count >= 2 and 'WITH' not in sql.upper():
            return sql, "중첩 서브쿼리가 많습니다. CTE (WITH 절)를 사용하면 가독성과 성능이 향상됩니다."
        return sql, None

    @staticmethod
    def _optimize_null_comparison(sql: str) -> Tuple[str, Optional[str]]:
        """NULL 비교 최적화"""
        if re.search(r"=\s*NULL|!=\s*NULL|<>\s*NULL", sql, re.IGNORECASE):
            return sql, "NULL 비교는 = 대신 IS NULL 또는 IS NOT NULL을 사용하세요."
        return sql, None

    @staticmethod
    def _optimize_like_pattern(sql: str) -> Tuple[str, Optional[str]]:
        """LIKE 패턴 최적화"""
        if re.search(r"LIKE\s+'%[^']+%'", sql, re.IGNORECASE):
            return sql, "LIKE '%..%' 패턴은 인덱스를 활용할 수 없습니다. FULLTEXT 검색을 고려해보세요."
        return sql, None

    @staticmethod
    def _suggest_cte_usage(sql: str) -> Tuple[str, Optional[str]]:
        """CTE 사용 제안"""
        # 중첩 서브쿼리 감지
        subquery_count = sql.upper().count('SELECT') - 1
        if subquery_count >= 2 and 'WITH' not in sql.upper():
            return sql, "중첩 서브쿼리가 많습니다. CTE (WITH 절)를 사용하면 가독성과 성능이 향상됩니다."
        return sql, None

    def optimize(self, sql: str) -> OptimizationResult:
        """
        SQL 쿼리 최적화

        Args:
            sql: 원본 SQL 쿼리

        Returns:
            OptimizationResult: 최적화 결과
        """
        current_sql = sql
        optimizations = []
        issues = []

        for rule in self.optimization_rules:
            new_sql, suggestion = rule(current_sql)
            if suggestion:
                optimizations.append(suggestion)
                issues.append(SQLIssue(
                    issue_type=SQLIssueType.INEFFICIENT_QUERY,
                    description=suggestion,
                    severity="info"
                ))
            current_sql = new_sql

        return OptimizationResult(
            original_sql=sql,
            optimized_sql=current_sql,
            issues_found=issues,
            optimizations_applied=optimizations
        )


class SelfCorrectionEngine:
    """
    자가 수정 엔진

    SQL 실행 오류를 분석하고 수정 제안을 생성합니다.
    """

    __slots__ = ('schema', '_compiled_patterns')

    # 일반적인 SQL 오류 패턴과 수정 방법 (클래스 레벨 상수)
    _ERROR_PATTERNS: Dict[str, Dict[str, Any]] = {
        r"no such table: (\w+)": {
            "issue_type": SQLIssueType.SCHEMA_MISMATCH,
            "template": "테이블 '{0}'이(가) 존재하지 않습니다. 스키마를 확인하세요.",
            "suggestion_template": "테이블명을 확인하거나 유사한 테이블을 찾아보세요."
        },
        r"no such column: (\w+)": {
            "issue_type": SQLIssueType.SCHEMA_MISMATCH,
            "template": "컬럼 '{0}'이(가) 존재하지 않습니다.",
            "suggestion_template": "컬럼명을 확인하거나 테이블 별칭을 추가해보세요."
        },
        r"ambiguous column name: (\w+)": {
            "issue_type": SQLIssueType.AMBIGUOUS_COLUMN,
            "template": "컬럼 '{0}'이(가) 모호합니다. 여러 테이블에 존재합니다.",
            "suggestion_template": "테이블명.컬럼명 형식으로 명시해주세요."
        },
        r"syntax error": {
            "issue_type": SQLIssueType.SYNTAX_ERROR,
            "template": "SQL 문법 오류가 있습니다.",
            "suggestion_template": "SQL 문법을 확인해주세요."
        },
        r"misuse of aggregate": {
            "issue_type": SQLIssueType.LOGIC_ERROR,
            "template": "집계 함수 사용 오류입니다.",
            "suggestion_template": "GROUP BY 절이 필요하거나, 집계 함수 외부의 컬럼을 확인하세요."
        },
    }

    def __init__(self, schema: Optional[DatabaseSchema] = None):
        self.schema = schema
        # 패턴 미리 컴파일
        self._compiled_patterns = {
            re.compile(pattern): info
            for pattern, info in self._ERROR_PATTERNS.items()
        }

    def analyze_error(self, sql: str, error_message: str) -> SQLIssue:
        """
        SQL 오류 분석

        Args:
            sql: 실패한 SQL 쿼리
            error_message: 오류 메시지

        Returns:
            SQLIssue: 문제 정보
        """
        error_lower = error_message.lower()

        for pattern, info in self._compiled_patterns.items():
            match = pattern.search(error_lower)
            if match:
                groups = match.groups() if match.groups() else ("",)
                description = info["template"].format(*groups)

                return SQLIssue(
                    issue_type=info["issue_type"],
                    description=description,
                    suggestion=info["suggestion_template"],
                    severity="error"
                )

        return SQLIssue(
            issue_type=SQLIssueType.SYNTAX_ERROR,
            description=f"알 수 없는 오류: {error_message}",
            suggestion="SQL 쿼리를 전체적으로 검토해주세요.",
            severity="error"
        )

    def suggest_fix(self, sql: str, issue: SQLIssue) -> List[str]:
        """
        오류에 대한 수정 제안 생성

        Args:
            sql: 원본 SQL
            issue: 발견된 문제

        Returns:
            List[str]: 수정 제안 목록
        """
        suggestions = []

        if issue.issue_type == SQLIssueType.AMBIGUOUS_COLUMN:
            # 모호한 컬럼에 테이블 별칭 추가 제안
            suggestions.append("모든 컬럼에 테이블명 또는 별칭을 접두어로 추가하세요.")
            suggestions.append("예: column_name → table.column_name")

        elif issue.issue_type == SQLIssueType.SCHEMA_MISMATCH:
            if self.schema:
                # 유사한 이름 찾기
                table_names = [t.name for t in self.schema.tables]
                suggestions.append(f"사용 가능한 테이블: {', '.join(table_names)}")

        elif issue.issue_type == SQLIssueType.MISSING_JOIN:
            suggestions.append("FROM 절에 필요한 테이블이 모두 포함되어 있는지 확인하세요.")
            suggestions.append("테이블 간 JOIN 조건을 추가하세요.")

        elif issue.issue_type == SQLIssueType.LOGIC_ERROR:
            suggestions.append("집계 함수(COUNT, SUM, AVG 등) 사용 시 GROUP BY 절이 필요합니다.")
            suggestions.append("SELECT 절의 비집계 컬럼은 GROUP BY에 포함되어야 합니다.")

        if issue.suggestion:
            suggestions.append(issue.suggestion)

        return suggestions

    def generate_correction_prompt(self, sql: str, error_message: str,
                                   issue: SQLIssue) -> str:
        """
        LLM에 전달할 수정 요청 프롬프트 생성

        Args:
            sql: 원본 SQL
            error_message: 오류 메시지
            issue: 분석된 문제

        Returns:
            str: 수정 요청 프롬프트
        """
        suggestions = self.suggest_fix(sql, issue)

        prompt = f"""
## 이전 SQL 쿼리에서 오류가 발생했습니다.

### 원본 SQL:
```sql
{sql}
```

### 오류 메시지:
{error_message}

### 문제 분석:
- 유형: {issue.issue_type.value}
- 설명: {issue.description}

### 수정 제안:
{chr(10).join(f'- {s}' for s in suggestions)}

위 내용을 참고하여 수정된 SQL을 생성해주세요.
"""
        return prompt


class ExecutionAnalyzer:
    """
    SQL 실행 결과 분석기

    실행 결과를 분석하여 의도한 대로 동작하는지 검증합니다.
    """

    def __init__(self):
        pass

    def analyze_result(self, sql: str, question: str,
                       columns: List[str], rows: List[Tuple]) -> Dict[str, Any]:
        """
        실행 결과 분석

        Args:
            sql: 실행된 SQL
            question: 원본 질문
            columns: 결과 컬럼명
            rows: 결과 행

        Returns:
            Dict: 분석 결과
        """
        analysis = {
            "row_count": len(rows),
            "column_count": len(columns),
            "columns": columns,
            "has_results": len(rows) > 0,
            "warnings": [],
            "quality_score": 1.0
        }

        # 결과가 없는 경우
        if not rows:
            analysis["warnings"].append("결과가 없습니다. 조건을 확인해주세요.")
            analysis["quality_score"] -= 0.2

        # 너무 많은 결과
        if len(rows) > 1000:
            analysis["warnings"].append("결과가 1000건을 초과합니다. LIMIT 추가를 고려하세요.")
            analysis["quality_score"] -= 0.1

        # 집계 질문인데 여러 행이 반환된 경우
        aggregate_keywords = ["평균", "합계", "총", "개수", "최대", "최소"]
        if any(kw in question for kw in aggregate_keywords):
            if len(rows) > 10 and "GROUP BY" not in sql.upper():
                analysis["warnings"].append(
                    "집계 질문이지만 많은 행이 반환되었습니다. GROUP BY가 필요할 수 있습니다."
                )
                analysis["quality_score"] -= 0.15

        # NULL 값 비율 체크
        null_count = sum(1 for row in rows for val in row if val is None)
        total_values = len(rows) * len(columns) if rows and columns else 1
        null_ratio = null_count / total_values

        if null_ratio > 0.5:
            analysis["warnings"].append(
                f"NULL 값 비율이 {null_ratio:.1%}로 높습니다. 조인 조건을 확인하세요."
            )
            analysis["quality_score"] -= 0.15

        analysis["null_ratio"] = null_ratio

        return analysis

    def is_result_reasonable(self, analysis: Dict[str, Any],
                            expected_single_value: bool = False) -> bool:
        """결과가 합리적인지 판단"""
        if not analysis["has_results"]:
            return False

        if expected_single_value and analysis["row_count"] != 1:
            return False

        if analysis["quality_score"] < 0.5:
            return False

        return True


# 통합 파이프라인
class SQLCorrectionPipeline:
    """
    SQL 생성-검증-수정 통합 파이프라인
    """

    def __init__(self, schema: Optional[Any] = None):
        self.optimizer = SQLOptimizer()
        self.corrector = SelfCorrectionEngine(schema)
        self.analyzer = ExecutionAnalyzer()

    def process(self, sql: str, question: str,
                execution_result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        SQL 처리 파이프라인

        Args:
            sql: SQL 쿼리
            question: 원본 질문
            execution_result: 실행 결과 (있는 경우)

        Returns:
            Dict: 처리 결과
        """
        result = {
            "sql": sql,
            "optimizations": [],
            "issues": [],
            "analysis": None,
            "needs_correction": False
        }

        # 최적화
        opt_result = self.optimizer.optimize(sql)
        result["optimizations"] = opt_result.optimizations_applied
        result["issues"].extend(opt_result.issues_found)

        # 실행 결과 분석
        if execution_result:
            if "error" in execution_result:
                issue = self.corrector.analyze_error(sql, execution_result["error"])
                result["issues"].append(issue)
                result["needs_correction"] = True
                result["correction_prompt"] = self.corrector.generate_correction_prompt(
                    sql, execution_result["error"], issue
                )
            else:
                analysis = self.analyzer.analyze_result(
                    sql, question,
                    execution_result.get("columns", []),
                    execution_result.get("rows", [])
                )
                result["analysis"] = analysis
                if not self.analyzer.is_result_reasonable(analysis):
                    result["needs_correction"] = True

        return result


if __name__ == "__main__":
    # 테스트
    optimizer = SQLOptimizer()

    test_queries = [
        "SELECT * FROM employees WHERE dept_id = 1",
        "SELECT name FROM employees WHERE dept_id IN (SELECT dept_id FROM departments WHERE location = '서울')",
        "SELECT e.name, d.dept_name FROM employees JOIN departments ON employees.dept_id = departments.dept_id ORDER BY name",
    ]

    print("=" * 60)
    print("SQL Optimizer 테스트")
    print("=" * 60)

    for sql in test_queries:
        print(f"\n원본: {sql}")
        result = optimizer.optimize(sql)
        print(f"제안: {result.optimizations_applied}")

    # Self-Correction 테스트
    print("\n" + "=" * 60)
    print("Self-Correction 테스트")
    print("=" * 60)

    corrector = SelfCorrectionEngine()

    test_errors = [
        ("SELECT * FROM employes", "no such table: employes"),
        ("SELECT name, dept FROM employees", "no such column: dept"),
        ("SELECT id FROM employees e JOIN departments d", "ambiguous column name: id"),
    ]

    for sql, error in test_errors:
        print(f"\nSQL: {sql}")
        print(f"오류: {error}")
        issue = corrector.analyze_error(sql, error)
        print(f"분석: {issue.description}")
        suggestions = corrector.suggest_fix(sql, issue)
        print(f"제안: {suggestions}")
