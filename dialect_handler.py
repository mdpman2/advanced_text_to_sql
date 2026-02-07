"""
Multi-Database SQL Dialect Handler (v2.2.1)

BigQuery, Snowflake, PostgreSQL, SQLite 등 다양한 SQL 방언을 지원합니다.
Spider 2.0-Snow/Lite 벤치마크의 멀티 데이터베이스 환경 처리가 핵심입니다.

지원 변환:
- SQLite ↔ BigQuery (GROUP_CONCAT ↔ ARRAY_AGG, strftime ↔ FORMAT_DATE)
- SQLite ↔ Snowflake (GROUP_CONCAT ↔ LISTAGG, FLATTEN)
- SQLite ↔ PostgreSQL (GROUP_CONCAT ↔ STRING_AGG, :: 타입캐스팅)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Callable, ClassVar, Dict, List, Optional

# 미리 컴파일된 정규식 패턴 (클래스 간 공유)
_DIALECT_PATTERNS: Dict[str, re.Pattern] = {
    # SQLite
    'date_trunc': re.compile(r"DATE_TRUNC\s*\(\s*'(\w+)'\s*,\s*(\w+)\s*\)", re.IGNORECASE),
    'array_agg': re.compile(r"\bARRAY_AGG\s*\(", re.IGNORECASE),
    # PostgreSQL
    'pg_cast': re.compile(r"(\w+)::(\w+)"),
    'ilike': re.compile(r"\bILIKE\b", re.IGNORECASE),
    # SQLite/BigQuery
    'group_concat': re.compile(r"\bGROUP_CONCAT\s*\(\s*(\w+)\s*\)", re.IGNORECASE),
    'strftime': re.compile(r"strftime\s*\(\s*'([^']+)'\s*,\s*(\w+)\s*\)", re.IGNORECASE),
}


class SQLDialect(Enum):
    """지원하는 SQL 방언"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    SQLSERVER = "sqlserver"


@dataclass(slots=True, frozen=True)
class DialectFeature:
    """불변 SQL 방언 특성"""
    dialect: SQLDialect
    supports_cte: bool = True
    supports_window_functions: bool = True
    supports_array_types: bool = False
    supports_json_types: bool = False
    date_format: str = "%Y-%m-%d"
    string_concat_operator: str = "||"
    limit_syntax: str = "LIMIT"  # LIMIT, TOP, FETCH FIRST
    quote_char: str = '"'  # ", `, [
    
    
class DialectConverter(ABC):
    """SQL 방언 변환기 기본 클래스"""
    
    @abstractmethod
    def convert(self, sql: str, source_dialect: SQLDialect) -> str:
        """SQL을 현재 방언으로 변환"""
        pass
    
    @abstractmethod
    def get_feature(self) -> DialectFeature:
        """방언 특성 반환"""
        pass


class SQLiteDialect(DialectConverter):
    """SQLite 방언 처리기"""
    
    # 클래스 레벨 캐싱된 feature 인스턴스
    _feature: ClassVar[Optional[DialectFeature]] = None
    
    def get_feature(self) -> DialectFeature:
        if SQLiteDialect._feature is None:
            SQLiteDialect._feature = DialectFeature(
                dialect=SQLDialect.SQLITE,
                supports_cte=True,
                supports_window_functions=True,
                supports_array_types=False,
                supports_json_types=True,  # SQLite 3.9+
                date_format="%Y-%m-%d",
                string_concat_operator="||",
                limit_syntax="LIMIT",
                quote_char='"'
            )
        return SQLiteDialect._feature
    
    def convert(self, sql: str, source_dialect: SQLDialect) -> str:
        if source_dialect == SQLDialect.SQLITE:
            return sql
        
        result = sql
        
        # BigQuery/Snowflake 특수 함수 변환
        if source_dialect in (SQLDialect.BIGQUERY, SQLDialect.SNOWFLAKE):
            # DATE_TRUNC -> strftime
            result = _DIALECT_PATTERNS['date_trunc'].sub(
                lambda m: self._convert_date_trunc(m.group(1), m.group(2)),
                result
            )
            
            # ARRAY_AGG -> GROUP_CONCAT
            result = _DIALECT_PATTERNS['array_agg'].sub("GROUP_CONCAT(", result)
        
        # PostgreSQL 특수 구문 변환
        if source_dialect == SQLDialect.POSTGRESQL:
            # :: 타입 캐스팅 -> CAST
            result = _DIALECT_PATTERNS['pg_cast'].sub(r"CAST(\1 AS \2)", result)
            # ILIKE -> LIKE
            result = _DIALECT_PATTERNS['ilike'].sub("LIKE", result)
        
        return result
    
    @staticmethod
    @lru_cache(maxsize=32)
    def _convert_date_trunc(unit: str, column: str) -> str:
        """DATE_TRUNC을 SQLite strftime으로 변환"""
        unit_map = {
            "year": "%Y-01-01",
            "month": "%Y-%m-01",
            "day": "%Y-%m-%d",
            "week": None,  # 특수 처리 필요
        }
        format_str = unit_map.get(unit.lower(), "%Y-%m-%d")
        if format_str:
            return f"strftime('{format_str}', {column})"
        return column


class BigQueryDialect(DialectConverter):
    """BigQuery 방언 처리기"""
    
    _feature: ClassVar[Optional[DialectFeature]] = None
    
    def get_feature(self) -> DialectFeature:
        if BigQueryDialect._feature is None:
            BigQueryDialect._feature = DialectFeature(
                dialect=SQLDialect.BIGQUERY,
                supports_cte=True,
                supports_window_functions=True,
                supports_array_types=True,
                supports_json_types=True,
                date_format="%Y-%m-%d",
                string_concat_operator="||",
                limit_syntax="LIMIT",
                quote_char='`'
            )
        return BigQueryDialect._feature
    
    def convert(self, sql: str, source_dialect: SQLDialect) -> str:
        if source_dialect == SQLDialect.BIGQUERY:
            return sql
        
        result = sql
        
        # SQLite 함수 -> BigQuery 함수
        if source_dialect == SQLDialect.SQLITE:
            # GROUP_CONCAT -> ARRAY_AGG + ARRAY_TO_STRING
            result = _DIALECT_PATTERNS['group_concat'].sub(
                r"ARRAY_TO_STRING(ARRAY_AGG(\1), ',')", result
            )
            
            # strftime -> FORMAT_DATE
            result = _DIALECT_PATTERNS['strftime'].sub(
                lambda m: self._convert_strftime(m.group(1), m.group(2)),
                result
            )
        
        # 큰따옴표 -> 백틱
        result = result.replace('"', '`')
        
        return result
    
    @staticmethod
    @lru_cache(maxsize=32)
    def _convert_strftime(format_str: str, column: str) -> str:
        """SQLite strftime을 BigQuery FORMAT_DATE로 변환"""
        bq_format = format_str.replace("%Y", "%Y").replace("%m", "%m").replace("%d", "%d")
        return f"FORMAT_DATE('{bq_format}', {column})"


class SnowflakeDialect(DialectConverter):
    """Snowflake 방언 처리기"""
    
    _feature: ClassVar[Optional[DialectFeature]] = None
    
    def get_feature(self) -> DialectFeature:
        if SnowflakeDialect._feature is None:
            SnowflakeDialect._feature = DialectFeature(
                dialect=SQLDialect.SNOWFLAKE,
                supports_cte=True,
                supports_window_functions=True,
                supports_array_types=True,
                supports_json_types=True,
                date_format="%Y-%m-%d",
                string_concat_operator="||",
                limit_syntax="LIMIT",
                quote_char='"'
            )
        return SnowflakeDialect._feature
    
    def convert(self, sql: str, source_dialect: SQLDialect) -> str:
        if source_dialect == SQLDialect.SNOWFLAKE:
            return sql
        
        result = sql
        
        # SQLite -> Snowflake
        if source_dialect == SQLDialect.SQLITE:
            # GROUP_CONCAT -> LISTAGG
            result = _DIALECT_PATTERNS['group_concat'].sub(r"LISTAGG(\1, ',')", result)
        
        return result


class PostgreSQLDialect(DialectConverter):
    """PostgreSQL 방언 처리기"""
    
    _feature: ClassVar[Optional[DialectFeature]] = None
    
    def get_feature(self) -> DialectFeature:
        if PostgreSQLDialect._feature is None:
            PostgreSQLDialect._feature = DialectFeature(
                dialect=SQLDialect.POSTGRESQL,
                supports_cte=True,
                supports_window_functions=True,
                supports_array_types=True,
                supports_json_types=True,
                date_format="%Y-%m-%d",
                string_concat_operator="||",
                limit_syntax="LIMIT",
                quote_char='"'
            )
        return PostgreSQLDialect._feature
    
    def convert(self, sql: str, source_dialect: SQLDialect) -> str:
        if source_dialect == SQLDialect.POSTGRESQL:
            return sql
        
        result = sql
        
        # SQLite -> PostgreSQL
        if source_dialect == SQLDialect.SQLITE:
            # GROUP_CONCAT -> STRING_AGG
            result = _DIALECT_PATTERNS['group_concat'].sub(r"STRING_AGG(\1, ',')", result)
        
        return result


class DialectManager:
    """
    SQL 방언 관리자
    
    다양한 데이터베이스 간 SQL 변환을 관리합니다.
    """
    
    def __init__(self):
        self.dialects: Dict[SQLDialect, DialectConverter] = {
            SQLDialect.SQLITE: SQLiteDialect(),
            SQLDialect.BIGQUERY: BigQueryDialect(),
            SQLDialect.SNOWFLAKE: SnowflakeDialect(),
            SQLDialect.POSTGRESQL: PostgreSQLDialect(),
        }
    
    def get_dialect(self, dialect: SQLDialect) -> DialectConverter:
        """특정 방언 처리기 반환"""
        if dialect not in self.dialects:
            raise ValueError(f"지원하지 않는 SQL 방언: {dialect}")
        return self.dialects[dialect]
    
    def convert(self, sql: str, 
                source: SQLDialect, 
                target: SQLDialect) -> str:
        """
        SQL을 한 방언에서 다른 방언으로 변환
        
        Args:
            sql: 원본 SQL
            source: 원본 방언
            target: 대상 방언
        
        Returns:
            str: 변환된 SQL
        """
        if source == target:
            return sql
        
        target_dialect = self.get_dialect(target)
        return target_dialect.convert(sql, source)
    
    def detect_dialect(self, sql: str) -> SQLDialect:
        """
        SQL에서 방언 감지 (휴리스틱)
        
        Args:
            sql: SQL 쿼리
        
        Returns:
            SQLDialect: 감지된 방언
        """
        sql_upper = sql.upper()
        
        # BigQuery 특징
        if "`" in sql or "ARRAY_AGG" in sql_upper or "UNNEST" in sql_upper:
            return SQLDialect.BIGQUERY
        
        # Snowflake 특징
        if "LISTAGG" in sql_upper or "FLATTEN" in sql_upper:
            return SQLDialect.SNOWFLAKE
        
        # PostgreSQL 특징
        if "::" in sql or "ILIKE" in sql_upper or "STRING_AGG" in sql_upper:
            return SQLDialect.POSTGRESQL
        
        # 기본값
        return SQLDialect.SQLITE
    
    def get_dialect_hints(self, dialect: SQLDialect) -> List[str]:
        """
        특정 방언에 대한 힌트 목록 반환
        (LLM 프롬프트에 포함할 정보)
        """
        hints = []
        feature = self.dialects[dialect].get_feature()
        
        if dialect == SQLDialect.BIGQUERY:
            hints.extend([
                "BigQuery는 백틱(`)을 사용하여 테이블/컬럼명을 인용합니다.",
                "날짜 함수: DATE_TRUNC, FORMAT_DATE, DATE_ADD 사용",
                "배열 지원: ARRAY_AGG, UNNEST, ARRAY_TO_STRING",
                "정규식: REGEXP_CONTAINS, REGEXP_EXTRACT 사용",
            ])
        
        elif dialect == SQLDialect.SNOWFLAKE:
            hints.extend([
                "Snowflake는 대소문자를 구분합니다 (큰따옴표 사용 시).",
                "리스트 집계: LISTAGG 함수 사용",
                "날짜 함수: DATE_TRUNC, DATEADD, DATEDIFF",
                "JSON 지원: PARSE_JSON, GET_PATH, FLATTEN",
            ])
        
        elif dialect == SQLDialect.POSTGRESQL:
            hints.extend([
                "PostgreSQL은 :: 연산자로 타입 캐스팅 가능",
                "대소문자 무시 검색: ILIKE 연산자",
                "문자열 집계: STRING_AGG 함수",
                "배열 지원: ARRAY_AGG, ANY, ALL",
            ])
        
        elif dialect == SQLDialect.SQLITE:
            hints.extend([
                "SQLite는 가볍고 타입이 유연합니다.",
                "문자열 집계: GROUP_CONCAT 함수",
                "날짜 함수: date(), datetime(), strftime()",
                "재귀 CTE 지원: WITH RECURSIVE",
            ])
        
        return hints


class MultiDatabaseQuery:
    """
    멀티 데이터베이스 쿼리 생성기
    
    여러 데이터베이스에 대해 동시에 쿼리를 생성할 수 있습니다.
    """
    
    def __init__(self):
        self.dialect_manager = DialectManager()
    
    def generate_for_all_dialects(self, base_sql: str, 
                                  source_dialect: SQLDialect = SQLDialect.SQLITE) -> Dict[SQLDialect, str]:
        """
        기본 SQL을 모든 지원 방언으로 변환
        
        Args:
            base_sql: 기본 SQL (source_dialect 기준)
            source_dialect: 기본 SQL의 방언
        
        Returns:
            Dict[SQLDialect, str]: 방언별 SQL
        """
        results = {}
        
        for dialect in SQLDialect:
            try:
                converted = self.dialect_manager.convert(base_sql, source_dialect, dialect)
                results[dialect] = converted
            except Exception as e:
                results[dialect] = f"-- 변환 실패: {e}\n{base_sql}"
        
        return results
    
    def get_dialect_specific_prompt(self, dialect: SQLDialect) -> str:
        """
        특정 방언에 맞는 프롬프트 생성
        """
        hints = self.dialect_manager.get_dialect_hints(dialect)
        
        prompt = f"""
## SQL 방언: {dialect.value.upper()}

### 주요 특징:
{chr(10).join(f'- {hint}' for hint in hints)}

### 주의사항:
- 위 방언에 맞는 SQL 문법을 사용하세요.
- 해당 데이터베이스에서 지원하지 않는 기능은 피하세요.
"""
        return prompt


# 테스트
if __name__ == "__main__":
    manager = DialectManager()
    multi_db = MultiDatabaseQuery()
    
    # 테스트 쿼리
    test_sql = """
    SELECT 
        d.dept_name,
        GROUP_CONCAT(e.name) as employees,
        AVG(e.salary) as avg_salary
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
    WHERE strftime('%Y', e.hire_date) >= '2020'
    GROUP BY d.dept_name
    ORDER BY avg_salary DESC
    """
    
    print("=" * 60)
    print("SQL 방언 변환 테스트")
    print("=" * 60)
    print(f"\n원본 (SQLite):\n{test_sql}")
    
    # 모든 방언으로 변환
    results = multi_db.generate_for_all_dialects(test_sql, SQLDialect.SQLITE)
    
    for dialect, sql in results.items():
        if dialect != SQLDialect.SQLITE:
            print(f"\n{dialect.value.upper()}:\n{sql}")
    
    # 방언 힌트 출력
    print("\n" + "=" * 60)
    print("BigQuery 프롬프트 힌트:")
    print(multi_db.get_dialect_specific_prompt(SQLDialect.BIGQUERY))
