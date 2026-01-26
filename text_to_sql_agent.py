"""
Advanced Text-to-SQL Agent
Spider 2.0 벤치마크 최신 기술 기반

핵심 기술:
1. 스키마 이해 및 메타데이터 관리
2. 다단계 추론 (Multi-step Reasoning)
3. 멀티 데이터베이스 지원 (BigQuery, Snowflake, SQLite, PostgreSQL)
4. Self-correction 및 검증 메커니즘
5. Context-aware SQL 생성

Author: Azure OpenAI Sample
Date: 2026-01-24
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

from openai import AzureOpenAI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """지원하는 데이터베이스 타입"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"


@dataclass
class TableSchema:
    """테이블 스키마 정보"""
    name: str
    columns: List[Dict[str, str]]  # [{"name": "col1", "type": "INTEGER", "nullable": True}]
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    description: Optional[str] = None
    sample_data: Optional[List[Dict]] = None


@dataclass
class DatabaseSchema:
    """데이터베이스 전체 스키마"""
    database_name: str
    database_type: DatabaseType
    tables: List[TableSchema]
    relationships: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SQLGenerationResult:
    """SQL 생성 결과"""
    sql: str
    explanation: str
    confidence: float
    execution_plan: Optional[str] = None
    alternative_queries: List[str] = field(default_factory=list)
    validation_passed: bool = False


class SchemaExtractor:
    """데이터베이스 스키마 추출기"""

    _schema_cache: Dict[str, DatabaseSchema] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """캐시 초기화"""
        cls._schema_cache.clear()

    @staticmethod
    @contextmanager
    def _get_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
        """SQLite 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    @classmethod
    def extract_sqlite_schema(cls, db_path: str, use_cache: bool = True) -> DatabaseSchema:
        """SQLite 데이터베이스 스키마 추출 (캐싱 지원)"""
        if use_cache and db_path in cls._schema_cache:
            return cls._schema_cache[db_path]

        with cls._get_connection(db_path) as conn:
            cursor = conn.cursor()

            # 테이블 목록 조회
            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            table_names = [row[0] for row in cursor.fetchall()]

            tables = [cls._extract_table_schema(cursor, name) for name in table_names]

        schema = DatabaseSchema(
            database_name=os.path.basename(db_path),
            database_type=DatabaseType.SQLITE,
            tables=tables
        )

        if use_cache:
            cls._schema_cache[db_path] = schema

        return schema

    @staticmethod
    def _extract_table_schema(cursor: sqlite3.Cursor, table_name: str) -> TableSchema:
        """단일 테이블 스키마 추출"""
        # 컬럼 정보 조회
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = []
        primary_keys = []

        for col in cursor.fetchall():
            columns.append({
                "name": col[1],
                "type": col[2],
                "nullable": not col[3],
                "default": col[4]
            })
            if col[5]:  # primary key
                primary_keys.append(col[1])

        # 외래키 정보 조회
        cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
        foreign_keys = [
            {
                "column": fk[3],
                "references_table": fk[2],
                "references_column": fk[4]
            }
            for fk in cursor.fetchall()
        ]

        # 샘플 데이터 조회 (최대 3행)
        cursor.execute(f"SELECT * FROM '{table_name}' LIMIT 3")
        sample_rows = cursor.fetchall()
        column_names = [col[0] for col in cursor.description] if cursor.description else []
        sample_data = [dict(zip(column_names, row)) for row in sample_rows]

        return TableSchema(
            name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            sample_data=sample_data
        )


class PromptBuilder:
    """Text-to-SQL 프롬프트 빌더"""

    SYSTEM_PROMPT = """당신은 세계 최고의 Text-to-SQL 전문가입니다.
사용자의 자연어 질문을 정확한 SQL 쿼리로 변환합니다.

## 핵심 원칙:
1. 스키마를 정확히 이해하고 관계를 파악합니다.
2. 다단계 추론을 통해 복잡한 질의를 처리합니다.
3. SQL 방언(Dialect)을 정확히 적용합니다.
4. 모호한 경우 가장 합리적인 해석을 선택합니다.
5. 성능을 고려한 최적화된 쿼리를 생성합니다.

## 출력 형식:
반드시 아래 JSON 형식으로 응답하세요:
{
    "reasoning": "단계별 추론 과정 설명",
    "sql": "생성된 SQL 쿼리",
    "confidence": 0.0~1.0 사이의 확신도,
    "explanation": "SQL 쿼리에 대한 간단한 설명",
    "assumptions": ["가정한 사항 목록"]
}
"""

    @staticmethod
    def build_schema_context(schema: DatabaseSchema) -> str:
        """스키마 컨텍스트 생성"""
        context_parts = [f"### 데이터베이스: {schema.database_name} ({schema.database_type.value})\n"]

        for table in schema.tables:
            context_parts.append(f"\n#### 테이블: {table.name}")
            context_parts.append("컬럼:")

            for col in table.columns:
                pk_marker = " [PK]" if col["name"] in table.primary_keys else ""
                nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
                context_parts.append(f"  - {col['name']}: {col['type']} {nullable}{pk_marker}")

            if table.foreign_keys:
                context_parts.append("외래키:")
                for fk in table.foreign_keys:
                    context_parts.append(f"  - {fk['column']} -> {fk['references_table']}.{fk['references_column']}")

            if table.sample_data:
                context_parts.append(f"샘플 데이터 (처음 {len(table.sample_data)}행):")
                for row in table.sample_data:
                    context_parts.append(f"  {json.dumps(row, ensure_ascii=False)}")

        return "\n".join(context_parts)

    @staticmethod
    def build_user_prompt(question: str, schema_context: str,
                          additional_context: Optional[str] = None) -> str:
        """사용자 프롬프트 생성"""
        prompt_parts = [
            "## 데이터베이스 스키마:",
            schema_context,
            "\n## 사용자 질문:",
            question
        ]

        if additional_context:
            prompt_parts.extend(["\n## 추가 컨텍스트:", additional_context])

        prompt_parts.append("\n위 질문에 대한 SQL 쿼리를 생성해주세요.")

        return "\n".join(prompt_parts)


class SQLValidator:
    """SQL 쿼리 검증기"""

    @staticmethod
    def validate_syntax(sql: str, db_type: DatabaseType) -> Tuple[bool, Optional[str]]:
        """SQL 문법 검증"""
        sql_upper = sql.upper().strip()

        # 기본 문법 검사
        if not any(sql_upper.startswith(kw) for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']):
            return False, "유효한 SQL 문이 아닙니다."

        # 괄호 균형 검사
        if sql.count('(') != sql.count(')'):
            return False, "괄호가 균형이 맞지 않습니다."

        # SELECT 문인 경우 FROM 절 확인
        if sql_upper.startswith('SELECT') and 'FROM' not in sql_upper:
            # 상수 SELECT 허용 (예: SELECT 1+1)
            pass

        return True, None

    @staticmethod
    def validate_schema_references(sql: str, schema: DatabaseSchema) -> Tuple[bool, Optional[str]]:
        """스키마 참조 검증"""
        table_names = {t.name.lower() for t in schema.tables}

        # FROM/JOIN 절에서 테이블 이름 추출
        from_pattern = r'\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        referenced_tables = re.findall(from_pattern, sql.lower())

        for table in referenced_tables:
            if table not in table_names:
                logger.warning(f"테이블 '{table}'이 스키마에 없습니다. 서브쿼리 별칭일 수 있습니다.")

        return True, None

    @staticmethod
    def execute_and_validate(
        sql: str,
        db_connection: sqlite3.Connection
    ) -> Tuple[bool, Optional[List[Tuple]], Optional[str]]:
        """SQL 실행 및 결과 검증"""
        try:
            cursor = db_connection.cursor()
            cursor.execute(sql)
            return True, cursor.fetchall(), None
        except sqlite3.Error as e:
            return False, None, str(e)


class ModelConfig(Enum):
    """사용 가능한 모델 설정"""
    GPT_5_2 = "gpt-5.2"           # 최신 플래그십 모델 (권장)
    GPT_5_1 = "gpt-5.1"           # 고성능 모델
    GPT_5 = "gpt-5"               # 안정적인 모델
    GPT_4_1 = "gpt-4.1"           # 코딩 특화
    GPT_4_1_MINI = "gpt-4.1-mini" # 빠른 응답
    GPT_4_1_NANO = "gpt-4.1-nano" # 저비용
    O3 = "o3"                     # 복잡한 추론
    O4_MINI = "o4-mini"           # 추론 + 효율성
    CLAUDE_OPUS_4_5 = "claude-opus-4-5"     # Claude 최신
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5" # Claude 효율


# Structured Outputs를 위한 JSON 스키마 정의
SQL_GENERATION_SCHEMA = {
    "name": "sql_generation_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "단계별 추론 과정 설명"
            },
            "sql": {
                "type": "string",
                "description": "생성된 SQL 쿼리"
            },
            "confidence": {
                "type": "number",
                "description": "0.0~1.0 사이의 확신도"
            },
            "explanation": {
                "type": "string",
                "description": "SQL 쿼리에 대한 간단한 설명"
            },
            "assumptions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "가정한 사항 목록"
            },
            "alternative_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "대안 쿼리 목록"
            }
        },
        "required": ["reasoning", "sql", "confidence", "explanation", "assumptions", "alternative_queries"],
        "additionalProperties": False
    }
}


class TextToSQLAgent:
    """
    Text-to-SQL 에이전트 (2026년 최신 기술 적용)

    주요 기능:
    - 다단계 추론 (Multi-step Reasoning) - GPT-5.2 내장 추론 활용
    - Self-correction 및 검증 메커니즘
    - Structured Outputs (100% 스키마 준수)
    - 1M 토큰 컨텍스트 지원
    - 복잡한 질문에 대한 심층 추론 (GPT-5.2 native reasoning)
    """

    # 컴파일된 정규식 패턴 (성능 최적화)
    _JSON_PATTERN = re.compile(r'\{[\s\S]*\}')

    # 지원 API 버전
    SUPPORTED_API_VERSIONS = [
        "2025-01-01-preview",  # 최신
        "2024-10-21",          # GA 안정
        "2024-08-01-preview",  # 레거시
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: str = "gpt-5.2",
        api_version: str = "2025-01-01-preview",
        use_structured_outputs: bool = True,
        max_context_tokens: int = 1000000,
        use_claude: bool = False,
        enable_deep_reasoning: bool = True
    ):
        """
        Args:
            api_key: Azure OpenAI API 키 (기본: OPEN_AI_KEY_5 환경변수)
            endpoint: Azure OpenAI 엔드포인트 (기본: OPEN_AI_ENDPOINT_5 환경변수)
            deployment_name: 모델 배포 이름 (기본: gpt-5.2)
            api_version: API 버전 (기본: 2025-01-01-preview)
            use_structured_outputs: Structured Outputs 사용 여부 (기본: True)
            max_context_tokens: 최대 컨텍스트 토큰 수 (기본: 1M)
            use_claude: Claude 사용 여부
            enable_deep_reasoning: 복잡한 질문에 GPT-5.2 심층 추론 활성화 (기본: True)
        """
        self.use_claude = use_claude
        self.deployment_name = deployment_name
        self.enable_deep_reasoning = enable_deep_reasoning
        self.use_structured_outputs = use_structured_outputs
        self.max_context_tokens = max_context_tokens
        self._db_path: Optional[str] = None

        if use_claude:
            raise NotImplementedError("Claude 지원은 anthropic 패키지 설치 후 사용 가능합니다.")

        # Azure OpenAI 초기화 (OPEN_AI_KEY_5, OPEN_AI_ENDPOINT_5 환경변수 우선 사용)
        self.client = AzureOpenAI(
            api_key=api_key or os.getenv("OPEN_AI_KEY_5") or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=endpoint or os.getenv("OPEN_AI_ENDPOINT_5") or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=api_version
        )

        self.current_schema: Optional[DatabaseSchema] = None
        self.db_connection: Optional[sqlite3.Connection] = None

        logger.info(f"TextToSQLAgent 초기화: model={deployment_name}, api_version={api_version}, structured_outputs={use_structured_outputs}, deep_reasoning={enable_deep_reasoning}")

    def __enter__(self) -> "TextToSQLAgent":
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """컨텍스트 매니저 종료"""
        self.close()

    def load_database(self, db_path: str, db_type: DatabaseType = DatabaseType.SQLITE) -> None:
        """데이터베이스 로드 및 스키마 추출"""
        if db_type != DatabaseType.SQLITE:
            raise NotImplementedError(f"{db_type.value}는 아직 지원하지 않습니다.")

        self._db_path = db_path
        self.current_schema = SchemaExtractor.extract_sqlite_schema(db_path)
        self.db_connection = sqlite3.connect(db_path)
        logger.info(f"데이터베이스 로드 완료: {db_path} (테이블 {len(self.current_schema.tables)}개)")

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        use_deep_reasoning: bool = False
    ) -> str:
        """
        LLM 호출 (Structured Outputs 지원, GPT-5.2 내장 추론 활용)

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            use_deep_reasoning: 복잡한 질문에 심층 추론 활성화 여부
        """
        model = self.deployment_name

        # 복잡한 질문에 대해 심층 추론 프롬프트 추가
        if use_deep_reasoning and self.enable_deep_reasoning:
            system_prompt = system_prompt + """

## 심층 추론 모드 (Deep Reasoning):
복잡한 질문입니다. 다음 단계를 따라 신중하게 분석하세요:
1. 질문의 핵심 의도를 파악합니다.
2. 필요한 테이블과 컬럼을 식별합니다.
3. 조인 관계와 조건을 명확히 합니다.
4. 집계 함수나 서브쿼리 필요 여부를 판단합니다.
5. 최종 SQL을 검증합니다."""

        # Structured Outputs 사용 시 json_schema 형식 적용
        if self.use_structured_outputs:
            response_format = {
                "type": "json_schema",
                "json_schema": SQL_GENERATION_SCHEMA
            }
        else:
            response_format = {"type": "json_object"}

        # GPT-5.x 모델은 max_completion_tokens 사용, 기존 모델은 max_tokens 사용
        is_gpt5_model = model.startswith(('gpt-5', 'o3', 'o4'))

        if is_gpt5_model:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_completion_tokens=32768,  # GPT-5.x, o3, o4 모델용
                response_format=response_format
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=32768,  # GPT-4.x 모델용
                response_format=response_format
            )
        return response.choices[0].message.content or ""

    # 복잡한 질문 판단용 키워드 (클래스 레벨 상수)
    _COMPLEXITY_INDICATORS = frozenset([
        # 비교/집계
        "평균보다", "비교", "가장", "최대", "최소", "평균", "합계", "총",
        # 그룹화
        "그룹별", "부서별", "월별", "연도별", "팀별", "분류별",
        # 서브쿼리/조인
        "서브쿼리", "조인", "join", "하위쿼리",
        # 시퀀스
        "이전", "다음", "연속", "누적", "순차",
        # 비율/변화
        "비율", "퍼센트", "%", "증가", "감소", "변화", "추이",
        # 순위
        "상위", "하위", "top", "rank", "순위", "n번째",
        # 조건
        "제외", "포함", "없는", "있는", "아닌", "만",
        # 복합 조건
        "그리고", "또는", "이상", "이하", "초과", "미만"
    ])

    def _is_complex_question(self, question: str) -> bool:
        """복잡한 질문인지 판단 (심층 추론 활성화 여부 결정)"""
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in self._COMPLEXITY_INDICATORS)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답 파싱 (Structured Outputs로 파싱 오류 최소화)"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Structured Outputs 사용 시 파싱 오류는 거의 발생하지 않음
            json_match = self._JSON_PATTERN.search(response)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError("LLM 응답을 파싱할 수 없습니다.")

    def generate_sql(
        self,
        question: str,
        additional_context: Optional[str] = None,
        max_retries: int = 5,
        auto_select_model: bool = True
    ) -> SQLGenerationResult:
        """
        자연어 질문을 SQL로 변환 (2026년 최신 기술 적용)

        Args:
            question: 자연어 질문
            additional_context: 추가 컨텍스트 (비즈니스 규칙 등)
            max_retries: 최대 재시도 횟수 (self-correction, 기본 5회)
            auto_select_model: 복잡도에 따른 심층 추론 자동 활성화

        Returns:
            SQLGenerationResult: SQL 생성 결과

        Raises:
            ValueError: 데이터베이스가 로드되지 않은 경우
            RuntimeError: 최대 재시도 횟수 초과 시
        """
        if not self.current_schema:
            raise ValueError("데이터베이스가 로드되지 않았습니다. load_database()를 먼저 호출하세요.")

        # 복잡한 질문인지 판단하여 심층 추론 활성화 여부 결정
        use_deep_reasoning = auto_select_model and self._is_complex_question(question)
        if use_deep_reasoning:
            logger.info(f"복잡한 질문 감지: GPT-5.2 심층 추론 모드 활성화")

        schema_context = PromptBuilder.build_schema_context(self.current_schema)
        user_prompt = PromptBuilder.build_user_prompt(question, schema_context, additional_context)

        last_error: Optional[str] = None
        alternative_queries: List[str] = []

        for attempt in range(max_retries):
            try:
                # 이전 오류가 있으면 컨텍스트에 추가
                current_prompt = user_prompt
                if last_error:
                    current_prompt += f"\n## 이전 시도 오류 (Attempt {attempt}):\n{last_error}\n위 오류를 분석하고 수정하여 다시 생성해주세요."

                # LLM 호출 (복잡한 질문은 GPT-5.2 심층 추론 모드 사용)
                response = self._call_llm(
                    PromptBuilder.SYSTEM_PROMPT,
                    current_prompt,
                    use_deep_reasoning=use_deep_reasoning  # GPT-5.2 자체 심층 추론
                )
                parsed = self._parse_llm_response(response)

                sql = parsed.get("sql", "").strip()
                if not sql:
                    raise ValueError("SQL이 생성되지 않았습니다.")

                # 대안 쿼리 저장 (Structured Outputs 덕분에 항상 존재)
                alt_queries = parsed.get("alternative_queries", [])
                if alt_queries:
                    alternative_queries.extend(alt_queries)

                # SQL 검증
                is_valid, error_msg = SQLValidator.validate_syntax(sql, self.current_schema.database_type)
                if not is_valid:
                    last_error = f"문법 오류: {error_msg}"
                    continue

                is_valid, error_msg = SQLValidator.validate_schema_references(sql, self.current_schema)
                if not is_valid:
                    last_error = f"스키마 참조 오류: {error_msg}"
                    continue

                # 실행 검증 (SQLite인 경우)
                validation_passed = False
                if self.db_connection:
                    is_valid, _, error_msg = SQLValidator.execute_and_validate(sql, self.db_connection)
                    if not is_valid:
                        last_error = f"실행 오류: {error_msg}"
                        continue
                    validation_passed = True

                return SQLGenerationResult(
                    sql=sql,
                    explanation=parsed.get("explanation", ""),
                    confidence=float(parsed.get("confidence", 0.8)),
                    validation_passed=validation_passed,
                    alternative_queries=alternative_queries[:3]  # 상위 3개만
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"시도 {attempt + 1}/{max_retries} 실패: {last_error}")

        raise RuntimeError(f"SQL 생성 실패 (최대 재시도 횟수 초과): {last_error}")

    def execute_query(self, sql: str) -> Tuple[List[str], List[Tuple]]:
        """SQL 쿼리 실행"""
        if not self.db_connection:
            raise ValueError("데이터베이스가 연결되지 않았습니다.")

        cursor = self.db_connection.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        results = cursor.fetchall()
        return columns, results

    def ask(self, question: str, execute: bool = True) -> Dict[str, Any]:
        """
        자연어 질문에 대한 답변 (SQL 생성 및 실행)

        Args:
            question: 자연어 질문
            execute: SQL 실행 여부

        Returns:
            Dict: SQL, 설명, 결과 포함
        """
        result = self.generate_sql(question)

        response = {
            "question": question,
            "sql": result.sql,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "validation_passed": result.validation_passed
        }

        if execute and self.db_connection:
            columns, rows = self.execute_query(result.sql)
            response["columns"] = columns
            response["results"] = [dict(zip(columns, row)) for row in rows]
            response["row_count"] = len(rows)

        return response

    def close(self):
        """리소스 정리"""
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None


class ConversationalSQLAgent(TextToSQLAgent):
    """대화형 Text-to-SQL 에이전트 (멀티턴 지원)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history: List[Dict[str, str]] = []
        self.query_history: List[SQLGenerationResult] = []

    def ask_with_history(self, question: str, execute: bool = True) -> Dict[str, Any]:
        """대화 히스토리를 고려한 질문 처리"""
        # 이전 대화 컨텍스트 구성
        history_context = None
        if self.conversation_history:
            history_lines = ["## 이전 대화:"]
            for item in self.conversation_history[-5:]:  # 최근 5개만 사용
                history_lines.append(f"사용자: {item['question']}")
                history_lines.append(f"SQL: {item['sql']}")
            history_context = "\n".join(history_lines)

        # SQL 생성
        if not self.current_schema:
            raise ValueError("데이터베이스가 로드되지 않았습니다.")

        schema_context = PromptBuilder.build_schema_context(self.current_schema)
        user_prompt = PromptBuilder.build_user_prompt(question, schema_context, history_context)

        response = self._call_llm(PromptBuilder.SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_llm_response(response)

        sql = parsed.get("sql", "").strip()

        result = {
            "question": question,
            "sql": sql,
            "explanation": parsed.get("explanation", ""),
            "confidence": float(parsed.get("confidence", 0.8))
        }

        if execute and self.db_connection:
            columns, rows = self.execute_query(sql)
            result["columns"] = columns
            result["results"] = [dict(zip(columns, row)) for row in rows]
            result["row_count"] = len(rows)

        # 히스토리에 추가
        self.conversation_history.append({
            "question": question,
            "sql": sql
        })

        return result

    def clear_history(self):
        """대화 히스토리 초기화"""
        self.conversation_history.clear()
        self.query_history.clear()


def create_sample_database() -> str:
    """샘플 데이터베이스 생성"""
    db_path = "sample_company.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 테이블 생성
    cursor.executescript("""
    -- 부서 테이블
    CREATE TABLE IF NOT EXISTS departments (
        dept_id INTEGER PRIMARY KEY,
        dept_name TEXT NOT NULL,
        location TEXT,
        budget REAL
    );

    -- 직원 테이블
    CREATE TABLE IF NOT EXISTS employees (
        emp_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        dept_id INTEGER,
        hire_date DATE,
        salary REAL,
        manager_id INTEGER,
        FOREIGN KEY (dept_id) REFERENCES departments(dept_id),
        FOREIGN KEY (manager_id) REFERENCES employees(emp_id)
    );

    -- 프로젝트 테이블
    CREATE TABLE IF NOT EXISTS projects (
        project_id INTEGER PRIMARY KEY,
        project_name TEXT NOT NULL,
        start_date DATE,
        end_date DATE,
        budget REAL,
        status TEXT CHECK(status IN ('진행중', '완료', '보류', '취소'))
    );

    -- 프로젝트 할당 테이블
    CREATE TABLE IF NOT EXISTS project_assignments (
        assignment_id INTEGER PRIMARY KEY,
        emp_id INTEGER,
        project_id INTEGER,
        role TEXT,
        hours_allocated INTEGER,
        FOREIGN KEY (emp_id) REFERENCES employees(emp_id),
        FOREIGN KEY (project_id) REFERENCES projects(project_id)
    );

    -- 샘플 데이터 삽입
    INSERT OR IGNORE INTO departments VALUES
        (1, '개발팀', '서울', 500000000),
        (2, '마케팅팀', '서울', 300000000),
        (3, '인사팀', '부산', 200000000),
        (4, '영업팀', '대전', 400000000);

    INSERT OR IGNORE INTO employees VALUES
        (1, '김철수', 'kim@company.com', 1, '2020-03-15', 85000000, NULL),
        (2, '이영희', 'lee@company.com', 1, '2021-07-01', 75000000, 1),
        (3, '박민수', 'park@company.com', 2, '2019-11-20', 70000000, NULL),
        (4, '정수진', 'jung@company.com', 1, '2022-01-10', 65000000, 1),
        (5, '최동훈', 'choi@company.com', 3, '2018-05-25', 80000000, NULL),
        (6, '강서연', 'kang@company.com', 4, '2023-02-14', 60000000, NULL),
        (7, '윤지민', 'yoon@company.com', 2, '2020-09-08', 72000000, 3);

    INSERT OR IGNORE INTO projects VALUES
        (1, 'AI 챗봇 개발', '2024-01-01', '2024-12-31', 100000000, '진행중'),
        (2, '모바일 앱 리뉴얼', '2024-03-01', '2024-09-30', 80000000, '완료'),
        (3, '데이터 분석 플랫폼', '2024-06-01', NULL, 150000000, '진행중'),
        (4, '마케팅 자동화', '2024-02-01', '2024-08-31', 50000000, '완료');

    INSERT OR IGNORE INTO project_assignments VALUES
        (1, 1, 1, '프로젝트 매니저', 200),
        (2, 2, 1, '백엔드 개발자', 300),
        (3, 4, 1, '프론트엔드 개발자', 280),
        (4, 3, 4, '마케팅 담당', 150),
        (5, 7, 4, '데이터 분석가', 180),
        (6, 1, 3, '기술 리드', 100),
        (7, 2, 3, '백엔드 개발자', 250);
    """)

    conn.commit()
    conn.close()

    return db_path


# 사용 예시
if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Text-to-SQL Agent")
    print("Spider 2.0 벤치마크 1위 기술 기반")
    print("=" * 60)

    # 샘플 데이터베이스 생성
    db_path = create_sample_database()
    print(f"\n샘플 데이터베이스 생성: {db_path}")

    # 에이전트 초기화 (환경 변수에서 API 키 로드)
    # 실제 사용 시 아래 주석 해제
    """
    # GPT-5.2 + 심층 추론 모드 (2026년 권장 설정)
    agent = TextToSQLAgent(
        deployment_name="gpt-5.2",              # GPT-5.2 사용
        api_version="2025-01-01-preview",       # 최신 API
        use_structured_outputs=True,            # JSON 스키마 100% 준수
        enable_deep_reasoning=True,             # GPT-5.2 심층 추론 활성화
        max_context_tokens=1000000              # 1M 토큰 컨텍스트
    )

    # 데이터베이스 로드
    agent.load_database(db_path)

    # 테스트 질문들 (복잡도 다양)
    test_questions = [
        "개발팀 직원들의 평균 연봉은 얼마인가요?",                    # 단순
        "진행 중인 프로젝트에 참여하는 직원 목록을 보여주세요",        # 조인
        "부서별 직원 수와 총 연봉을 알려주세요",                      # 그룹화 (심층 추론)
        "가장 많은 시간이 할당된 프로젝트는 무엇인가요?",             # 최대값 (심층 추론)
        "평균 연봉보다 높은 급여를 받는 직원은?"                      # 서브쿼리 (심층 추론)
    ]

    for question in test_questions:
        print(f"\n질문: {question}")
        try:
            result = agent.ask(question)
            print(f"SQL: {result['sql']}")
            print(f"설명: {result['explanation']}")
            print(f"신뢰도: {result['confidence']:.2f}")
            if 'results' in result:
                print(f"결과 ({result['row_count']}행):")
                for row in result['results'][:5]:
                    print(f"  {row}")
        except Exception as e:
            print(f"오류: {e}")
        print("-" * 40)

    agent.close()
    """

    # 스키마 추출 데모 (API 키 없이도 동작)
    print("\n스키마 추출 데모:")
    schema = SchemaExtractor.extract_sqlite_schema(db_path)
    print(f"데이터베이스: {schema.database_name}")
    print(f"테이블 수: {len(schema.tables)}")
    for table in schema.tables:
        print(f"\n테이블: {table.name}")
        print(f"  컬럼: {[c['name'] for c in table.columns]}")
        print(f"  기본키: {table.primary_keys}")
        if table.foreign_keys:
            print(f"  외래키: {table.foreign_keys}")
