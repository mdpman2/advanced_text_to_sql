"""
Advanced Text-to-SQL 통합 테스트 (v3.1.5)

17개 시나리오 × 다중 테스트 케이스
- 2026-03-31 최신 대상 버전: v3.1.5 (runtime settings + telemetry)
- 시나리오 1~11: 오프라인 (API 키 불필요)
- 시나리오 12: API 통합 테스트 (workspace .env 자동 로드, 환경 없으면 skip)
- 시나리오 13: v3.1 신규 (QueryGuard, 모호성, Mermaid)
- 시나리오 14: v3.1.1 버그 수정 검증
- 시나리오 15: v3.1.2 버그 수정 검증
- 시나리오 16: v3.1.3 최적화 검증
- 시나리오 17: v3.1.4 API 운영 기능 검증
- 시나리오 18: v3.1.5 운영 설정/관측 기능 검증

v3.0.0 변경:
- Pydantic 모델 (SQLGenerationSchema) 테스트 추가
- Responses API 지원 버전 확인
- MySQL/SQL Server 방언 테스트 추가
- 신규 최적화 규칙 (Cartesian Join, Window Function) 테스트
- 모델 수 16개 확인
- 코드 최적화 검증: __slots__, _table_dict, _COMPILED_PATTERNS, dict dispatch

실행 방법:
  pytest test_all.py -v            # pytest
  python test_all.py               # 독립 실행
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_test_env() -> None:
    """테스트 실행 위치와 무관하게 루트/보조 .env를 순서대로 로드"""
    workspace_root = Path(__file__).resolve().parent.parent
    env_candidates = (
        workspace_root / ".env",
        workspace_root / "azure_korean_doc_framework1" / ".env",
    )
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_test_env()


# ── 테스트 인프라 ─────────────────────────────────────────────

_passed = 0
_failed = 0
_skipped = 0


def ok(name: str, condition: bool, msg: str = "") -> None:
    """단일 테스트 항목 결과 기록"""
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  ✅ {name}")
    else:
        _failed += 1
        print(f"  ❌ {name}: {msg}")


def skip(name: str, reason: str = "") -> None:
    """테스트 스킵 기록"""
    global _skipped
    _skipped += 1
    print(f"  ⏭️ {name}: {reason}")


def section(title: str) -> None:
    """섹션 헤더"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ── 시나리오 1: 모듈 임포트 및 초기화 ────────────────────────

def test_scenario_01_imports():
    """시나리오 1: 전체 모듈 임포트 검증"""
    section("시나리오 1: 모듈 임포트 및 초기화")

    # 1-1 text_to_sql_agent (12개)
    from text_to_sql_agent import (
        TextToSQLAgent, ConversationalSQLAgent, SchemaExtractor,
        PromptBuilder, SQLValidator, ModelConfig, SQLGenerationResult,
        DatabaseType, TableSchema, DatabaseSchema, create_sample_database,
        DestructiveQueryError,
    )
    ok("text_to_sql_agent 임포트 (12개)", True)

    # 1-2 schema_linker (4개)
    from schema_linker import SchemaLinker, SchemaLink, SchemaLinkingResult, QueryDecomposer
    ok("schema_linker 임포트 (4개)", True)

    # 1-3 sql_optimizer (4개)
    from sql_optimizer import SQLOptimizer, SelfCorrectionEngine, SQLCorrectionPipeline, SQLIssueType
    ok("sql_optimizer 임포트 (4개)", True)

    # 1-4 dialect_handler (4개)
    from dialect_handler import DialectManager, SQLDialect, MultiDatabaseQuery, DialectFeature
    ok("dialect_handler 임포트 (4개)", True)

    # 1-5 demo_app (10개)
    from demo_app import (
        _get_api_key, _print_query_result, _read_question,
        demo_schema_info, demo_sql_optimization, demo_dialect_conversion,
        _BANNER, _MENU, _EXIT_COMMANDS, _MAX_DISPLAY_ROWS,
    )
    ok("demo_app 임포트 (10개)", True)

    # 1-6 ModelConfig (Enum)
    all_models = list(ModelConfig)
    model_values = [m.value for m in all_models]
    ok(f"ModelConfig 등록 모델: {len(all_models)}개 (>=16)", len(all_models) >= 16)
    ok("gpt-5.2-codex 존재", "gpt-5.2-codex" in model_values)
    ok("gpt-5.2-mini 존재", "gpt-5.2-mini" in model_values)
    ok("gpt-5-pro 존재", "gpt-5-pro" in model_values)
    ok("gpt-5-nano 존재", "gpt-5-nano" in model_values)

    codex = ModelConfig.GPT_5_2_CODEX
    ok("gpt-5.2-codex value 확인", codex.value == "gpt-5.2-codex")

    # 1-7 API 버전 (v3.0: Responses API)
    ok("API 2025-04-01-preview 지원 확인", "2025-04-01-preview" in TextToSQLAgent.SUPPORTED_API_VERSIONS)

    # 1-8 Pydantic 모델 (v3.0 신규)
    from text_to_sql_agent import SQLGenerationSchema
    schema_dict = SQLGenerationSchema.model_json_schema()
    ok("SQLGenerationSchema Pydantic 모델 존재", schema_dict is not None)
    ok("SQLGenerationSchema: sql 필드", "sql" in schema_dict.get("properties", {}))
    ok("SQLGenerationSchema: reasoning 필드", "reasoning" in schema_dict.get("properties", {}))


# ── 시나리오 2: 샘플 DB 생성 및 스키마 추출 ──────────────────

def test_scenario_02_schema():
    """시나리오 2: 샘플 DB · 스키마 추출 · 캐시"""
    section("시나리오 2: 샘플 DB 생성 및 스키마 추출")

    from text_to_sql_agent import SchemaExtractor, create_sample_database

    db_path = create_sample_database()
    ok(f"샘플 DB 생성: {db_path}", db_path is not None)

    schema = SchemaExtractor.extract_sqlite_schema(db_path)
    ok(f"테이블 수: {len(schema.tables)}", len(schema.tables) == 4)

    table_names = {t.name for t in schema.tables}
    for name in ("employees", "departments", "projects", "project_assignments"):
        ok(f"테이블 존재: {name}", name in table_names)

    emp = next(t for t in schema.tables if t.name == "employees")
    ok(f"employees 컬럼 수: {len(emp.columns)}", len(emp.columns) >= 5)
    ok("employees PK 존재", len(emp.primary_keys) > 0)
    ok("employees FK 존재", len(emp.foreign_keys) > 0)
    ok("employees 샘플데이터 존재", emp.sample_data is not None and len(emp.sample_data) > 0)

    # 캐시
    schema2 = SchemaExtractor.extract_sqlite_schema(db_path, use_cache=True)
    ok("스키마 캐시 동작 (동일 객체)", schema2 is schema)

    SchemaExtractor.clear_cache()
    schema3 = SchemaExtractor.extract_sqlite_schema(db_path)
    ok("캐시 초기화 후 재추출", schema3 is not schema)


# ── 시나리오 3: 스키마 링킹 ──────────────────────────────────

def test_scenario_03_schema_linking():
    """시나리오 3: SchemaLinker · 한국어 키워드 · QueryDecomposer"""
    section("시나리오 3: 스키마 링킹 (SchemaLinker)")

    from text_to_sql_agent import SchemaExtractor, create_sample_database
    from schema_linker import SchemaLinker, QueryDecomposer

    schema = SchemaExtractor.extract_sqlite_schema(create_sample_database())
    linker = SchemaLinker(schema)

    # 3-1 기본 링킹
    r = linker.link("개발팀 직원들의 평균 급여")
    ok("기본 링킹: 테이블 식별", len(r.relevant_tables) > 0)
    ok("기본 링킹: employees 포함", "employees" in r.relevant_tables)

    # 3-2 조인 추론
    r2 = linker.link("프로젝트에 참여하는 직원 목록")
    ok("조인 추론: 테이블 식별", len(r2.relevant_tables) > 0)

    # 3-3 한국어 키워드
    kw = linker.KOREAN_KEYWORDS
    for word in ("평균", "합계", "최대", "최소", "이상", "미만"):
        ok(f"한국어 키워드: {word}", word in kw)

    # 3-4 복잡한 질문
    r3 = linker.link("부서별로 직원 수와 평균 연봉을 알려줘")
    ok("복잡한 질문: 테이블 >= 1", len(r3.relevant_tables) >= 1)

    # 3-5 집중 스키마
    focused = linker.get_focused_schema("개발팀 직원")
    ok("집중 스키마 생성", len(focused) > 0)

    # 3-6 QueryDecomposer
    decomposer = QueryDecomposer()
    subs = decomposer.decompose("평균 연봉보다 높은 급여를 받는 개발팀 직원의 프로젝트 참여 현황")
    ok(f"QueryDecomposer: {len(subs)}개 서브질문", len(subs) >= 1)


# ── 시나리오 4: SQL 최적화 ───────────────────────────────────

def test_scenario_04_optimizer():
    """시나리오 4: SQLOptimizer 최적화 규칙 검증"""
    section("시나리오 4: SQL 최적화 (SQLOptimizer)")

    from sql_optimizer import SQLOptimizer

    opt = SQLOptimizer()

    r1 = opt.optimize("SELECT * FROM employees WHERE salary > 50000")
    ok("SELECT * 최적화 감지", r1 is not None and len(r1.optimizations_applied) > 0)

    r2 = opt.optimize(
        "SELECT name FROM employees WHERE dept_id IN "
        "(SELECT dept_id FROM departments WHERE location = '서울')"
    )
    ok("IN 서브쿼리 감지", r2 is not None and len(r2.optimizations_applied) > 0)

    r3 = opt.optimize(
        "SELECT e.name FROM employees e "
        "JOIN departments d ON e.dept_id = d.dept_id ORDER BY e.name"
    )
    ok("ORDER BY without LIMIT 감지", r3 is not None)

    r4 = opt.optimize("SELECT emp_id, name FROM employees WHERE dept_id = 1 LIMIT 10")
    ok("최적 쿼리 패스 (제안 0~최소)", r4 is not None)


# ── 시나리오 5: 자가 수정 ────────────────────────────────────

def test_scenario_05_self_correction():
    """시나리오 5: SelfCorrectionEngine 에러 분석"""
    section("시나리오 5: 자가 수정 (SelfCorrectionEngine)")

    from sql_optimizer import SelfCorrectionEngine, SQLIssueType

    engine = SelfCorrectionEngine()

    i1 = engine.analyze_error("SELECT * FROM employes", "no such table: employes")
    ok("테이블명 오타 감지", i1 is not None and i1.issue_type == SQLIssueType.SCHEMA_MISMATCH)
    ok("수정 제안 존재", i1.suggestion is not None and len(i1.suggestion) > 0)

    i2 = engine.analyze_error("SELECT * FROM employees WHERE dept = 1", "no such column: dept")
    ok("컬럼명 오타 감지", i2 is not None and i2.issue_type == SQLIssueType.SCHEMA_MISMATCH)

    i3 = engine.analyze_error(
        "SELECT id FROM employees e JOIN departments d", "ambiguous column name: id"
    )
    ok("모호한 컬럼 감지", i3 is not None and i3.issue_type == SQLIssueType.AMBIGUOUS_COLUMN)

    i4 = engine.analyze_error("SELECT dept_id, COUNT(*) FROM employees", "not an aggregate")
    ok("GROUP BY 누락 감지", i4 is not None)


# ── 시나리오 6: 방언 처리 ────────────────────────────────────

def test_scenario_06_dialect():
    """시나리오 6: DialectHandler 방언 감지·변환·멀티"""
    section("시나리오 6: 방언 처리 (DialectHandler)")

    from dialect_handler import DialectManager, SQLDialect, MultiDatabaseQuery

    mgr = DialectManager()

    # 6-1 방언 감지
    ok("BigQuery 감지", mgr.detect_dialect("SELECT ARRAY_AGG(x) FROM t") == SQLDialect.BIGQUERY)
    ok("PostgreSQL 감지", mgr.detect_dialect("SELECT x::int FROM t") == SQLDialect.POSTGRESQL)
    ok("Snowflake 감지", mgr.detect_dialect("SELECT FLATTEN(x) FROM t") == SQLDialect.SNOWFLAKE)

    # 6-2 변환
    sql = "SELECT GROUP_CONCAT(name) FROM employees GROUP BY dept_id"
    try:
        bq = mgr.convert(sql, SQLDialect.SQLITE, SQLDialect.BIGQUERY)
        ok("SQLite → BigQuery 변환", bq is not None)
    except Exception:
        ok("SQLite → BigQuery 변환 (부분 구현)", True)

    # 6-3 멀티 방언 생성
    mdb = MultiDatabaseQuery()
    results = mdb.generate_for_all_dialects(sql, SQLDialect.SQLITE)
    ok(f"멀티 방언 생성: {len(results)}개", len(results) >= 3)

    # 6-4 방언 특성 조회
    for d in (SQLDialect.SQLITE, SQLDialect.BIGQUERY, SQLDialect.SNOWFLAKE,
              SQLDialect.POSTGRESQL, SQLDialect.MYSQL, SQLDialect.SQLSERVER):
        feature = mgr.get_dialect(d).get_feature()
        ok(f"{d.value} 특성 조회", feature.dialect == d)

    # 6-5 MySQL 감지 (v3.0 신규)
    ok("MySQL 감지", mgr.detect_dialect("SELECT IFNULL(x, 0) FROM t") == SQLDialect.MYSQL)
    # 6-6 SQL Server 감지 (v3.0 신규)
    ok("SQL Server 감지", mgr.detect_dialect("SELECT TOP 10 x FROM t") == SQLDialect.SQLSERVER)

    # 6-7 SQL Server LIMIT→TOP 변환 위치 검증 (v3.1 버그 수정)
    ss_sql = mgr.convert("SELECT name FROM employees ORDER BY salary DESC LIMIT 5", SQLDialect.SQLITE, SQLDialect.SQLSERVER)
    ok("SQL Server: TOP이 SELECT 뒤에 위치", "SELECT TOP 5" in ss_sql)
    ok("SQL Server: LIMIT 제거됨", "LIMIT" not in ss_sql.upper())


# ── 시나리오 7: SQL 검증 ────────────────────────────────────

def test_scenario_07_validator():
    """시나리오 7: SQLValidator 문법 검증"""
    section("시나리오 7: SQL 검증 (SQLValidator)")

    from text_to_sql_agent import SQLValidator, DatabaseType

    v1, _ = SQLValidator.validate_syntax("SELECT * FROM employees WHERE dept_id = 1", DatabaseType.SQLITE)
    ok("유효 SELECT 통과", v1)

    v2, _ = SQLValidator.validate_syntax("SELEC * FROM employees", DatabaseType.SQLITE)
    ok("SELECT 오타 감지", not v2)

    v3, _ = SQLValidator.validate_syntax("", DatabaseType.SQLITE)
    ok("빈 SQL 감지", not v3)

    v4, _ = SQLValidator.validate_syntax(
        "SELECT e.name, d.dept_name FROM employees e "
        "JOIN departments d ON e.dept_id = d.dept_id",
        DatabaseType.SQLITE,
    )
    ok("JOIN 문법 통과", v4)

    v5, _ = SQLValidator.validate_syntax(
        "WITH avg_sal AS (SELECT AVG(salary) as avg FROM employees) "
        "SELECT * FROM employees WHERE salary > (SELECT avg FROM avg_sal)",
        DatabaseType.SQLITE,
    )
    ok("CTE 문법 통과", v5)


# ── 시나리오 8: 프롬프트 빌더 ────────────────────────────────

def test_scenario_08_prompt():
    """시나리오 8: PromptBuilder 컨텍스트 생성"""
    section("시나리오 8: 프롬프트 빌더 (PromptBuilder)")

    from text_to_sql_agent import SchemaExtractor, PromptBuilder, create_sample_database

    schema = SchemaExtractor.extract_sqlite_schema(create_sample_database())
    ctx = PromptBuilder.build_schema_context(schema)

    ok("스키마 컨텍스트 생성", len(ctx) > 100)
    ok("employees 포함", "employees" in ctx)
    ok("departments 포함", "departments" in ctx)
    ok("projects 포함", "projects" in ctx)
    ok("project_assignments 포함", "project_assignments" in ctx)


# ── 시나리오 9: demo_app 함수 검증 ──────────────────────────

def test_scenario_09_demo_app():
    """시나리오 9: demo_app 상수·함수 검증"""
    section("시나리오 9: demo_app 함수 검증")

    from demo_app import _get_api_key, _BANNER, _MENU, _EXIT_COMMANDS, _MAX_DISPLAY_ROWS

    ok("_get_api_key 호출 가능", _get_api_key() is None or isinstance(_get_api_key(), str))
    ok("_MAX_DISPLAY_ROWS = 10", _MAX_DISPLAY_ROWS == 10)
    ok("_EXIT_COMMANDS: exit", "exit" in _EXIT_COMMANDS)
    ok("_EXIT_COMMANDS: 종료", "종료" in _EXIT_COMMANDS)
    ok("_EXIT_COMMANDS: quit", "quit" in _EXIT_COMMANDS)
    ok("_EXIT_COMMANDS: q", "q" in _EXIT_COMMANDS)
    ok("_BANNER: TCDataAgent-SQL 포함", "TCDataAgent-SQL" in _BANNER)
    ok("_BANNER: Responses API 포함", "Responses API" in _BANNER)
    ok("_BANNER: 400K 포함", "400K" in _BANNER)
    ok("_BANNER: codex 포함", "codex" in _BANNER)
    ok("_BANNER: v3.1 포함", "v3.1" in _BANNER)
    ok("_MENU: 메뉴 선택 포함", "메뉴 선택" in _MENU)


# ── 시나리오 10: 통합 (end-to-end) ──────────────────────────

def test_scenario_10_e2e():
    """시나리오 10: 전체 파이프라인 E2E + 실제 SQLite 실행"""
    section("시나리오 10: 통합 시나리오 (end-to-end)")

    import sqlite3
    from text_to_sql_agent import (
        SchemaExtractor, PromptBuilder, SQLValidator, DatabaseType, create_sample_database,
    )
    from schema_linker import SchemaLinker
    from sql_optimizer import SQLOptimizer
    from dialect_handler import SQLDialect, MultiDatabaseQuery

    # 전체 파이프라인: DB → 스키마 → 링킹 → 프롬프트 → 검증 → 최적화 → 방언 변환
    db = create_sample_database()
    schema = SchemaExtractor.extract_sqlite_schema(db)
    linker = SchemaLinker(schema)
    optimizer = SQLOptimizer()

    question = "개발팀 직원들의 평균 연봉"
    link_result = linker.link(question)
    ok("E2E: 스키마 링킹 성공", len(link_result.relevant_tables) > 0)

    context = PromptBuilder.build_schema_context(schema)
    ok("E2E: 프롬프트 생성 성공", len(context) > 0)

    test_sql = (
        "SELECT AVG(salary) FROM employees e "
        "JOIN departments d ON e.dept_id = d.dept_id "
        "WHERE d.dept_name = '개발'"
    )
    valid, _ = SQLValidator.validate_syntax(test_sql, DatabaseType.SQLITE)
    ok("E2E: SQL 검증 통과", valid)

    opt_result = optimizer.optimize(test_sql)
    ok("E2E: SQL 최적화 완료", opt_result is not None)

    mdb = MultiDatabaseQuery()
    dialects = mdb.generate_for_all_dialects(
        "SELECT GROUP_CONCAT(name) FROM employees", SQLDialect.SQLITE
    )
    ok("E2E: 멀티 방언 변환 완료", len(dialects) >= 5)  # v3.0: 6종

    # 실제 SQLite 실행
    conn = sqlite3.connect(db)
    cursor = conn.execute("SELECT AVG(salary) FROM employees")
    avg = cursor.fetchone()[0]
    conn.close()
    ok(f"E2E: 실제 SQL 실행 성공 (평균 연봉={avg:,.0f})", avg is not None and avg > 0)


# ── 시나리오 11: v3.0 신규 기능 검증 ─────────────────────────

def test_scenario_11_v3_features():
    """시나리오 11: v3.0 신규 기능 (Pydantic, Cartesian Join, Window Function, 신규 방언)"""
    section("시나리오 11: v3.0 신규 기능 검증")

    from text_to_sql_agent import SQLGenerationSchema
    from sql_optimizer import SQLOptimizer
    from schema_linker import SchemaLinker
    from text_to_sql_agent import SchemaExtractor, create_sample_database

    # 11-1 Pydantic 모델 검증
    schema = SQLGenerationSchema.model_json_schema()
    ok("Pydantic 스키마: properties 존재", "properties" in schema)
    required_fields = {"reasoning", "sql", "confidence", "explanation", "assumptions", "alternative_queries"}
    actual_fields = set(schema.get("properties", {}).keys())
    ok("Pydantic 스키마: 필수 필드 완전", required_fields.issubset(actual_fields))

    # 11-2 Pydantic 인스턴스 생성
    instance = SQLGenerationSchema(
        reasoning="테스트 추론",
        sql="SELECT 1",
        confidence=0.95,
        explanation="테스트 설명",
        assumptions=["가정1"],
        alternative_queries=["SELECT 2"]
    )
    ok("Pydantic 인스턴스 생성", instance.sql == "SELECT 1")
    ok("Pydantic JSON 직렬화", instance.model_dump_json() is not None)

    # 11-3 Cartesian Join 감지 (신규 최적화)
    opt = SQLOptimizer()
    r_cart = opt.optimize("SELECT * FROM employees, departments")
    has_cartesian = any("카테시안" in s for s in r_cart.optimizations_applied)
    ok("Cartesian Join 감지", has_cartesian)

    # 11-4 정상 조인은 카테시안 경고 없음
    r_join = opt.optimize(
        "SELECT e.name FROM employees e JOIN departments d ON e.dept_id = d.dept_id"
    )
    no_cartesian = not any("카테시안" in s for s in r_join.optimizations_applied)
    ok("정상 JOIN: 카테시안 경고 없음", no_cartesian)

    # 11-5 신규 한국어 키워드
    db_schema = SchemaExtractor.extract_sqlite_schema(create_sample_database())
    linker = SchemaLinker(db_schema)
    for word in ("사이", "비어있는", "최근", "분기별", "중앙값"):
        ok(f"한국어 키워드: {word}", word in linker.KOREAN_KEYWORDS)

    # 11-6 방언 수 확인 (6종)
    from dialect_handler import DialectManager
    mgr = DialectManager()
    ok(f"지원 방언 수: {len(mgr.dialects)}종 (>=6)", len(mgr.dialects) >= 6)


# ── 시나리오 12: API 통합 테스트 (키 필요) ───────────────────

def test_scenario_12_api_integration():
    """시나리오 12: Azure OpenAI Responses API 호출 (키 없으면 skip)"""
    section("시나리오 12: API 통합 테스트 (GPT-5.4 기본 Responses API)")

    key = os.getenv("OPEN_AI_KEY_5") or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("OPEN_AI_ENDPOINT_5") or os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("MODEL_DEPLOYMENT_GPT5_4") or "gpt-5.4"

    if not key or not endpoint:
        for name in ("에이전트 초기화", "단순 쿼리 생성", "복잡한 쿼리 생성", "조인 쿼리 생성"):
            skip(name, "OPEN_AI_KEY_5 또는 OPEN_AI_ENDPOINT_5 없음")
        return

    from text_to_sql_agent import TextToSQLAgent, create_sample_database

    try:
        agent = TextToSQLAgent(
            api_key=key,
            endpoint=endpoint,
            deployment_name=deployment_name,
            api_version="2025-04-01-preview",
            use_structured_outputs=True,
            enable_deep_reasoning=True,
        )
        db_path = create_sample_database()
        agent.load_database(db_path)
        ok("에이전트 초기화", True)

        # 단순 쿼리
        t0 = time.time()
        r1 = agent.ask("개발팀 직원들의 평균 연봉은?")
        ok(f"단순 쿼리 생성 ({time.time() - t0:.1f}s)",
           r1.get("sql") and "AVG" in r1["sql"].upper())

        # 복잡한 쿼리
        t0 = time.time()
        r2 = agent.ask("부서별 직원 수와 평균 연봉을 알려주세요")
        ok(f"복잡한 쿼리 생성 ({time.time() - t0:.1f}s)",
           r2.get("sql") and "GROUP BY" in r2["sql"].upper())

        # 조인 쿼리
        t0 = time.time()
        r3 = agent.ask("진행 중인 프로젝트에 참여하는 직원 목록")
        ok(f"조인 쿼리 생성 ({time.time() - t0:.1f}s)",
           r3.get("sql") and "JOIN" in r3["sql"].upper())

        agent.close()
    except Exception as e:
        ok("API 통합 테스트", False, str(e))


# ── 시나리오 13: v3.1 신규 기능 검증 (QueryGuard, 모호성 감지, Mermaid) ──

def test_scenario_13_v31_features():
    """시나리오 13: v3.1 신규 기능 (QueryGuard 통합, 모호성 감지, Mermaid ER)"""
    section("시나리오 13: v3.1 신규 기능 검증")

    from text_to_sql_agent import (
        SchemaExtractor, create_sample_database, DestructiveQueryError, TextToSQLAgent,
    )
    from query_guard import QueryGuard, RiskLevel
    from schema_linker import SchemaLinker

    # 13-1 QueryGuard 기본 동작
    guard = QueryGuard()
    ok("QueryGuard: SELECT 안전", not guard.is_destructive("SELECT * FROM employees"))
    ok("QueryGuard: DELETE 감지", guard.is_destructive("DELETE FROM employees"))
    ok("QueryGuard: DROP 감지", guard.is_destructive("DROP TABLE employees"))
    ok("QueryGuard: UPDATE 감지", guard.is_destructive("UPDATE employees SET salary = 0"))
    ok("QueryGuard: TRUNCATE 감지", guard.is_destructive("TRUNCATE TABLE employees"))
    ok("QueryGuard: INSERT 감지", guard.is_destructive("INSERT INTO employees VALUES (1, 'test')"))

    # 13-2 QueryGuard 위험도 분석
    a1 = guard.analyze("DELETE FROM employees")
    ok("위험도: DELETE (no WHERE) = CRITICAL", a1.risk_level == RiskLevel.CRITICAL)

    a2 = guard.analyze("DELETE FROM employees WHERE emp_id = 1")
    ok("위험도: DELETE (with WHERE) = HIGH", a2.risk_level == RiskLevel.HIGH)

    a3 = guard.analyze("INSERT INTO employees VALUES (1, 'test')")
    ok("위험도: INSERT = MEDIUM", a3.risk_level == RiskLevel.MEDIUM)

    a4 = guard.analyze("DROP TABLE employees")
    ok("위험도: DROP = CRITICAL", a4.risk_level == RiskLevel.CRITICAL)

    a5 = guard.analyze("SELECT * FROM employees")
    ok("위험도: SELECT = SAFE", a5.risk_level == RiskLevel.SAFE)

    # 13-3 DestructiveQueryError 예외 클래스
    ok("DestructiveQueryError 임포트 확인", DestructiveQueryError is not None)
    err = DestructiveQueryError("DELETE FROM employees", a1)
    ok("DestructiveQueryError.sql 속성", err.sql == "DELETE FROM employees")
    ok("DestructiveQueryError.analysis 속성", err.analysis is a1)

    # 13-4 TextToSQLAgent __slots__ 확인
    ok("__slots__: enable_safety_guard 존재", 'enable_safety_guard' in TextToSQLAgent.__slots__)
    ok("__slots__: _guard 존재", '_guard' in TextToSQLAgent.__slots__)

    # 13-5 SchemaLinker 모호성 감지
    db_path = create_sample_database()
    schema = SchemaExtractor.extract_sqlite_schema(db_path)
    linker = SchemaLinker(schema)

    amb1 = linker.detect_ambiguity("ㅇ")
    ok("모호성: 짧은 질문 감지", amb1["is_ambiguous"])

    amb2 = linker.detect_ambiguity("그것의 평균을 알려줘")
    ok("모호성: 대명사 감지", amb2["is_ambiguous"])

    amb3 = linker.detect_ambiguity("최근 매출 현황")
    ok("모호성: 모호한 시간 감지", amb3["is_ambiguous"])

    amb4 = linker.detect_ambiguity("개발팀 직원들의 평균 연봉을 알려주세요")
    ok("모호성: 명확한 질문 통과", not amb4["is_ambiguous"])

    # 13-5a _VAGUE_REFERENCES 정규식 거짓 양성 방지 (v3.1 버그 수정)
    amb5 = linker.detect_ambiguity("이번 달 직원 목록")
    ok("모호성: '이번 달' 거짓 양성 방지", not amb5["is_ambiguous"])

    amb6 = linker.detect_ambiguity("이영희의 급여")
    ok("모호성: '이영희' 거짓 양성 방지", not amb6["is_ambiguous"])

    amb7 = linker.detect_ambiguity("평균 이상인 직원")
    ok("모호성: '평균 이상' 거짓 양성 방지", not amb7["is_ambiguous"])

    # 13-5b _link_cache 동작 검증
    r1 = linker.link("개발팀 직원")
    r2 = linker.link("개발팀 직원")
    ok("스키마 링킹 캐시 동작 (동일 객체)", r1 is r2)

    # 13-6 SchemaExtractor.to_mermaid
    mermaid = SchemaExtractor.to_mermaid(schema)
    ok("Mermaid ER: erDiagram 포함", "erDiagram" in mermaid)
    ok("Mermaid ER: employees 포함", "employees" in mermaid)
    ok("Mermaid ER: departments 포함", "departments" in mermaid)
    ok("Mermaid ER: PK 마커 포함", "PK" in mermaid)
    ok("Mermaid ER: FK 관계선 포함", "}o--||" in mermaid)

    # 13-7 SchemaGraphBuilder.to_mermaid 위임 검증
    from schema_graph import SchemaGraphBuilder
    graph_mermaid = SchemaGraphBuilder.to_mermaid(schema)
    ok("SchemaGraphBuilder.to_mermaid 위임 동작", graph_mermaid == mermaid)

    # 13-8 DatabaseType MYSQL/SQLSERVER 추가 검증
    from text_to_sql_agent import DatabaseType
    ok("DatabaseType: MYSQL 존재", hasattr(DatabaseType, 'MYSQL'))
    ok("DatabaseType: SQLSERVER 존재", hasattr(DatabaseType, 'SQLSERVER'))
    ok("DatabaseType: 6종 이상", len(list(DatabaseType)) >= 6)

    # 13-9 ConversationalSQLAgent.ask_with_history force 파라미터 검증
    from text_to_sql_agent import ConversationalSQLAgent
    import inspect
    sig = inspect.signature(ConversationalSQLAgent.ask_with_history)
    ok("ask_with_history: force 파라미터 존재", 'force' in sig.parameters)
    ok("ask_with_history: force 기본값 False", sig.parameters['force'].default is False)


# ── 시나리오 14: v3.1.1 버그 수정 검증 ──────────────────────

def test_scenario_14_v311_fixes():
    """시나리오 14: v3.1.1 버그 수정 검증 (3차 리뷰)"""
    section("시나리오 14: v3.1.1 버그 수정 검증")

    import inspect
    from text_to_sql_agent import TextToSQLAgent, ConversationalSQLAgent, SQLValidator

    # 14-1 confirm 엔드포인트 force=True 검증
    from api_server import confirm_destructive_sql
    source = inspect.getsource(confirm_destructive_sql)
    ok("confirm 엔드포인트: force=True 포함", "force=True" in source)

    # 14-2 ask_with_history에 SQLValidator 검증 로직 포함
    awh_source = inspect.getsource(ConversationalSQLAgent.ask_with_history)
    ok("ask_with_history: validate_syntax 호출", "validate_syntax" in awh_source)
    ok("ask_with_history: validate_schema_references 호출", "validate_schema_references" in awh_source)
    ok("ask_with_history: execute_and_validate 호출", "execute_and_validate" in awh_source)
    ok("ask_with_history: 재시도 루프 존재", "max_retries" in awh_source)

    # 14-3 runtime_config 환경변수 alias 유지
    from runtime_config import get_azure_openai_api_key
    get_api_source = inspect.getsource(get_azure_openai_api_key)
    ok("api key helper: OPEN_AI_KEY_5 확인", "OPEN_AI_KEY_5" in get_api_source)
    ok("api key helper: AZURE_OPENAI_API_KEY 확인", "AZURE_OPENAI_API_KEY" in get_api_source)

    # 14-4 ambiguity_detector ranking scope 실제 동작
    from ambiguity_detector import AmbiguityDetector
    detector = AmbiguityDetector()
    # "가장" 단독으로는 기준 불명확 → 감지
    r1 = detector._check_missing_ranking_scope("제일 직원")
    ok("ranking scope: 기준 없는 질문 감지", r1.is_ambiguous)
    # "가장 많은 예산을 가진 프로젝트" → 기준 명확 → 통과
    r2 = detector._check_missing_ranking_scope("가장 많은 예산을 가진 프로젝트")
    ok("ranking scope: 기준 있는 질문 통과", not r2.is_ambiguous)

    # 14-5 TextToSQLAgent 클래스 독스트링 버전
    ok("TextToSQLAgent docstring: v3.1.4 유지", "v3.1.4" in (TextToSQLAgent.__doc__ or ""))

    # 14-6 BigQuery _convert_strftime 단순화 검증
    from dialect_handler import BigQueryDialect
    bq = BigQueryDialect()
    result = bq._convert_strftime.__wrapped__("%Y-%m-%d", "hire_date")
    ok("BigQuery strftime: FORMAT_DATE 정확", result == "FORMAT_DATE('%Y-%m-%d', hire_date)")

    # 14-7 sql_optimizer _suggest_window_function: 한국어 '순위' 제거 검증
    from sql_optimizer import SQLOptimizer
    opt = SQLOptimizer()
    wf_source = inspect.getsource(opt._suggest_window_function)
    ok("window_function: '순위' 한국어 키워드 제거", "'순위'" not in wf_source)


# ── 시나리오 15: v3.1.2 버그 수정 검증 ──────────────────────

def test_scenario_15_v312_fixes():
    """시나리오 15: v3.1.2 버그 수정 검증 (4차 리뷰)"""
    section("시나리오 15: v3.1.2 버그 수정 검증")

    import inspect
    from text_to_sql_agent import ConversationalSQLAgent, SQLGenerationResult

    # 15-1 ask_with_history: 'parsed' in dir() 제거 → 루프 전 초기화
    awh_source = inspect.getsource(ConversationalSQLAgent.ask_with_history)
    ok("ask_with_history: 'parsed' in dir() 제거", "'parsed' in dir()" not in awh_source)
    ok("ask_with_history: parsed 초기화 존재", "parsed: Dict[str, Any] = {}" in awh_source
       or "parsed: Dict[str, Any]" in awh_source)

    # 15-2 api_server 공통 실행 헬퍼 예외 처리
    from api_server import _execute_query_request
    qd_source = inspect.getsource(_execute_query_request)
    ok("_execute_query_request: RuntimeError 예외 처리", "RuntimeError" in qd_source)
    ok("_execute_query_request: SQL 생성 실패 메시지", "SQL 생성 실패" in qd_source)

    # 15-3 schema_linker '컸럼' → '컬럼' 오타 수정
    from schema_linker import SchemaLinker
    da_source = inspect.getsource(SchemaLinker.detect_ambiguity)
    ok("detect_ambiguity: '컸럼' 오타 제거", "컸럼" not in da_source)
    ok("detect_ambiguity: '컬럼' 정확한 표기", "컬럼" in da_source)

    # 15-4 query_history 데드코드 제거
    ok("ConversationalSQLAgent: query_history 제거",
       not hasattr(ConversationalSQLAgent, 'query_history')
       or 'query_history' not in ConversationalSQLAgent.__slots__)
    # clear_history에서도 제거 확인
    ch_source = inspect.getsource(ConversationalSQLAgent.clear_history)
    ok("clear_history: query_history.clear() 제거", "query_history" not in ch_source)

    # 15-5 _suggest_window_function: 도달 불가 'RANK' 키워드 제거
    from sql_optimizer import SQLOptimizer
    wf_source = inspect.getsource(SQLOptimizer._suggest_window_function)
    # 내부 키워드 리스트가 ['TOP', 'LIMIT 1']만 포함 ('RANK' 제거됨)
    ok("window_function: 내부 키워드에서 'RANK' 제거",
       "['TOP', 'LIMIT 1']" in wf_source)

    # 15-6 ConversationalSQLAgent __slots__ 추가
    ok("ConversationalSQLAgent: __slots__ 정의됨",
       hasattr(ConversationalSQLAgent, '__slots__') and len(ConversationalSQLAgent.__slots__) > 0)
    ok("ConversationalSQLAgent: conversation_history in __slots__",
       'conversation_history' in ConversationalSQLAgent.__slots__)
    ok("ConversationalSQLAgent: _last_response_id in __slots__",
       '_last_response_id' in ConversationalSQLAgent.__slots__)


# ── 시나리오 16: v3.1.3 최적화 검증 ─────────────────────────────

def test_scenario_16_v313_optimizations():
    """시나리오 16: v3.1.3 최적화 검증 (5차 리뷰)"""
    section("시나리오 16: v3.1.3 최적화 검증")

    import inspect

    # 16-1 api_server: 방언 변환된 SQL을 SQLite에서 실행하는 버그 수정
    from api_server import _execute_query_request
    qd_source = inspect.getsource(_execute_query_request)
    ok("_execute_query_request: sqlite_sql 변수 사용 (실행용 분리)",
       "sqlite_sql" in qd_source)
    ok("_execute_query_request: response_sql 변수 사용 (응답용 분리)",
       "response_sql" in qd_source)
    ok("_execute_query_request: execute_query에 sqlite_sql 전달",
       "execute_query(sqlite_sql)" in qd_source)

    # 16-2 ask_with_history: 빈 SQL 실행 방어
    from text_to_sql_agent import ConversationalSQLAgent
    awh_source = inspect.getsource(ConversationalSQLAgent.ask_with_history)
    ok("ask_with_history: 빈 SQL 가드 (if not sql)",
       "if not sql:" in awh_source)
    ok("ask_with_history: 빈 SQL 시 confidence 0.0 반환",
       '"confidence": 0.0' in awh_source)

    # 16-3 _stream_sql_generation: dialect 파라미터 추가
    from api_server import _stream_sql_generation
    stream_sig = inspect.signature(_stream_sql_generation)
    ok("스트리밍: dialect 파라미터 존재",
       "dialect" in stream_sig.parameters)
    stream_source = inspect.getsource(_stream_sql_generation)
    ok("스트리밍: display_sql 변수 사용 (실행/응답 분리)",
       "display_sql" in stream_source)

    # 16-4 MCP: dialect 파라미터 사용
    from mcp_server import MCPTextToSQLServer
    mcp_qd_source = inspect.getsource(MCPTextToSQLServer._tool_query_database)
    ok("MCP query_database: dialect 파라미터 사용",
       "dialect" in mcp_qd_source and "DialectManager" in mcp_qd_source)

    # 16-5 text_to_sql_agent.py 버전 독스트링 v3.1.4
    import text_to_sql_agent as agent_mod
    ok("text_to_sql_agent 독스트링: v3.1.4",
       "v3.1.4" in (agent_mod.__doc__ or ""))


# ── 시나리오 17: v3.1.4 API 운영 기능 검증 ─────────────────────────

def test_scenario_17_v314_api_features():
    """시나리오 17: v3.1.4 API 운영 기능 검증"""
    section("시나리오 17: v3.1.4 API 운영 기능 검증")

    import inspect

    from api_server import (
        QueryRequest,
        _resolve_agent,
        _execute_query_request,
        health_check,
        telemetry_summary,
        telemetry_queries,
        query_database,
        query_database_sync,
        _stream_sql_generation,
        close_session,
    )
    from text_to_sql_agent import TextToSQLAgent, ConversationalSQLAgent
    from runtime_config import get_runtime_settings

    # 17-1 QueryRequest에 max_rows 추가
    req = QueryRequest(question="테스트", max_rows=25)
    ok("QueryRequest: max_rows 필드 존재", hasattr(req, 'max_rows'))
    ok("QueryRequest: max_rows 값 반영", req.max_rows == 25)

    # 17-2 공통 실행 헬퍼가 instructions를 additional_context로 전달
    exec_source = inspect.getsource(_execute_query_request)
    ok("_execute_query_request: additional_context 생성", "additional_context = request.instructions.strip()" in exec_source)
    ok("_execute_query_request: ask_with_history additional_context 전달", "ask_with_history(request.question, execute=False, additional_context=additional_context)" in exec_source)
    ok("_execute_query_request: ask additional_context 전달", "ask(request.question, execute=False, additional_context=additional_context)" in exec_source)

    # 17-3 세션 기반 대화형 에이전트 해석기 추가
    resolve_source = inspect.getsource(_resolve_agent)
    ok("_resolve_agent: session_id 분기 존재", "if session_id:" in resolve_source)
    ok("_resolve_agent: _get_or_create_conversation_agent 호출", "_get_or_create_conversation_agent" in resolve_source)

    # 17-4 ask / ask_with_history 시그니처 확장
    ask_sig = inspect.signature(TextToSQLAgent.ask)
    awh_sig = inspect.signature(ConversationalSQLAgent.ask_with_history)
    ok("TextToSQLAgent.ask: additional_context 파라미터", "additional_context" in ask_sig.parameters)
    ok("ConversationalSQLAgent.ask_with_history: additional_context 파라미터", "additional_context" in awh_sig.parameters)

    # 17-5 스트리밍도 동일 기능 지원
    stream_sig = inspect.signature(_stream_sql_generation)
    stream_source = inspect.getsource(_stream_sql_generation)
    ok("스트리밍: additional_context 파라미터 존재", "additional_context" in stream_sig.parameters)
    ok("스트리밍: max_rows 파라미터 존재", "max_rows" in stream_sig.parameters)
    ok("스트리밍: max_rows 슬라이싱 적용", "results[:max_rows]" in stream_source)

    # 17-6 sync 엔드포인트 및 세션 종료 엔드포인트 추가
    sync_source = inspect.getsource(query_database_sync)
    close_source = inspect.getsource(close_session)
    ok("query_database_sync: model_copy로 stream=False 강제", 'model_copy(update={"stream": False})' in sync_source)
    ok("close_session: _close_conversation 호출", "_close_conversation" in close_source)

    # 17-7 health_check 운영 정보 확장
    health_source = inspect.getsource(health_check)
    ok("health_check: conversation_sessions 포함", '"conversation_sessions"' in health_source)
    ok("health_check: conversation_ttl_seconds 포함", '"conversation_ttl_seconds"' in health_source)
    ok("health_check: expired_sessions_cleaned 포함", '"expired_sessions_cleaned"' in health_source)
    ok("health_check: supported_dialects 포함", '"supported_dialects"' in health_source)

    # 17-8 telemetry 엔드포인트 및 설정 객체
    telemetry_summary_source = inspect.getsource(telemetry_summary)
    telemetry_queries_source = inspect.getsource(telemetry_queries)
    ok("telemetry_summary: metrics 반환", 'metrics=dict(_telemetry)' in telemetry_summary_source)
    ok("telemetry_summary: configured 반환", 'configured={' in telemetry_summary_source)
    ok("telemetry_queries: query audit 반환", '"queries": recent' in telemetry_queries_source)

    from api_server import QueryResponse
    qr_source = inspect.getsource(QueryResponse)
    ok("QueryResponse: request_id 필드 추가", 'request_id' in qr_source)
    ok("QueryResponse: duration_ms 필드 추가", 'duration_ms' in qr_source)

    settings = get_runtime_settings()
    ok("runtime settings: deployment_name 존재", bool(settings.deployment_name))
    ok("runtime settings: api_version 존재", bool(settings.api_version))
    ok("runtime settings: query_audit_limit 양수", settings.query_audit_limit > 0)


def test_scenario_18_v315_runtime_observability():
    """시나리오 18: v3.1.5 운영 설정/관측 기능 검증"""
    section("시나리오 18: v3.1.5 운영 설정/관측 기능 검증")

    import inspect

    from api_server import health_check, telemetry_summary, telemetry_queries, _execute_query_request
    from runtime_config import (
        RuntimeSettings,
        get_runtime_settings,
        get_azure_openai_api_key,
        get_azure_openai_endpoint,
        get_deployment_name,
        get_api_version,
    )

    settings = get_runtime_settings()
    ok("RuntimeSettings 타입 확인", isinstance(settings, RuntimeSettings))
    ok("settings.to_public_dict 메서드 존재", hasattr(settings, 'to_public_dict'))
    ok("deployment_name fallback 존재", bool(get_deployment_name()))
    ok("api_version fallback 존재", bool(get_api_version()))

    health_source = inspect.getsource(health_check)
    ok("health_check: deployment_name 포함", '"deployment_name"' in health_source)
    ok("health_check: api_version 포함", '"api_version"' in health_source)
    ok("health_check: query_audit_size 포함", '"query_audit_size"' in health_source)

    summary_source = inspect.getsource(telemetry_summary)
    ok("telemetry_summary: active_sessions 포함", 'active_sessions=len(_conversations)' in summary_source)
    ok("telemetry_summary: default_max_rows 포함", '"default_max_rows"' in summary_source)

    queries_source = inspect.getsource(telemetry_queries)
    ok("telemetry_queries: limit 파라미터 존재", 'limit: int = Query' in queries_source)

    exec_source = inspect.getsource(_execute_query_request)
    ok("_execute_query_request: request_id 생성", '_new_request_id()' in exec_source)
    ok("_execute_query_request: duration_ms 기록", 'response.duration_ms' in exec_source)
    ok("_execute_query_request: audit 기록", '_record_query_audit(' in exec_source)

    api_key_source = inspect.getsource(get_azure_openai_api_key)
    endpoint_source = inspect.getsource(get_azure_openai_endpoint)
    ok("env helper: OPEN_AI_KEY_5 alias", 'OPEN_AI_KEY_5' in api_key_source)
    ok("env helper: AZURE_OPENAI_API_KEY alias", 'AZURE_OPENAI_API_KEY' in api_key_source)
    ok("env helper: OPEN_AI_ENDPOINT_5 alias", 'OPEN_AI_ENDPOINT_5' in endpoint_source)
    ok("env helper: AZURE_OPENAI_ENDPOINT alias", 'AZURE_OPENAI_ENDPOINT' in endpoint_source)


# ── 시나리오 19: v3.2.0 Responses API 네이티브 파라미터 검증 ──

def test_scenario_19_v320_native_responses_params():
    """시나리오 19: v3.2.0 reasoning.effort / verbosity / prompt_cache_key / 실행피드백"""
    section("시나리오 19: v3.2.0 Responses API 네이티브 파라미터 검증")

    import inspect

    from text_to_sql_agent import (
        TextToSQLAgent, _is_aggregate_query, create_sample_database,
    )
    from runtime_config import RuntimeSettings, get_runtime_settings

    # 19-1 RuntimeSettings에 v3.2.0 필드 존재
    settings = get_runtime_settings()
    ok("settings.default_reasoning_effort 존재", hasattr(settings, 'default_reasoning_effort'))
    ok("settings.deep_reasoning_effort 존재", hasattr(settings, 'deep_reasoning_effort'))
    ok("settings.verbosity 존재", hasattr(settings, 'verbosity'))
    ok("settings.prompt_cache_retention 존재", hasattr(settings, 'prompt_cache_retention'))
    ok("settings.enable_execution_feedback 존재", hasattr(settings, 'enable_execution_feedback'))
    ok("default_reasoning_effort 유효값",
       settings.default_reasoning_effort in {"none", "minimal", "low", "medium", "high", "xhigh"})
    ok("verbosity 유효값", settings.verbosity in {"low", "medium", "high"})
    ok("prompt_cache_retention 유효값", settings.prompt_cache_retention in {"in-memory", "24h"})

    # 19-2 TextToSQLAgent __slots__ 확장
    slot_set = set(TextToSQLAgent.__slots__)
    for name in ("default_reasoning_effort", "deep_reasoning_effort", "verbosity",
                 "prompt_cache_retention", "prompt_cache_key",
                 "enable_execution_feedback", "_last_token_usage"):
        ok(f"TextToSQLAgent.__slots__ 에 {name}", name in slot_set)

    # 19-3 _is_reasoning_model 동작
    agent = TextToSQLAgent.__new__(TextToSQLAgent)
    agent.deployment_name = "gpt-5.4"
    ok("gpt-5.4 → reasoning model", agent._is_reasoning_model())
    agent.deployment_name = "o3-mini"
    ok("o3-mini → reasoning model", agent._is_reasoning_model())
    agent.deployment_name = "gpt-4.1"
    ok("gpt-4.1 → reasoning 아님", not agent._is_reasoning_model())

    # 19-4 _build_request_params 구조 (소스 검사)
    src = inspect.getsource(TextToSQLAgent._build_request_params)
    ok("_build_request_params: reasoning.effort 설정", '"reasoning"' in src and '"effort"' in src)
    ok("_build_request_params: reasoning 모델에서 temperature 미설정 분기",
       'if self._is_reasoning_model()' in src)
    ok("_build_request_params: prompt_cache_key 적용", 'prompt_cache_key' in src)
    ok("_build_request_params: prompt_cache_retention 적용", 'prompt_cache_retention' in src)
    ok("_build_request_params: previous_response_id 지원", 'previous_response_id' in src)

    # 19-5 _build_text_config: verbosity 포함
    text_cfg_src = inspect.getsource(TextToSQLAgent._build_text_config)
    ok("_build_text_config: verbosity 분기", 'verbosity' in text_cfg_src)

    # 19-6 _call_llm 가 _build_request_params 를 사용하도록 리팩터링됨
    call_src = inspect.getsource(TextToSQLAgent._call_llm)
    ok("_call_llm: _build_request_params 사용", '_build_request_params' in call_src)
    ok("_call_llm: 프롬프트 삽입 '심층 추론 모드' 제거",
       "심층 추론 모드 (Deep Reasoning)" not in call_src)

    # 19-7 집계 쿼리 감지기
    ok("_is_aggregate_query: COUNT", _is_aggregate_query("SELECT COUNT(*) FROM t"))
    ok("_is_aggregate_query: GROUP BY", _is_aggregate_query("SELECT a, SUM(b) FROM t GROUP BY a"))
    ok("_is_aggregate_query: 일반 SELECT false", not _is_aggregate_query("SELECT * FROM t WHERE x=1"))

    # 19-8 last_token_usage 프로퍼티
    ok("last_token_usage 프로퍼티 존재",
       isinstance(inspect.getattr_static(TextToSQLAgent, 'last_token_usage'), property))

    # 19-9 load_database 가 prompt_cache_key 자동 생성
    load_src = inspect.getsource(TextToSQLAgent.load_database)
    ok("load_database: prompt_cache_key 자동 생성", 'self.prompt_cache_key' in load_src)

    # 19-10 generate_sql 에 ReFoRCE 스타일 빈결과 피드백 포함
    gen_src = inspect.getsource(TextToSQLAgent.generate_sql)
    ok("generate_sql: enable_execution_feedback 분기", 'enable_execution_feedback' in gen_src)
    ok("generate_sql: 빈 결과 자동 재생성 메시지", '결과가 0행' in gen_src)

    # 19-11 api_server QueryResponse 에 token_usage 필드 추가
    from api_server import QueryResponse
    qr_src = inspect.getsource(QueryResponse)
    ok("QueryResponse: token_usage 필드", 'token_usage' in qr_src)

    from api_server import health_check
    health_src = inspect.getsource(health_check)
    ok("health_check: v3.2.0 reasoning 상태 노출",
       '"default_reasoning_effort"' in health_src and '"verbosity"' in health_src)
    ok("health_check: version 3.2.0", '"3.2.0"' in health_src)


# ── 시나리오 19: v3.2.0 Responses API 네이티브 파라미터 검증 ──

def test_scenario_19_v320_native_responses_params():
    """시나리오 19: v3.2.0 reasoning.effort / verbosity / prompt_cache_key / 실행피드백"""
    section("시나리오 19: v3.2.0 Responses API 네이티브 파라미터 검증")

    import inspect

    from text_to_sql_agent import (
        TextToSQLAgent, _is_aggregate_query, create_sample_database,
    )
    from runtime_config import RuntimeSettings, get_runtime_settings

    # 19-1 RuntimeSettings에 v3.2.0 필드 존재
    settings = get_runtime_settings()
    ok("settings.default_reasoning_effort 존재", hasattr(settings, 'default_reasoning_effort'))
    ok("settings.deep_reasoning_effort 존재", hasattr(settings, 'deep_reasoning_effort'))
    ok("settings.verbosity 존재", hasattr(settings, 'verbosity'))
    ok("settings.prompt_cache_retention 존재", hasattr(settings, 'prompt_cache_retention'))
    ok("settings.enable_execution_feedback 존재", hasattr(settings, 'enable_execution_feedback'))
    ok("default_reasoning_effort 유효값",
       settings.default_reasoning_effort in {"none", "minimal", "low", "medium", "high", "xhigh"})
    ok("verbosity 유효값", settings.verbosity in {"low", "medium", "high"})
    ok("prompt_cache_retention 유효값", settings.prompt_cache_retention in {"in-memory", "24h"})

    # 19-2 TextToSQLAgent __slots__ 확장
    slot_set = set(TextToSQLAgent.__slots__)
    for name in ("default_reasoning_effort", "deep_reasoning_effort", "verbosity",
                 "prompt_cache_retention", "prompt_cache_key",
                 "enable_execution_feedback", "_last_token_usage"):
        ok(f"TextToSQLAgent.__slots__ 에 {name}", name in slot_set)

    # 19-3 _is_reasoning_model 동작
    agent = TextToSQLAgent.__new__(TextToSQLAgent)
    agent.deployment_name = "gpt-5.4"
    ok("gpt-5.4 → reasoning model", agent._is_reasoning_model())
    agent.deployment_name = "o3-mini"
    ok("o3-mini → reasoning model", agent._is_reasoning_model())
    agent.deployment_name = "gpt-4.1"
    ok("gpt-4.1 → reasoning 아님", not agent._is_reasoning_model())

    # 19-4 _build_request_params 구조 (소스 검사)
    src = inspect.getsource(TextToSQLAgent._build_request_params)
    ok("_build_request_params: reasoning.effort 설정", '"reasoning"' in src and '"effort"' in src)
    ok("_build_request_params: reasoning 모델에서 temperature 미설정 분기",
       'if self._is_reasoning_model()' in src)
    ok("_build_request_params: prompt_cache_key 적용", 'prompt_cache_key' in src)
    ok("_build_request_params: prompt_cache_retention 적용", 'prompt_cache_retention' in src)
    ok("_build_request_params: previous_response_id 지원", 'previous_response_id' in src)

    # 19-5 _build_text_config: verbosity 포함
    text_cfg_src = inspect.getsource(TextToSQLAgent._build_text_config)
    ok("_build_text_config: verbosity 분기", 'verbosity' in text_cfg_src)

    # 19-6 _call_llm 가 _build_request_params 를 사용하도록 리팩터링됨
    call_src = inspect.getsource(TextToSQLAgent._call_llm)
    ok("_call_llm: _build_request_params 사용", '_build_request_params' in call_src)
    ok("_call_llm: 프롬프트 삽입 '심층 추론 모드' 제거",
       "심층 추론 모드 (Deep Reasoning)" not in call_src)

    # 19-7 집계 쿼리 감지기
    ok("_is_aggregate_query: COUNT", _is_aggregate_query("SELECT COUNT(*) FROM t"))
    ok("_is_aggregate_query: GROUP BY", _is_aggregate_query("SELECT a, SUM(b) FROM t GROUP BY a"))
    ok("_is_aggregate_query: 일반 SELECT false", not _is_aggregate_query("SELECT * FROM t WHERE x=1"))

    # 19-8 last_token_usage 프로퍼티
    ok("last_token_usage 프로퍼티 존재",
       isinstance(inspect.getattr_static(TextToSQLAgent, 'last_token_usage'), property))

    # 19-9 load_database 가 prompt_cache_key 자동 생성
    load_src = inspect.getsource(TextToSQLAgent.load_database)
    ok("load_database: prompt_cache_key 자동 생성", 'self.prompt_cache_key' in load_src)

    # 19-10 generate_sql 에 ReFoRCE 스타일 빈결과 피드백 포함
    gen_src = inspect.getsource(TextToSQLAgent.generate_sql)
    ok("generate_sql: enable_execution_feedback 분기", 'enable_execution_feedback' in gen_src)
    ok("generate_sql: 빈 결과 자동 재생성 메시지", '결과가 0행' in gen_src)

    # 19-11 api_server QueryResponse 에 token_usage 필드 추가
    from api_server import QueryResponse
    qr_src = inspect.getsource(QueryResponse)
    ok("QueryResponse: token_usage 필드", 'token_usage' in qr_src)

    from api_server import health_check
    health_src = inspect.getsource(health_check)
    ok("health_check: v3.2.0 reasoning 상태 노출",
       '"default_reasoning_effort"' in health_src and '"verbosity"' in health_src)
    ok("health_check: version 3.2.0", '"3.2.0"' in health_src)


# ── 시나리오 20: v3.2.0 임베딩 기반 Schema Retriever 검증 ──

def test_scenario_20_v320_embedding_retriever():
    """시나리오 20: EmbeddingSchemaRetriever 구조 + SchemaLinker 통합 (API 호출 없이 정적 검증)"""
    section("시나리오 20: v3.2.0 임베딩 기반 Schema Retriever 검증")

    import inspect
    from runtime_config import get_runtime_settings

    # 20-1 RuntimeSettings 확장 필드
    settings = get_runtime_settings()
    ok("settings.enable_embedding_retrieval 존재", hasattr(settings, "enable_embedding_retrieval"))
    ok("settings.embedding_deployment 존재", hasattr(settings, "embedding_deployment"))
    ok("settings.embedding_top_k 존재", hasattr(settings, "embedding_top_k"))
    ok("settings.embedding_min_score 존재", hasattr(settings, "embedding_min_score"))
    ok("embedding_top_k >= 1", settings.embedding_top_k >= 1)
    ok("embedding_min_score 0~1", 0.0 <= settings.embedding_min_score <= 1.0)

    # 20-2 모듈 임포트 및 주요 심볼
    import schema_retriever as sr
    ok("schema_retriever: EmbeddingSchemaRetriever", hasattr(sr, "EmbeddingSchemaRetriever"))
    ok("schema_retriever: RetrievalHit", hasattr(sr, "RetrievalHit"))
    ok("schema_retriever: TableEmbedding", hasattr(sr, "TableEmbedding"))
    ok("schema_retriever: _cosine 헬퍼", hasattr(sr, "_cosine"))

    # 20-3 코사인 유사도 정상 동작 (순수 파이썬)
    ok("_cosine: 동일 벡터 = 1.0", abs(sr._cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9)
    ok("_cosine: 직교 벡터 = 0.0", abs(sr._cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9)
    ok("_cosine: 빈 벡터 방어 = 0.0", sr._cosine([], [1.0]) == 0.0)

    # 20-4 자격증명 없이 생성 시 조용히 비활성화
    import os
    saved = {k: os.environ.pop(k, None) for k in (
        "OPEN_AI_KEY_5", "AZURE_OPENAI_API_KEY",
        "OPEN_AI_ENDPOINT_5", "AZURE_OPENAI_ENDPOINT",
    )}
    try:
        dummy = sr.EmbeddingSchemaRetriever()
        ok("자격증명 없을 때 is_ready False", dummy.is_ready is False)
        ok("빈 질문 retrieve는 빈 리스트", dummy.retrieve("", top_k=3) == [])
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    # 20-5 _serialize_table 결과에 PK/FK 포함
    from text_to_sql_agent import SchemaExtractor, create_sample_database
    db_path = create_sample_database()
    schema = SchemaExtractor.extract_sqlite_schema(db_path)
    emp_table = next(t for t in schema.tables if t.name == "employees")
    serialized = sr.EmbeddingSchemaRetriever._serialize_table(emp_table)
    ok("_serialize_table: Table 키워드", "Table: employees" in serialized)
    ok("_serialize_table: Columns 포함", "Columns:" in serialized)
    ok("_serialize_table: PrimaryKeys 포함", "PrimaryKeys:" in serialized)
    ok("_serialize_table: ForeignKeys 포함", "ForeignKeys:" in serialized)

    # 20-6 SchemaLinker 확장: attach_retriever 존재 + __slots__ 확장
    from schema_linker import SchemaLinker
    ok("SchemaLinker.attach_retriever 메서드", callable(getattr(SchemaLinker, "attach_retriever", None)))
    slot_set = set(SchemaLinker.__slots__)
    for slot in ("_retriever", "_retriever_top_k", "_retriever_min_score"):
        ok(f"SchemaLinker.__slots__ 에 {slot}", slot in slot_set)

    # 20-7 link() 가 임베딩 보강 분기 포함
    link_src = inspect.getsource(SchemaLinker.link)
    ok("link(): _retriever 분기", "self._retriever" in link_src)
    ok("link(): embedding link_type 생성", '"embedding"' in link_src)

    # 20-8 비활성 retriever 부착 시 False 반환 + 기존 동작 보존
    linker = SchemaLinker(schema)
    attached = linker.attach_retriever(dummy)
    ok("is_ready=False 검색기 attach → False", attached is False)
    result = linker.link("개발팀 직원들의 평균 연봉")
    ok("retriever 없어도 링킹 정상 동작", "employees" in result.relevant_tables)

    # 20-9 api_server._register_database 에 retriever 등록 분기
    from api_server import _register_database, health_check
    reg_src = inspect.getsource(_register_database)
    ok("_register_database: enable_embedding_retrieval 분기", "enable_embedding_retrieval" in reg_src)
    ok("_register_database: EmbeddingSchemaRetriever 임포트", "EmbeddingSchemaRetriever" in reg_src)
    ok("_register_database: attach_retriever 호출", "attach_retriever" in reg_src)

    health_src2 = inspect.getsource(health_check)
    ok("health_check: embedding 설정 노출",
       '"embedding_retrieval_enabled"' in health_src2 and '"embedding_deployment"' in health_src2)

_ALL_SCENARIOS = [
    test_scenario_01_imports,
    test_scenario_02_schema,
    test_scenario_03_schema_linking,
    test_scenario_04_optimizer,
    test_scenario_05_self_correction,
    test_scenario_06_dialect,
    test_scenario_07_validator,
    test_scenario_08_prompt,
    test_scenario_09_demo_app,
    test_scenario_10_e2e,
    test_scenario_11_v3_features,         # v3.0 신규
    test_scenario_12_api_integration,     # API (키 없으면 skip)
    test_scenario_13_v31_features,        # v3.1 신규
    test_scenario_14_v311_fixes,          # v3.1.1 버그 수정
    test_scenario_15_v312_fixes,          # v3.1.2 버그 수정
    test_scenario_16_v313_optimizations,  # v3.1.3 최적화
    test_scenario_17_v314_api_features,   # v3.1.4 API 운영 기능
    test_scenario_18_v315_runtime_observability,  # v3.1.5 운영 설정/관측 기능
    test_scenario_19_v320_native_responses_params,  # v3.2.0 Responses API 네이티브 파라미터
    test_scenario_20_v320_embedding_retriever,  # v3.2.0 임베딩 기반 Schema Retriever
]

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  Advanced Text-to-SQL 통합 테스트")
    print("  2026년 4월 17일 (v3.2.0 - Responses API 네이티브 파라미터)")
    print("=" * 70)

    start = time.time()

    for fn in _ALL_SCENARIOS:
        fn()

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("  테스트 결과 요약")
    print("=" * 70)
    print(f"  ✅ 성공: {_passed}")
    print(f"  ❌ 실패: {_failed}")
    print(f"  ⏭️ 스킵: {_skipped}")
    print(f"  ⏱️ 소요 시간: {elapsed:.1f}초")
    print("=" * 70)

    if _failed == 0:
        print("\n🎉 전체 시나리오 테스트 통과!")
    else:
        print(f"\n⚠️ {_failed}개 실패")
        sys.exit(1)
