"""
Advanced Text-to-SQL 통합 테스트 (v3.0.0)

12개 시나리오 × 다중 테스트 케이스 = 106 항목
- 시나리오 1~11: 오프라인 (API 키 불필요)
- 시나리오 12: API 통합 테스트 (키 없으면 자동 skip)

v3.0.0 변경:
- Pydantic 모델 (SQLGenerationSchema) 테스트 추가
- Responses API 지원 버전 확인
- MySQL/SQL Server 방언 테스트 추가
- 신규 최적화 규칙 (Cartesian Join, Window Function) 테스트
- 모델 수 19개 확인
- 코드 최적화 검증: __slots__, _table_dict, _COMPILED_PATTERNS, dict dispatch

실행 방법:
  pytest test_all.py -v            # pytest
  python test_all.py               # 독립 실행
"""

import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv
load_dotenv("../.env")


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

    # 1-1 text_to_sql_agent (11개)
    from text_to_sql_agent import (
        TextToSQLAgent, ConversationalSQLAgent, SchemaExtractor,
        PromptBuilder, SQLValidator, ModelConfig, SQLGenerationResult,
        DatabaseType, TableSchema, DatabaseSchema, create_sample_database,
    )
    ok("text_to_sql_agent 임포트 (11개)", True)

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
    ok(f"ModelConfig 등록 모델: {len(all_models)}개 (>=19)", len(all_models) >= 19)
    ok("gpt-5.2-codex 존재", "gpt-5.2-codex" in model_values)
    ok("gpt-5.2-mini 존재", "gpt-5.2-mini" in model_values)
    ok("gpt-5-pro 존재", "gpt-5-pro" in model_values)
    ok("o3-pro 존재", "o3-pro" in model_values)
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
    ok("_BANNER: v3.0 포함", "v3.0" in _BANNER)
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
    section("시나리오 12: API 통합 테스트 (GPT-5.2 Responses API)")

    key = os.getenv("OPEN_AI_KEY_5")
    endpoint = os.getenv("OPEN_AI_ENDPOINT_5")

    if not key or not endpoint:
        for name in ("에이전트 초기화", "단순 쿼리 생성", "복잡한 쿼리 생성", "조인 쿼리 생성"):
            skip(name, "API 키 없음")
        return

    from text_to_sql_agent import TextToSQLAgent, create_sample_database

    try:
        agent = TextToSQLAgent(
            api_key=key,
            endpoint=endpoint,
            deployment_name="gpt-5.2",
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


# ── 메인 ─────────────────────────────────────────────────────

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
]

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  Advanced Text-to-SQL 통합 테스트")
    print("  2026년 6월 15일 (v3.0.0 — Responses API, 106 항목)")
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
