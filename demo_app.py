"""
Advanced Text-to-SQL Demo Application (v2.2.1)

Spider 2.0 벤치마크 #1 TCDataAgent-SQL (93.97%) 참조 기술 기반의 Text-to-SQL 데모.
GPT-5.2 / gpt-5.2-codex, API v1, 400K context window 지원. (2026-02 최신)

실행 방법:
    python demo_app.py

환경 변수:
    - AZURE_OPENAI_API_KEY / OPEN_AI_KEY_5: Azure OpenAI API 키
    - AZURE_OPENAI_ENDPOINT / OPEN_AI_ENDPOINT_5: Azure OpenAI 엔드포인트

최적화 (v2.2.1):
    - 미사용 import 9개 제거, dispatch dict, _get_api_key/_print_query_result DRY
    - run_sample_questions 리소스 누수 수정 (agent 루프 밖 1회 생성 + finally close)
"""

import os
from typing import Any

# 로컬 모듈 임포트
from text_to_sql_agent import (
    TextToSQLAgent,
    ConversationalSQLAgent,
    create_sample_database,
    SchemaExtractor,
)
from schema_linker import SchemaLinker
from sql_optimizer import SQLOptimizer
from dialect_handler import SQLDialect, MultiDatabaseQuery


# 상수
_BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🏆 Advanced Text-to-SQL Agent (2026-02)                        ║
║   ─────────────────────────────────────────────────────────────  ║
║   Spider 2.0 #1 TCDataAgent-SQL (93.97%) 참조 기술 기반         ║
║   GPT-5.2 / gpt-5.2-codex · API v1 · 400K Context              ║
║                                                                  ║
║   Features:                                                      ║
║   • Multi-step Reasoning + Contextual Scaling                    ║
║   • Schema Linking + Relational Knowledge Graph                  ║
║   • Self-Correction (5-round)                                    ║
║   • Multi-Database Support (SQLite, BigQuery, Snowflake, etc.)  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

_MENU = """
==================================================
메뉴 선택:
  1. 자연어 질문 입력 (SQL 생성 및 실행)
  2. 스키마 정보 보기
  3. SQL 최적화 분석
  4. 멀티 방언 SQL 변환
  5. 대화형 모드 (히스토리 유지)
  6. 샘플 질문 실행
  0. 종료
=================================================="""

_EXIT_COMMANDS = frozenset(['exit', '종료', 'quit', 'q'])
_MAX_DISPLAY_ROWS = 10


# ── 공통 유틸 ──────────────────────────────────────────────

def _get_api_key() -> str | None:
    """Azure OpenAI API 키 조회 (중복 호출 제거용)"""
    return os.getenv("AZURE_OPENAI_API_KEY")


def _print_query_result(result: dict[str, Any], *, max_rows: int = _MAX_DISPLAY_ROWS) -> None:
    """SQL 생성 결과를 출력하는 공통 헬퍼 (3곳에서 재사용)"""
    print(f"\n🔍 생성된 SQL:\n   {result['sql']}")
    print(f"\n💬 설명: {result['explanation']}")
    if 'confidence' in result:
        print(f"🎯 신뢰도: {result['confidence']:.1%}")
    if 'results' in result and result['results']:
        row_count = result['row_count']
        print(f"\n📊 결과 ({row_count}행):")
        for row in result['results'][:max_rows]:
            print(f"   {row}")
        if row_count > max_rows:
            print(f"   ... 외 {row_count - max_rows}행")


def demo_schema_info(db_path: str) -> None:
    """스키마 정보 출력"""
    schema = SchemaExtractor.extract_sqlite_schema(db_path)
    
    print("\n📊 데이터베이스 스키마 정보")
    print("=" * 50)
    print(f"데이터베이스: {schema.database_name}")
    print(f"테이블 수: {len(schema.tables)}")
    
    for table in schema.tables:
        print(f"\n📋 테이블: {table.name}")
        print("  컬럼:")
        for col in table.columns:
            pk = " [PK]" if col["name"] in table.primary_keys else ""
            nullable = "" if col.get("nullable", True) else " NOT NULL"
            print(f"    - {col['name']}: {col['type']}{nullable}{pk}")
        
        if table.foreign_keys:
            print("  외래키:")
            for fk in table.foreign_keys:
                print(f"    - {fk['column']} → {fk['references_table']}.{fk['references_column']}")
        
        if table.sample_data:
            print(f"  샘플 데이터 ({len(table.sample_data)}행):")
            for i, row in enumerate(table.sample_data[:2], 1):
                print(f"    {i}. {row}")


def demo_sql_optimization() -> None:
    """SQL 최적화 데모"""
    print("\n🔧 SQL 최적화 분석")
    print("=" * 50)
    
    optimizer = SQLOptimizer()
    
    test_queries = (
        ("SELECT * FROM employees WHERE salary > 50000",
         "SELECT * 사용"),
        
        ("SELECT name FROM employees WHERE dept_id IN (SELECT dept_id FROM departments WHERE location = '서울')",
         "IN 서브쿼리"),
        
        ("SELECT e.name, d.dept_name FROM employees e "
         "JOIN departments d ON e.dept_id = d.dept_id "
         "ORDER BY e.name",
         "ORDER BY without LIMIT"),
    )
    
    for sql, description in test_queries:
        print(f"\n📝 {description}")
        print(f"   SQL: {sql[:80]}...")
        result = optimizer.optimize(sql)
        if result.optimizations_applied:
            print("   💡 최적화 제안:")
            for opt in result.optimizations_applied:
                print(f"      - {opt}")
        else:
            print("   ✅ 최적화 제안 없음")


def demo_dialect_conversion() -> None:
    """SQL 방언 변환 데모"""
    print("\n🌐 멀티 데이터베이스 SQL 변환")
    print("=" * 50)
    
    multi_db = MultiDatabaseQuery()
    
    base_sql = """SELECT 
    d.dept_name,
    GROUP_CONCAT(e.name) as employee_names,
    COUNT(*) as emp_count,
    AVG(e.salary) as avg_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY d.dept_name
HAVING COUNT(*) > 1
ORDER BY avg_salary DESC"""
    
    print(f"\n📝 원본 SQL (SQLite):\n{base_sql}")
    
    results = multi_db.generate_for_all_dialects(base_sql, SQLDialect.SQLITE)
    
    for dialect in (SQLDialect.BIGQUERY, SQLDialect.SNOWFLAKE, SQLDialect.POSTGRESQL):
        print(f"\n🔄 {dialect.value.upper()}:")
        print(results[dialect])


def run_sample_questions(db_path: str, use_api: bool = False) -> None:
    """샘플 질문 실행"""
    print("\n📚 샘플 질문 실행")
    print("=" * 50)

    sample_questions = [
        "개발팀 직원들의 평균 연봉은 얼마인가요?",
        "부서별 직원 수를 보여주세요",
        "진행 중인 프로젝트에 참여하는 직원 목록",
        "2020년 이후 입사한 직원 중 연봉이 7000만원 이상인 사람",
        "가장 많은 예산을 가진 부서는?",
    ]

    schema = SchemaExtractor.extract_sqlite_schema(db_path)
    linker = SchemaLinker(schema)

    # API 모드: 에이전트를 루프 바깥에서 1회만 생성 (리소스 누수 수정)
    agent: TextToSQLAgent | None = None
    if use_api:
        try:
            agent = TextToSQLAgent()
            agent.load_database(db_path)
        except Exception as e:
            print(f"  ⚠️ 에이전트 초기화 오류: {e}")
            agent = None

    try:
        for i, question in enumerate(sample_questions, 1):
            print(f"\n질문 {i}: {question}")
            linking_result = linker.link(question)
            print(f"  📎 관련 테이블: {', '.join(linking_result.relevant_tables)}")

            if agent:
                try:
                    _print_query_result(agent.ask(question), max_rows=3)
                except Exception as e:
                    print(f"  ⚠️ API 호출 오류: {e}")
            else:
                print("  💡 집중 스키마:")
                for line in linker.get_focused_schema(question).split('\n')[:10]:
                    print(f"     {line}")
    finally:
        if agent:
            agent.close()


def _read_question() -> str | None:
    """사용자 질문 입력. exit 시 None 반환."""
    question = input("\n🗣️ 질문: ").strip()
    if question.lower() in _EXIT_COMMANDS:
        return None
    return question or ""


def interactive_mode(db_path: str) -> None:
    """대화형 모드"""
    print("\n💬 대화형 모드")
    print("=" * 50)
    print("질문을 입력하세요. 'exit' 또는 '종료'로 나갑니다.")
    print("대화 히스토리가 유지됩니다.\n")

    try:
        if not _get_api_key():
            print("⚠️ AZURE_OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            print("   스키마 링킹 데모만 실행합니다.\n")
            schema = SchemaExtractor.extract_sqlite_schema(db_path)
            linker = SchemaLinker(schema)

            while (question := _read_question()) is not None:
                if not question:
                    continue
                result = linker.link(question)
                print("\n📎 스키마 분석 결과:")
                print(f"   관련 테이블: {', '.join(result.relevant_tables)}")
                print(f"   관련 컬럼: {dict(result.relevant_columns)}")
                if result.inferred_joins:
                    print(f"   추론된 조인: {result.inferred_joins}")
                print("\n💡 집중 스키마:")
                print(linker.get_focused_schema(question))
            return

        # API 키가 있으면 실제 에이전트 사용
        agent = ConversationalSQLAgent()
        agent.load_database(db_path)
        try:
            while (question := _read_question()) is not None:
                if not question:
                    continue
                try:
                    _print_query_result(agent.ask_with_history(question))
                except Exception as e:
                    print(f"\n❌ 오류: {e}")
        finally:
            agent.close()

    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")


def single_question_mode(db_path: str) -> None:
    """단일 질문 모드"""
    print("\n🎯 자연어 질문 입력")
    print("=" * 50)

    question = input("질문을 입력하세요: ").strip()
    if not question:
        print("질문이 입력되지 않았습니다.")
        return

    schema = SchemaExtractor.extract_sqlite_schema(db_path)
    linker = SchemaLinker(schema)

    print("\n📎 스키마 분석...")
    linking = linker.link(question)
    print(f"   관련 테이블: {', '.join(linking.relevant_tables)}")

    if not _get_api_key():
        print("\n⚠️ AZURE_OPENAI_API_KEY가 설정되지 않아 SQL 생성을 건너뜁니다.")
        print("💡 집중 스키마:")
        print(linker.get_focused_schema(question))
        return

    try:
        print("\n🤖 SQL 생성 중...")
        agent = TextToSQLAgent()
        agent.load_database(db_path)
        try:
            _print_query_result(agent.ask(question))
        finally:
            agent.close()
    except Exception as e:
        print(f"\n❌ 오류: {e}")


def main() -> None:
    """메인 함수"""
    print(_BANNER)

    print("📦 샘플 데이터베이스 준비 중...")
    db_path = create_sample_database()
    print(f"   ✅ 데이터베이스 생성 완료: {db_path}")

    # dispatch dict — if/elif 7단 분기 → O(1) 룩업
    dispatch: dict[str, callable] = {
        "1": lambda: single_question_mode(db_path),
        "2": lambda: demo_schema_info(db_path),
        "3": demo_sql_optimization,
        "4": demo_dialect_conversion,
        "5": lambda: interactive_mode(db_path),
        "6": lambda: run_sample_questions(db_path, use_api=_get_api_key() is not None),
    }

    while True:
        print(_MENU)
        choice = input("\n선택: ").strip()
        if choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        handler = dispatch.get(choice)
        if handler:
            handler()
        else:
            print("\n⚠️ 잘못된 선택입니다. 다시 선택해주세요.")


if __name__ == "__main__":
    main()
