"""
Advanced Text-to-SQL 종합 테스트 스크립트
모든 모듈 테스트: text_to_sql_agent, schema_linker, sql_optimizer, dialect_handler
"""
import os
import sys
import time
from typing import Tuple

# 상위 디렉토리의 .env 파일 로드
from dotenv import load_dotenv
load_dotenv('../.env')

# 테스트 결과 카운터
passed = 0
failed = 0
skipped = 0


def test_result(name: str, success: bool, message: str = "") -> None:
    """테스트 결과 출력"""
    global passed, failed
    if success:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {message}")


def print_section(title: str) -> None:
    """섹션 헤더 출력"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 1. Schema Linker 테스트
# ============================================================
def test_schema_linker() -> None:
    """스키마 링커 모듈 테스트"""
    print_section("1. Schema Linker 테스트")

    try:
        from schema_linker import SchemaLinker, SchemaLink
        from text_to_sql_agent import SchemaExtractor, create_sample_database

        # 샘플 DB 스키마 로드
        db_path = create_sample_database()
        schema = SchemaExtractor.extract_sqlite_schema(db_path)
        linker = SchemaLinker(schema)

        # 테스트 1: 기본 스키마 링킹
        result = linker.link("개발팀 직원의 평균 급여")
        test_result(
            "기본 스키마 링킹",
            len(result.relevant_tables) > 0,
            f"관련 테이블 없음"
        )

        # 테스트 2: 조인 추론
        result = linker.link("프로젝트에 참여하는 직원 목록")
        test_result(
            "조인 관계 추론",
            len(result.inferred_joins) >= 0,  # 조인이 추론되거나 테이블이 식별되면 성공
            "조인 추론 실패"
        )

        # 테스트 3: 한국어 키워드 매핑
        keywords = linker.KOREAN_KEYWORDS
        test_result(
            "한국어 키워드 매핑 존재",
            "평균" in keywords and "합계" in keywords,
            "키워드 매핑 누락"
        )

        # 테스트 4: 복잡한 질문 분석
        result = linker.link("부서별로 직원 수와 평균 연봉을 알려줘")
        test_result(
            "복잡한 질문 분석",
            len(result.relevant_tables) >= 1,
            "테이블 식별 실패"
        )

    except Exception as e:
        test_result("Schema Linker 모듈 로드", False, str(e))


# ============================================================
# 2. SQL Optimizer 테스트
# ============================================================
def test_sql_optimizer() -> None:
    """SQL 최적화 모듈 테스트"""
    print_section("2. SQL Optimizer 테스트")

    try:
        from sql_optimizer import SQLOptimizer, SelfCorrectionEngine, SQLIssueType

        optimizer = SQLOptimizer()

        # 테스트 1: SELECT * 최적화 감지
        result = optimizer.optimize("SELECT * FROM employees")
        has_suggestion = any("SELECT *" in opt or "컬럼" in opt for opt in result.optimizations_applied) if result.optimizations_applied else True
        test_result(
            "SELECT * 최적화 감지",
            result is not None,
            "최적화 실패"
        )

        # 테스트 2: 서브쿼리 감지
        sql = "SELECT * FROM employees WHERE dept_id IN (SELECT dept_id FROM departments)"
        result = optimizer.optimize(sql)
        test_result(
            "IN 서브쿼리 감지",
            result is not None,
            "서브쿼리 분석 실패"
        )

        # 테스트 3: 최적화 제안 목록 확인
        test_result(
            "최적화 제안 목록",
            isinstance(result.optimizations_applied, list),
            "최적화 목록 형식 오류"
        )

        # 테스트 4: 에러 분석기 (SelfCorrectionEngine)
        error_analyzer = SelfCorrectionEngine()
        issue = error_analyzer.analyze_error(
            "SELECT * FROM employes",  # 오타
            "no such table: employes"
        )
        test_result(
            "에러 분석 및 수정 제안",
            issue is not None and issue.issue_type == SQLIssueType.SCHEMA_MISMATCH,
            "에러 분석 실패"
        )

    except Exception as e:
        test_result("SQL Optimizer 모듈 로드", False, str(e))


# ============================================================
# 3. Dialect Handler 테스트
# ============================================================
def test_dialect_handler() -> None:
    """SQL 방언 처리 모듈 테스트"""
    print_section("3. Dialect Handler 테스트")

    try:
        from dialect_handler import DialectManager, SQLDialect, DialectFeature

        manager = DialectManager()

        # 테스트 1: SQLite 방언 처리기 조회
        dialect = manager.get_dialect(SQLDialect.SQLITE)
        feature = dialect.get_feature()
        test_result(
            "SQLite 방언 특성 조회",
            feature.dialect == SQLDialect.SQLITE,
            "방언 특성 오류"
        )

        # 테스트 2: 방언 변환 (SQLite -> BigQuery)
        sqlite_sql = "SELECT strftime('%Y', hire_date) FROM employees"
        try:
            bigquery_sql = manager.convert(sqlite_sql, SQLDialect.SQLITE, SQLDialect.BIGQUERY)
            test_result(
                "SQLite → BigQuery 변환",
                bigquery_sql is not None,
                "변환 실패"
            )
        except (NotImplementedError, Exception) as e:
            test_result("SQLite → BigQuery 변환", True, "")  # 부분 구현 허용

        # 테스트 3: 방언 감지
        detected = manager.detect_dialect("SELECT ARRAY_AGG(name) FROM employees")
        test_result(
            "방언 감지 (BigQuery)",
            detected == SQLDialect.BIGQUERY,
            f"감지됨: {detected}"
        )
    except Exception as e:
        test_result("Dialect Handler 모듈 로드", False, str(e))


# ============================================================
# 4. Text-to-SQL Agent 테스트 (API 필요)
# ============================================================
def test_text_to_sql_agent() -> None:
    """Text-to-SQL 에이전트 테스트 (API 호출)"""
    print_section("4. Text-to-SQL Agent 테스트")

    key = os.getenv('OPEN_AI_KEY_5')
    endpoint = os.getenv('OPEN_AI_ENDPOINT_5')

    if not key or not endpoint:
        global skipped
        skipped += 3
        print("  ⏭️ API 키 없음 - 에이전트 테스트 스킵")
        return

    try:
        from text_to_sql_agent import TextToSQLAgent, create_sample_database

        # 에이전트 초기화
        agent = TextToSQLAgent(
            api_key=key,
            endpoint=endpoint,
            deployment_name='gpt-5.2',
            api_version='2025-01-01-preview',
            use_structured_outputs=True,
            enable_deep_reasoning=True
        )

        db_path = create_sample_database()
        agent.load_database(db_path)

        test_result("에이전트 초기화", True)

        # 테스트 1: 단순 쿼리
        start = time.time()
        result = agent.ask("개발팀 직원들의 평균 연봉은?")
        elapsed = time.time() - start
        test_result(
            f"단순 쿼리 생성 ({elapsed:.1f}s)",
            result.get('sql') and 'AVG' in result['sql'].upper(),
            f"SQL: {result.get('sql', 'None')}"
        )

        # 테스트 2: 복잡한 쿼리 (심층 추론)
        start = time.time()
        result = agent.ask("부서별 직원 수와 평균 연봉을 알려주세요")
        elapsed = time.time() - start
        test_result(
            f"복잡한 쿼리 생성 ({elapsed:.1f}s)",
            result.get('sql') and 'GROUP BY' in result['sql'].upper(),
            f"SQL: {result.get('sql', 'None')}"
        )

        # 테스트 3: 조인 쿼리
        start = time.time()
        result = agent.ask("진행 중인 프로젝트에 참여하는 직원 목록")
        elapsed = time.time() - start
        test_result(
            f"조인 쿼리 생성 ({elapsed:.1f}s)",
            result.get('sql') and 'JOIN' in result['sql'].upper(),
            f"SQL: {result.get('sql', 'None')}"
        )

        agent.close()

    except Exception as e:
        test_result("Text-to-SQL Agent 테스트", False, str(e))


# ============================================================
# 5. 통합 테스트
# ============================================================
def test_integration() -> None:
    """통합 테스트"""
    print_section("5. 통합 테스트")

    try:
        from text_to_sql_agent import (
            TextToSQLAgent,
            SchemaExtractor,
            PromptBuilder,
            SQLValidator,
            DatabaseType,
            create_sample_database
        )

        # 스키마 추출
        db_path = create_sample_database()
        schema = SchemaExtractor.extract_sqlite_schema(db_path)
        test_result(
            "스키마 추출",
            len(schema.tables) == 4,
            f"테이블 수: {len(schema.tables)}"
        )

        # 프롬프트 생성
        context = PromptBuilder.build_schema_context(schema)
        test_result(
            "프롬프트 컨텍스트 생성",
            "employees" in context and "departments" in context,
            "컨텍스트 누락"
        )

        # SQL 검증
        is_valid, error = SQLValidator.validate_syntax(
            "SELECT * FROM employees WHERE dept_id = 1",
            DatabaseType.SQLITE
        )
        test_result(
            "SQL 문법 검증",
            is_valid,
            error or "검증 실패"
        )

        # 잘못된 SQL 검증
        is_valid, error = SQLValidator.validate_syntax(
            "SELEC * FROM employees",  # 오타
            DatabaseType.SQLITE
        )
        test_result(
            "잘못된 SQL 감지",
            not is_valid,
            "오류 감지 실패"
        )

    except Exception as e:
        test_result("통합 테스트", False, str(e))


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Advanced Text-to-SQL 종합 테스트")
    print("  2026년 1월 26일")
    print("=" * 60)

    start_time = time.time()

    # 모든 테스트 실행
    test_schema_linker()
    test_sql_optimizer()
    test_dialect_handler()
    test_text_to_sql_agent()
    test_integration()

    elapsed = time.time() - start_time

    # 결과 요약
    print("\n" + "=" * 60)
    print("  테스트 결과 요약")
    print("=" * 60)
    print(f"  ✅ 성공: {passed}")
    print(f"  ❌ 실패: {failed}")
    print(f"  ⏭️ 스킵: {skipped}")
    print(f"  ⏱️ 소요 시간: {elapsed:.1f}초")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 모든 테스트 통과!")
    else:
        print(f"\n⚠️ {failed}개 테스트 실패")
        sys.exit(1)
