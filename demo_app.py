"""
Advanced Text-to-SQL Demo Application

Spider 2.0 벤치마크 1위 기술 기반의 Text-to-SQL 데모 애플리케이션입니다.

실행 방법:
    python demo_app.py

환경 변수 설정 필요:
    - AZURE_OPENAI_API_KEY: Azure OpenAI API 키
    - AZURE_OPENAI_ENDPOINT: Azure OpenAI 엔드포인트
    
또는 Anthropic Claude 사용:
    - ANTHROPIC_API_KEY: Anthropic API 키
"""

import os
import sys
import json
import sqlite3
from typing import Optional
from datetime import datetime

# 로컬 모듈 임포트
from text_to_sql_agent import (
    TextToSQLAgent, 
    ConversationalSQLAgent,
    create_sample_database,
    SchemaExtractor,
    DatabaseType
)
from schema_linker import SchemaLinker, QueryDecomposer
from sql_optimizer import SQLOptimizer, SelfCorrectionEngine, SQLCorrectionPipeline
from dialect_handler import DialectManager, SQLDialect, MultiDatabaseQuery


# 상수
_BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🏆 Advanced Text-to-SQL Agent                                 ║
║   ─────────────────────────────────────────────────────────────  ║
║   Based on Spider 2.0 Benchmark Latest Technology               ║
║                                                                  ║
║                                                                  ║
║   Features:                                                      ║
║   • Multi-step Reasoning                                         ║
║   • Schema Linking                                               ║
║   • Self-Correction                                              ║
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


def print_banner() -> None:
    """배너 출력"""
    print(_BANNER)


def print_menu() -> None:
    """메뉴 출력"""
    print(_MENU)


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


def run_sample_questions(db_path: str, use_api: bool = False):
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
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n질문 {i}: {question}")
        
        # 스키마 링킹 결과
        linking_result = linker.link(question)
        print(f"  📎 관련 테이블: {', '.join(linking_result.relevant_tables)}")
        
        if use_api:
            # 실제 API 호출 (API 키 설정 시)
            try:
                agent = TextToSQLAgent()
                agent.load_database(db_path)
                result = agent.ask(question)
                print(f"  🔍 생성된 SQL: {result['sql']}")
                print(f"  💬 설명: {result['explanation']}")
                if 'results' in result and result['results']:
                    print(f"  📊 결과 ({result['row_count']}행):")
                    for row in result['results'][:3]:
                        print(f"     {row}")
                agent.close()
            except Exception as e:
                print(f"  ⚠️ API 호출 오류: {e}")
        else:
            # API 없이 스키마 링킹만 표시
            print(f"  💡 집중 스키마:")
            focused_schema = linker.get_focused_schema(question)
            for line in focused_schema.split('\n')[:10]:
                print(f"     {line}")


def interactive_mode(db_path: str):
    """대화형 모드"""
    print("\n💬 대화형 모드")
    print("=" * 50)
    print("질문을 입력하세요. 'exit' 또는 '종료'로 나갑니다.")
    print("대화 히스토리가 유지됩니다.\n")
    
    try:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            print("⚠️ AZURE_OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            print("   스키마 링킹 데모만 실행합니다.\n")
            
            schema = SchemaExtractor.extract_sqlite_schema(db_path)
            linker = SchemaLinker(schema)
            
            while True:
                question = input("\n🗣️ 질문: ").strip()
                if question.lower() in _EXIT_COMMANDS:
                    break
                if not question:
                    continue
                
                print("\n📎 스키마 분석 결과:")
                result = linker.link(question)
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
        
        while True:
            question = input("\n🗣️ 질문: ").strip()
            if question.lower() in _EXIT_COMMANDS:
                break
            if not question:
                continue
            
            try:
                result = agent.ask_with_history(question)
                print(f"\n🔍 SQL:\n{result['sql']}")
                print(f"\n💬 설명: {result['explanation']}")
                
                if 'results' in result:
                    print(f"\n📊 결과 ({result['row_count']}행):")
                    for row in result['results'][:10]:
                        print(f"   {row}")
                    if result['row_count'] > 10:
                        print(f"   ... 외 {result['row_count'] - 10}행")
                        
            except Exception as e:
                print(f"\n❌ 오류: {e}")
        
        agent.close()
        
    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")


def single_question_mode(db_path: str):
    """단일 질문 모드"""
    print("\n🎯 자연어 질문 입력")
    print("=" * 50)
    
    question = input("질문을 입력하세요: ").strip()
    if not question:
        print("질문이 입력되지 않았습니다.")
        return
    
    schema = SchemaExtractor.extract_sqlite_schema(db_path)
    linker = SchemaLinker(schema)
    
    print(f"\n📎 스키마 분석...")
    result = linker.link(question)
    print(f"   관련 테이블: {', '.join(result.relevant_tables)}")
    
    # API 키 확인
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        try:
            print("\n🤖 SQL 생성 중...")
            agent = TextToSQLAgent()
            agent.load_database(db_path)
            result = agent.ask(question)
            
            print(f"\n🔍 생성된 SQL:")
            print(f"   {result['sql']}")
            print(f"\n💬 설명: {result['explanation']}")
            print(f"🎯 신뢰도: {result['confidence']:.1%}")
            
            if 'results' in result:
                print(f"\n📊 실행 결과 ({result['row_count']}행):")
                for row in result['results'][:10]:
                    print(f"   {row}")
            
            agent.close()
        except Exception as e:
            print(f"\n❌ 오류: {e}")
    else:
        print("\n⚠️ AZURE_OPENAI_API_KEY가 설정되지 않아 SQL 생성을 건너뜁니다.")
        print("💡 집중 스키마:")
        print(linker.get_focused_schema(question))


def main():
    """메인 함수"""
    print_banner()
    
    # 샘플 데이터베이스 생성
    print("📦 샘플 데이터베이스 준비 중...")
    db_path = create_sample_database()
    print(f"   ✅ 데이터베이스 생성 완료: {db_path}")
    
    while True:
        print_menu()
        choice = input("\n선택: ").strip()
        
        if choice == "1":
            single_question_mode(db_path)
        elif choice == "2":
            demo_schema_info(db_path)
        elif choice == "3":
            demo_sql_optimization()
        elif choice == "4":
            demo_dialect_conversion()
        elif choice == "5":
            interactive_mode(db_path)
        elif choice == "6":
            use_api = os.getenv("AZURE_OPENAI_API_KEY") is not None
            run_sample_questions(db_path, use_api=use_api)
        elif choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("\n⚠️ 잘못된 선택입니다. 다시 선택해주세요.")


if __name__ == "__main__":
    main()
