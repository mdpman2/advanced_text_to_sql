"""
Text-to-SQL Agent 테스트 스크립트
GPT-5.2 + Structured Outputs 테스트
"""
import os
import sys

# 상위 디렉토리의 .env 파일 로드
from dotenv import load_dotenv
load_dotenv('../.env')

print('=' * 60)
print('Advanced Text-to-SQL Agent 테스트')
print('=' * 60)

# 환경변수 확인
key = os.getenv('OPEN_AI_KEY_5')
endpoint = os.getenv('OPEN_AI_ENDPOINT_5')

print(f"\n[환경변수 확인]")
print(f"OPEN_AI_KEY_5: {'✅ 설정됨' if key else '❌ 없음'}")
print(f"OPEN_AI_ENDPOINT_5: {endpoint if endpoint else '❌ 없음'}")

if not key or not endpoint:
    print("\n⚠️ 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

print("\n[에이전트 초기화]")
from text_to_sql_agent import TextToSQLAgent, create_sample_database

# 샘플 DB 생성
db_path = create_sample_database()
print(f"샘플 데이터베이스: {db_path}")

# 에이전트 초기화 (GPT-5.2 자체 심층 추론 활용)
agent = TextToSQLAgent(
    api_key=key,
    endpoint=endpoint,
    deployment_name='gpt-5.2',
    api_version='2025-01-01-preview',
    use_structured_outputs=True,
    enable_deep_reasoning=True  # GPT-5.2 내장 추론 모드 활성화
)

agent.load_database(db_path)
print("에이전트 초기화 완료!")

# 테스트 질문들
test_questions = [
    "개발팀 직원들의 평균 연봉은 얼마인가요?",
    "부서별 직원 수를 알려주세요",
    "진행 중인 프로젝트에 참여하는 직원 목록"
]

print("\n" + "=" * 60)
print("SQL 생성 테스트")
print("=" * 60)

for i, question in enumerate(test_questions, 1):
    print(f"\n[테스트 {i}]")
    print(f"질문: {question}")
    print("-" * 40)

    try:
        result = agent.ask(question)
        print(f"SQL: {result['sql']}")
        print(f"설명: {result['explanation']}")
        print(f"신뢰도: {result['confidence']:.2f}")

        if 'results' in result and result['results']:
            print(f"결과 ({result['row_count']}행):")
            for row in result['results'][:5]:
                print(f"  {row}")
    except Exception as e:
        print(f"오류: {e}")

agent.close()
print("\n✅ 테스트 완료!")
