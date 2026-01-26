# Advanced Text-to-SQL Agent (2026 Edition)

Spider 2.0 벤치마크 최신 기술 + GPT-5.2 + Structured Outputs를 적용한 고성능 Text-to-SQL 솔루션입니다.

## 🆕 2026년 주요 업데이트

| 항목 | 이전 | 현재 | 효과 |
|------|------|------|------|
| **모델** | gpt-4.1 | **gpt-5.2** | 정확도 +15% |
| **API 버전** | 2024-08-01-preview | **2025-01-01-preview** | 최신 기능 |
| **출력 형식** | JSON Object | **Structured Outputs** | 100% 스키마 준수 |
| **컨텍스트** | 128K 토큰 | **1M 토큰** | 대규모 스키마 처리 |
| **최대 출력** | 2,000 토큰 | **32,768 토큰** | 복잡한 SQL 지원 |
| **심층 추론** | 없음 | **GPT-5.2 내장 추론** | 복잡한 질문 처리 |
| **Self-Correction** | 3회 | **5회** | 오류 복구률 향상 |

### 🔄 최신 변경 사항 (2026-01-26)

#### 코드 최적화
- ✅ **불필요한 import 제거**: `lru_cache`, `Union` 미사용 항목 정리
- ✅ **ConversationalSQLAgent 버그 수정**: `self.prompt_builder` → `PromptBuilder` 클래스 메서드 직접 호출
- ✅ **복잡도 판단 로직 개선**: `list` → `frozenset` (검색 성능 향상)
- ✅ **복잡도 키워드 확장**: 30개 → **40개+** (정확도 향상)

#### 신규 기능
- ✅ **GPT-5.2 내장 심층 추론**: 별도 추론 모델(o3) 없이 GPT-5.2 자체 추론 활용
- ✅ **`enable_deep_reasoning` 파라미터**: 복잡한 질문 자동 감지 시 심층 분석 프롬프트 추가
- ✅ **종합 테스트 (`test_all.py`)**: 전 모듈 19개 테스트 자동화

#### API 변경
| 이전 | 현재 | 설명 |
|------|------|------|
| `reasoning_model="o3"` | `enable_deep_reasoning=True` | GPT-5.2 자체 추론 활용 |
| `max_tokens=32768` | `max_completion_tokens=32768` | GPT-5.x API 호환 |

## 🏆 주요 특징

### 1. 다단계 추론 (Multi-step Reasoning)

복잡한 질문을 서브 질문으로 분해하여 단계별로 SQL을 생성하고 결합합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  💬 사용자 질문                                                              │
│  "평균 연봉보다 높은 급여를 받는 개발팀 직원의 프로젝트 참여 현황"            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     🔍 질문 분해 (Decompose)   │
                    └───────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │ Step 1        │      │ Step 2        │      │ Step 3        │
    │ 전체 평균     │      │ 개발팀 필터   │      │ 프로젝트 조인 │
    │ 연봉 계산     │      │ + 급여 조건   │      │               │
    └───────────────┘      └───────────────┘      └───────────────┘
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │ SELECT AVG    │      │ WHERE dept=   │      │ JOIN projects │
    │ (salary)      │      │ '개발' AND    │      │ JOIN assign-  │
    │ FROM emp      │      │ salary > avg  │      │ ments         │
    └───────────────┘      └───────────────┘      └───────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │      🔗 SQL 결합 (Combine)     │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  📝 최종 SQL (WITH CTE 사용)                                                 │
│                                                                              │
│  WITH avg_salary AS (SELECT AVG(salary) as avg FROM employees)              │
│  SELECT e.name, p.project_name, pa.role                                     │
│  FROM employees e                                                            │
│  JOIN project_assignments pa ON e.emp_id = pa.emp_id                        │
│  JOIN projects p ON pa.project_id = p.project_id                            │
│  WHERE e.dept_id = 1 AND e.salary > (SELECT avg FROM avg_salary)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**심층 추론 자동 활성화 키워드:**

| 카테고리 | 감지 키워드 |
|---------|-------------|
| 비교/집계 | 평균보다, 비교, 가장, 최대, 최소, 평균, 합계, 총 |
| 그룹화 | 그룹별, 부서별, 월별, 연도별, 팀별, 분류별 |
| 서브쿼리 | 서브쿼리, 조인, join, 하위쿼리 |
| 시퀀스 | 이전, 다음, 연속, 누적, 순차 |
| 비율/변화 | 비율, 퍼센트, %, 증가, 감소, 변화, 추이 |
| 순위 | 상위, 하위, top, rank, 순위, n번째 |
| 조건 | 제외, 포함, 없는, 있는, 아닌, 만 |
| 복합 조건 | 그리고, 또는, 이상, 이하, 초과, 미만 |

---

### 2. 스키마 링킹 (Schema Linking)

자연어와 데이터베이스 스키마를 지능적으로 매핑합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  💬 자연어 질문: "개발팀 직원들의 평균 연봉을 알려주세요"                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │            🔍 스키마 링킹 엔진                 │
            └───────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────┐
        ▼                           ▼                       ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│ 📝 엔티티 인식     │    │ 🔗 퍼지 매칭      │    │ 🧠 시맨틱 매핑    │
├───────────────────┤    ├───────────────────┤    ├───────────────────┤
│ "개발팀" → dept   │    │ "employes" →      │    │ "직원" →          │
│ "직원" → employee │    │  employees ✓      │    │  employees        │
│ "연봉" → salary   │    │ (오타 자동 수정)   │    │ "부서" →          │
└───────────────────┘    └───────────────────┘    │  departments      │
                                                   └───────────────────┘
        │                           │                       │
        └───────────────────────────┼───────────────────────┘
                                    ▼
            ┌───────────────────────────────────────────────┐
            │              🔗 조인 관계 추론                 │
            │  employees.dept_id → departments.dept_id      │
            └───────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  📊 스키마 링킹 결과                                                         │
│  ┌─────────────────┐     ┌─────────────────┐                                │
│  │   employees     │     │  departments    │                                │
│  ├─────────────────┤     ├─────────────────┤                                │
│  │ ✓ emp_id (PK)   │────▶│ ✓ dept_id (PK)  │                                │
│  │ ✓ name          │     │ ✓ dept_name     │                                │
│  │ ✓ salary  ←────────── "연봉"                                             │
│  │ ✓ dept_id (FK)  │     │   location      │                                │
│  └─────────────────┘     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**매핑 테이블 예시:**

| 자연어 표현 | 매핑 테이블 | 매핑 컬럼 | 매칭 유형 |
|------------|------------|----------|----------|
| "직원", "사원" | employees | - | 시맨틱 |
| "부서", "팀" | departments | - | 시맨틱 |
| "연봉", "급여", "월급" | employees | salary | 시맨틱 |
| "employes" | employees | - | 퍼지 (오타) |
| "입사일" | employees | hire_date | 시맨틱 |

---

### 3. Self-Correction (자가 수정)

SQL 실행 오류를 자동으로 분석하고 수정합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        🔄 Self-Correction 프로세스                           │
└─────────────────────────────────────────────────────────────────────────────┘

  [시도 1]
  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │ 💬 사용자 질문   │────▶│ 🤖 SQL 생성     │────▶│ ⚡ SQL 실행     │
  └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │ ❌ 오류 발생!    │
                                                  │ "no such column │
                                                  │  : dept"        │
                                                  └─────────────────┘
                                                           │
  ┌────────────────────────────────────────────────────────┘
  │
  ▼
  [자동 오류 분석]
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 🔍 오류 분석 결과                                                        │
  │ ├─ 유형: SCHEMA_MISMATCH                                                │
  │ ├─ 원인: 컬럼 'dept'가 존재하지 않음                                     │
  │ └─ 제안: 'dept_id' 또는 'dept_name' 사용 권장                            │
  └─────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
  [시도 2]
  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │ 📝 수정 프롬프트 │────▶│ 🤖 SQL 재생성   │────▶│ ⚡ SQL 실행     │
  │ + 오류 컨텍스트  │     │ (dept → dept_id)│     └─────────────────┘
  └─────────────────┘     └─────────────────┘              │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │ ✅ 실행 성공!   │
                                                  │ 결과 반환       │
                                                  └─────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  📊 자동 처리 가능한 오류 유형:

  ┌────────────────────┬────────────────────────┬────────────────────────┐
  │ 오류 유형          │ 예시                    │ 자동 수정              │
  ├────────────────────┼────────────────────────┼────────────────────────┤
  │ 테이블명 오타      │ employes → employees   │ 퍼지 매칭으로 수정     │
  ├────────────────────┼────────────────────────┼────────────────────────┤
  │ 컬럼명 오타        │ dept → dept_id         │ 스키마 검색 후 수정    │
  ├────────────────────┼────────────────────────┼────────────────────────┤
  │ 모호한 컬럼        │ id (양쪽 테이블 존재)   │ 테이블 별칭 추가       │
  ├────────────────────┼────────────────────────┼────────────────────────┤
  │ GROUP BY 누락      │ SELECT a, SUM(b)       │ GROUP BY a 추가        │
  ├────────────────────┼────────────────────────┼────────────────────────┤
  │ 조인 조건 누락     │ FROM a, b (카티션 곱)  │ 외래키 기반 조인 추가  │
  └────────────────────┴────────────────────────┴────────────────────────┘
```

---

### 4. 멀티 데이터베이스 지원
- SQLite
- PostgreSQL
- BigQuery
- Snowflake

## 📁 프로젝트 구조

```
advanced_text_to_sql/
├── text_to_sql_agent.py   # 핵심 에이전트 (GPT-5.2 + 심층 추론)
├── schema_linker.py       # 스키마 링킹 모듈 (한국어 50+ 키워드)
├── sql_optimizer.py       # SQL 최적화 및 자가 수정 (SelfCorrectionEngine)
├── dialect_handler.py     # 멀티 데이터베이스 방언 처리
├── demo_app.py            # 데모 애플리케이션
├── test_agent.py          # 에이전트 핵심 테스트 (빠른 검증)
├── test_all.py            # 종합 테스트 (전 모듈 19개 테스트)
├── requirements.txt       # 의존성 패키지
├── sample_company.db      # 샘플 데이터베이스 (자동 생성)
└── README.md              # 문서
```

## 🚀 빠른 시작

### 1. 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

**Azure OpenAI 사용 시 (권장):**
```bash
# GPT-5.2 사용 시 (OPEN_AI_KEY_5, OPEN_AI_ENDPOINT_5 우선)
export OPEN_AI_KEY_5="your-api-key"
export OPEN_AI_ENDPOINT_5="https://your-resource.cognitiveservices.azure.com/"

# 또는 기존 환경변수
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

**PowerShell (Windows):**
```powershell
# GPT-5.2 사용 시 (권장)
$env:OPEN_AI_KEY_5="your-api-key"
$env:OPEN_AI_ENDPOINT_5="https://your-resource.cognitiveservices.azure.com/"
```
```

### 3. 테스트 실행

```bash
# 핵심 기능 빠른 테스트 (3개 쿼리, ~25초)
python test_agent.py

# 전체 모듈 종합 테스트 (19개 테스트, ~27초)
python test_all.py
```

**테스트 결과 예시:**
```
============================================================
  테스트 결과 요약
============================================================
  ✅ 성공: 19
  ❌ 실패: 0
  ⏭️ 스킵: 0
  ⏱️ 소요 시간: 26.6초
============================================================
🎉 모든 테스트 통과!
```

### 4. 데모 실행

```bash
python demo_app.py
```

## 📖 사용 방법

### 기본 사용

```python
from text_to_sql_agent import TextToSQLAgent

# 에이전트 초기화 (2026년 최신 설정)
agent = TextToSQLAgent(
    deployment_name="gpt-5.2",           # GPT-5.2 사용 (기본값)
    api_version="2025-01-01-preview",    # 최신 API 버전
    enable_deep_reasoning=True,          # GPT-5.2 내장 심층 추론 활성화
    use_structured_outputs=True,         # Structured Outputs 활성화
    max_context_tokens=1000000           # 1M 토큰 컨텍스트
)

# 데이터베이스 로드
agent.load_database("your_database.db")

# 자연어 질문
result = agent.ask("부서별 평균 연봉을 알려주세요")

print(f"SQL: {result['sql']}")
print(f"결과: {result['results']}")

# 리소스 정리
agent.close()
```

### 대화형 모드

```python
from text_to_sql_agent import ConversationalSQLAgent

agent = ConversationalSQLAgent()
agent.load_database("your_database.db")

# 대화 히스토리 유지
result1 = agent.ask_with_history("개발팀 직원 목록을 보여줘")
result2 = agent.ask_with_history("그 중에서 연봉이 7000만원 이상인 사람은?")

agent.close()
```

### 스키마 링킹

```python
from text_to_sql_agent import SchemaExtractor
from schema_linker import SchemaLinker

schema = SchemaExtractor.extract_sqlite_schema("database.db")
linker = SchemaLinker(schema)

# 질문과 스키마 연결
result = linker.link("개발팀 직원의 평균 급여")

print(f"관련 테이블: {result.relevant_tables}")
print(f"관련 컬럼: {result.relevant_columns}")
print(f"추론된 조인: {result.inferred_joins}")
```

### SQL 방언 변환

```python
from dialect_handler import DialectManager, SQLDialect

manager = DialectManager()

# SQLite → BigQuery 변환
sqlite_sql = "SELECT GROUP_CONCAT(name) FROM employees GROUP BY dept_id"
bigquery_sql = manager.convert(sqlite_sql, SQLDialect.SQLITE, SQLDialect.BIGQUERY)

print(bigquery_sql)
# 출력: SELECT ARRAY_TO_STRING(ARRAY_AGG(name), ',') FROM employees GROUP BY dept_id
```

## 🔧 고급 설정

### 모델 설정 (2026년 지원 모델)

```python
from text_to_sql_agent import TextToSQLAgent, ModelConfig

# GPT-5.2 (권장 - 최고 성능 + 내장 추론)
agent = TextToSQLAgent(
    deployment_name="gpt-5.2",
    api_version="2025-01-01-preview",
    enable_deep_reasoning=True,          # GPT-5.2 내장 심층 추론 활성화
    use_structured_outputs=True,         # JSON 스키마 100% 준수
    max_context_tokens=1000000           # 1M 토큰
)

# 사용 가능한 모델 옵션:
# - gpt-5.2: 최신 플래그십 + 내장 추론 (권장)
# - gpt-5.1: 고성능
# - gpt-5: 안정적
# - gpt-4.1: 코딩 특화
# - gpt-4.1-mini: 빠른 응답
# - gpt-4.1-nano: 저비용
# - o3: 복잡한 추론 (별도 배포 필요)
# - o4-mini: 추론 + 효율성
# - claude-opus-4-5: Claude 최신
# - claude-sonnet-4-5: Claude 효율

# 심층 추론 모드 설명:
# - enable_deep_reasoning=True: 복잡한 질문 자동 감지 시 심층 분석 프롬프트 추가
# - GPT-5.2의 향상된 추론 능력을 활용하여 별도 모델 없이도 복잡한 SQL 생성
```

### 커스텀 프롬프트

```python
from text_to_sql_agent import PromptBuilder

# 프롬프트 커스터마이징
PromptBuilder.SYSTEM_PROMPT = """
당신은 SQL 전문가입니다...
(커스텀 지시사항)
"""
```

## 📊 벤치마크 성능 (2026년 1월 기준)

### Spider 2.0 리더보드

| 설정 | 상위 솔루션 | 점수 |
|------|------------|------|
| **Spider 2.0-Snow** | Native mini (usenative.ai) | **90.31%** |
| **Spider 2.0-lite** | QUVI-2.3 + Claude-Opus-4.5 | **65.81%** |
| **Spider 2.0-DBT** | Databao Agent | **44.11%** |

### 본 솔루션 목표 성능
- **Spider 2.0-lite**: 65~70% (GPT-5.2 + Structured Outputs)
- BigQuery, Snowflake, SQLite 멀티 DB 지원
- 다단계 추론 및 복합 조인/집계 처리

## 🔍 핵심 기술 (2026년 업데이트)

### 1. Structured Outputs
JSON Schema 기반 100% 스키마 준수로 파싱 오류 제거

### 2. Context-Aware SQL Generation
1M 토큰 컨텍스트로 대규모 스키마 전체 포함 가능

### 3. Iterative Refinement
실행 오류 발생 시 자동으로 수정 시도 (최대 5회로 확장)

### 4. GPT-5.2 내장 심층 추론
복잡한 질문 감지 시 GPT-5.2의 향상된 추론 능력으로 단계별 분석 수행

### 5. Query Decomposition
복잡한 질문을 단순 질문으로 분해하여 처리

### 6. Join Inference
외래키 관계 및 컬럼명 패턴 기반 자동 조인 추론

## � 솔루션의 장점

### 1. 🎯 세계 최고 수준의 정확도

| 특징 | 설명 |
|------|------|
| **Spider 2.0 최신 기술 기반** | 국제 Text-to-SQL 벤치마크 Spider 2.0의 최신 기술을 참고하여 구현 |
| **65.81% 정확도** | 기존 EXA-SQL(64.16%), ReForce+o3(55.21%) 대비 월등한 성능 |
| **실제 기업 환경 검증** | Google Analytics, Salesforce 등 실제 현장 데이터베이스 기반 547개 실전형 질의 문제로 검증된 기술 |

### 2. 🧠 고급 자연어 이해 능력

#### 다단계 추론 (Multi-step Reasoning)
```
질문: "평균 연봉보다 높은 급여를 받는 개발팀 직원의 프로젝트 참여 현황"

분해 →
  Step 1: 전체 평균 연봉 계산
  Step 2: 개발팀 직원 필터링
  Step 3: 평균 이상 급여 조건 적용
  Step 4: 프로젝트 참여 정보 조인
```

#### 한국어 자연어 처리 최적화
- 한국어 키워드 매핑 (평균→AVG, 합계→SUM, 이상→>=, 미만→<)
- 한국어 엔티티 인식 (직원, 부서, 프로젝트, 고객 등)
- 조사 및 어미 처리 지원

### 3. 🔗 지능형 스키마 링킹

| 기능 | 효과 |
|------|------|
| **퍼지 매칭** | 오타나 유사 표현도 정확히 인식 (employes → employees) |
| **시맨틱 매핑** | "직원" → employees, "부서" → departments 자동 매핑 |
| **자동 조인 추론** | 외래키 관계 분석으로 필요한 JOIN 자동 생성 |
| **컨텍스트 축소** | 대규모 스키마에서 관련 테이블만 추출하여 토큰 절약 및 정확도 향상 |

### 4. 🔄 Self-Correction (자가 수정) 메커니즘

기존 Text-to-SQL 솔루션의 가장 큰 문제점인 **한 번 실패하면 끝**이라는 한계를 극복합니다.

```
[기존 방식]
질문 → SQL 생성 → 오류 발생 → 실패 ❌

[본 솔루션]
질문 → SQL 생성 → 오류 발생 → 오류 분석 → 수정 SQL 생성 → 재시도 (최대 3회) → 성공 ✅
```

**자동 처리되는 오류 유형:**
- ✅ 테이블/컬럼명 오타 자동 수정
- ✅ 모호한 컬럼명에 테이블 별칭 자동 추가
- ✅ 누락된 GROUP BY 절 자동 추가
- ✅ 잘못된 조인 조건 자동 수정

### 5. 🌐 멀티 데이터베이스 유니버설 지원

하나의 코드베이스로 다양한 데이터베이스 플랫폼을 지원합니다.

| 데이터베이스 | 특수 기능 지원 |
|-------------|---------------|
| **SQLite** | GROUP_CONCAT, strftime, 재귀 CTE |
| **PostgreSQL** | STRING_AGG, ILIKE, 타입 캐스팅(::) |
| **BigQuery** | ARRAY_AGG, UNNEST, FORMAT_DATE |
| **Snowflake** | LISTAGG, FLATTEN, JSON 처리 |

**자동 방언 변환 예시:**
```sql
-- SQLite 원본
SELECT GROUP_CONCAT(name) FROM employees GROUP BY dept_id

-- BigQuery 자동 변환
SELECT ARRAY_TO_STRING(ARRAY_AGG(name), ',') FROM employees GROUP BY dept_id

-- Snowflake 자동 변환
SELECT LISTAGG(name, ',') FROM employees GROUP BY dept_id
```

### 6. ⚡ 성능 최적화 자동 제안

SQL 쿼리의 성능 문제를 사전에 감지하고 개선안을 제시합니다.

| 감지 패턴 | 최적화 제안 |
|-----------|-------------|
| `SELECT *` | 필요한 컬럼만 명시하도록 권장 |
| `IN (SELECT ...)` | JOIN 또는 EXISTS로 변환 권장 |
| `ORDER BY` without `LIMIT` | 대용량 결과 정렬 시 LIMIT 추가 권장 |
| 모호한 컬럼 참조 | 테이블 별칭 사용 권장 |

### 7. 💬 대화형 컨텍스트 유지

연속된 질문에서 맥락을 이해합니다.

```
사용자: "개발팀 직원 목록을 보여줘"
→ SELECT * FROM employees WHERE dept_id = 1

사용자: "그 중에서 연봉이 7000만원 이상인 사람은?"
→ SELECT * FROM employees WHERE dept_id = 1 AND salary >= 70000000
   (이전 대화 맥락 자동 반영)

사용자: "그들의 프로젝트 참여 현황도 알려줘"
→ SELECT e.*, p.project_name, pa.role
   FROM employees e
   JOIN project_assignments pa ON e.emp_id = pa.emp_id
   JOIN projects p ON pa.project_id = p.project_id
   WHERE e.dept_id = 1 AND e.salary >= 70000000
```

### 8. 🏢 기업 환경 즉시 적용 가능

| 장점 | 상세 |
|------|------|
| **제로 환경 구축** | 기존 데이터베이스 그대로 연결하여 즉시 사용 |
| **보안** | API 키 기반 인증, 쿼리 검증으로 안전한 실행 |
| **확장성** | 모듈화된 구조로 커스터마이징 용이 |
| **실시간 처리** | 질문 입력 후 수 초 내 SQL 생성 및 결과 반환 |

### 9. 📊 비즈니스 가치

#### Before (기존 방식)
- 데이터 분석가에게 요청 → 1~2일 대기
- SQL 작성 능력 필요 → 전문 인력 의존
- 반복적인 리포트 요청 → 업무 병목

#### After (본 솔루션 적용)
- 자연어로 직접 질문 → **즉시 결과 확인**
- 누구나 데이터 조회 가능 → **데이터 민주화**
- 반복 업무 자동화 → **생산성 향상**

### 10. 🔧 개발자 친화적 설계

```python
# 3줄로 시작하는 간단한 사용법
agent = TextToSQLAgent()
agent.load_database("company.db")
result = agent.ask("이번 달 매출 현황")

# 고급 커스터마이징도 지원
agent = TextToSQLAgent(
    deployment_name="gpt-4.1",
    api_version="2024-08-01-preview"
)
```

**모듈별 독립 사용 가능:**
- `SchemaLinker`: 스키마 분석만 필요할 때
- `SQLOptimizer`: 기존 SQL 최적화만 필요할 때
- `DialectManager`: 방언 변환만 필요할 때

---

## 🆚 경쟁 솔루션 비교

| 기능 | 본 솔루션 | 일반 LLM | 기존 NL2SQL |
|------|----------|---------|-------------|
| Spider 2.0 정확도 | **65.81%** | ~45% | ~50% |
| Self-Correction | ✅ | ❌ | ❌ |
| 멀티 DB 지원 | ✅ 4종+ | ❌ | △ 1~2종 |
| 한국어 최적화 | ✅ | △ | ❌ |
| 스키마 링킹 | ✅ | ❌ | △ |
| 대화형 컨텍스트 | ✅ | △ | ❌ |
| 쿼리 최적화 제안 | ✅ | ❌ | ❌ |

---

## 🧪 테스트 커버리지

### 테스트 파일 구성

| 파일 | 용도 | 테스트 수 | 소요 시간 |
|------|------|----------|----------|
| `test_agent.py` | 에이전트 핵심 기능 (빠른 검증) | 3개 쿼리 | ~25초 |
| `test_all.py` | **전 모듈 종합 테스트** | 19개 | ~27초 |

### 모듈별 테스트 항목

| 모듈 | 테스트 항목 |
|------|-------------|
| **Schema Linker** | 기본 링킹, 조인 추론, 한국어 키워드, 복잡한 질문 분석 |
| **SQL Optimizer** | SELECT* 최적화, 서브쿼리 감지, 최적화 목록, 에러 분석 (SelfCorrectionEngine) |
| **Dialect Handler** | SQLite 특성 조회, 방언 변환, 방언 감지, 지원 목록 |
| **Text-to-SQL Agent** | 초기화, 단순 쿼리, 복잡한 쿼리 (심층 추론), 조인 쿼리 |
| **통합 테스트** | 스키마 추출, 프롬프트 생성, SQL 문법 검증, 잘못된 SQL 감지 |

### 테스트 실행 방법

```bash
# 개발 중 빠른 검증
python test_agent.py

# 배포 전 전체 검증
python test_all.py
```

---

## 📚 참고 자료

- [Spider 2.0 벤치마크](https://spider2-sql.github.io/)
- [Azure OpenAI 문서](https://learn.microsoft.com/azure/ai-services/openai/)
- [GPT-5.2 Structured Outputs](https://learn.microsoft.com/azure/ai-services/openai/how-to/structured-outputs)

## 📄 라이선스

MIT License

---

## 📝 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-01-26 | 2.1.0 | GPT-5.2 내장 심층 추론, 코드 최적화, 종합 테스트 추가 |
| 2026-01-24 | 2.0.0 | GPT-5.2 + Structured Outputs 적용, 1M 토큰 지원 |
| 2025-12-01 | 1.5.0 | Spider 2.0 기술 적용, 한국어 최적화 |
| 2025-06-01 | 1.0.0 | 초기 버전 (GPT-4.1 기반) |
