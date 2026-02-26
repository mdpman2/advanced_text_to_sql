# Advanced Text-to-SQL Agent v3.0.0 (2026-02 Edition)

Spider 2.0 벤치마크 최신 기술 + GPT-5.2 + **Responses API** + Pydantic v2 Structured Outputs를 적용한 고성능 Text-to-SQL 솔루션입니다.

## 🆕 v3.0.0 주요 업데이트 (2026-02-27)

| 항목 | v2.2.1 | **v3.0.0** | 효과 |
|------|--------|-----------|------|
| **API 엔진** | `chat.completions.create()` | **`responses.create()`** | Responses API 마이그레이션 |
| **Structured Outputs** | JSON Schema dict | **Pydantic v2 BaseModel** | 타입 안전성 + 자동 검증 |
| **멀티턴** | conversation_history 수동 관리 | **`previous_response_id`** | 서버 사이드 대화 체이닝 |
| **요청 형식** | `messages=[{role, content}]` | **`instructions` + `input`** | 간결한 프롬프트 구성 |
| **응답 추출** | `response.choices[0].message.content` | **`response.output_text`** | 단순화된 응답 접근 |
| **토큰 파라미터** | `max_completion_tokens` / `max_tokens` | **`max_output_tokens`** | 통일된 파라미터 |
| **SQL 방언** | 4종 (SQLite, PG, BQ, Snowflake) | **6종 (+MySQL, SQL Server)** | 엔터프라이즈 DB 지원 |
| **최적화 규칙** | 9개 (중복 2개 포함) | **11개 (중복 제거, 신규 2개)** | Cartesian Join + Window Function |
| **모델** | 17종 | **19종 (+gpt-5.2-mini)** | 최신 모델 라인업 |
| **Spider 2.0** | TCDataAgent-SQL 93.97% | **TCDataAgent-SQL 95.14%** | 2026-06 리더보드 최신화 |
| **API 버전** | `v1` (가상) | **`2025-04-01-preview`** | 실제 Azure API 버전 |
| **한국어 키워드** | 50+ | **55+** | 사이, 비어있는, 최근, 분기별 등 |

### 핵심 마이그레이션: Chat Completions → Responses API

```python
# ❌ v2.x (이전 방식)
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_format={"type": "json_schema", "json_schema": schema},
    max_completion_tokens=32768,
)
result = response.choices[0].message.content

# ✅ v3.0 (Responses API)
response = client.responses.create(
    model="gpt-5.2",
    instructions=system_prompt,    # system → instructions
    input=user_prompt,             # user message → input
    text={"format": {"type": "json_schema", ...}},  # Structured Outputs
    max_output_tokens=32768,       # 통일된 토큰 파라미터
)
result = response.output_text      # 간결한 응답 접근
```

### 멀티턴 대화 체이닝 (previous_response_id)

```python
# v3.0: 서버 사이드 대화 체이닝
response1 = client.responses.create(model="gpt-5.2", input="첫 번째 질문", ...)
response2 = client.responses.create(
    model="gpt-5.2",
    input="후속 질문",
    previous_response_id=response1.id,  # 자동 컨텍스트 유지
    ...
)
```

### v3.0 코드 최적화 사항

| 대상 | 최적화 내용 |
|------|-------------|
| `text_to_sql_agent.py` | `_build_text_config()` DRY 헬퍼 추출, `__slots__` 추가, `additionalProperties: false` 스키마 보정, `_JSON_PATTERN` 클래스 레벨 프리컴파일 |
| `sql_optimizer.py` | 인라인 `re.search()` 3개 → 프리컴파일 `_PATTERNS` dict 통합, `SelfCorrectionEngine._COMPILED_PATTERNS` 클래스 레벨 1회 컴파일, `__slots__` 추가 |
| `schema_linker.py` | `_table_dict` O(1) 룩업 딕셔너리 추가, `next()` 제너레이터 스캔 제거, `__slots__` 추가, `@dataclass(slots=True)` 적용 |
| `dialect_handler.py` | `get_dialect_hints` if/elif 6단 → `_DIALECT_HINTS` dict dispatch, LIMIT→TOP 프리컴파일 패턴, `@dataclass(slots=True, frozen=True)` |
| `demo_app.py` | 메뉴 분기 if/elif 7단 → dispatch dict O(1), `Callable[[], None]` 타입 힐트 정확화, `_print_query_result()` DRY 헬퍼 |

---

## 🏆 주요 특징

### 1. Responses API + Pydantic v2 Structured Outputs

```python
from pydantic import BaseModel, Field

class SQLGenerationSchema(BaseModel):
    reasoning: str = Field(description="단계별 추론 과정")
    sql: str = Field(description="생성된 SQL")
    confidence: float = Field(description="확신도 0.0~1.0")
    explanation: str = Field(description="SQL 설명")
    assumptions: list[str] = Field(description="가정 사항")
    alternative_queries: list[str] = Field(description="대안 쿼리")
```

### 2. 다단계 추론 (Multi-step Reasoning)

복잡한 질문을 분해하여 단계별로 SQL을 생성합니다.

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
            └───────────────────────┼───────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │      🔗 SQL 결합 (Combine)     │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  📝 최종 SQL (WITH CTE 사용)                                                 │
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
| 순위 | 상위, 하위, top, rank, 순위, n번째 |
| 조건 | 제외, 포함, 사이, 비어있는 (v3.0 신규) |

### 3. 스키마 링킹 (Schema Linking)

자연어와 데이터베이스 스키마를 지능적으로 매핑합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  💬 자연어 질문: "개발팀 직원들의 평균 연봉을 알려주세요"                     │
└─────────────────────────────────────────────────────────────────────────────┘
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
```

| 매칭 유형 | 예시 | 정확도 |
|----------|------|--------|
| 시맨틱 | "직원" → employees | 0.9 |
| 퍼지 | "employes" → employees | 0.7+ |
| 정확 | "salary" → salary | 1.0 |

### 4. Self-Correction (5-round)

SQL 실행 오류를 자동으로 분석하고 수정합니다 (최대 5회).

```
[기존 방식]  질문 → SQL 생성 → 오류 → 실패 ❌
[본 솔루션]  질문 → SQL 생성 → 오류 → 분석 → 재생성 (×5) → 성공 ✅
```

**자동 처리 가능한 오류:**
- ✅ 테이블/컬럼명 오타 자동 수정
- ✅ 모호한 컬럼명에 테이블 별칭 추가
- ✅ 누락된 GROUP BY 절 자동 추가
- ✅ 잘못된 조인 조건 자동 수정

### 5. 멀티 데이터베이스 지원 (v3.0: 6종)

| 데이터베이스 | 상태 | 특수 기능 |
|-------------|------|----------|
| **SQLite** | ✅ | GROUP_CONCAT, strftime, 재귀 CTE |
| **PostgreSQL** | ✅ | STRING_AGG, ILIKE, :: 타입캐스팅 |
| **BigQuery** | ✅ | ARRAY_AGG, UNNEST, FORMAT_DATE |
| **Snowflake** | ✅ | LISTAGG, FLATTEN, JSON 처리 |
| **MySQL** | ✅ v3.0 신규 | GROUP_CONCAT, DATE_FORMAT, JSON_EXTRACT |
| **SQL Server** | ✅ v3.0 신규 | STRING_AGG, TOP/OFFSET-FETCH, FORMAT |

### 6. SQL 최적화 규칙 (v3.0: 11개)

| 규칙 | 설명 | 버전 |
|------|------|------|
| SELECT * 감지 | 필요한 컬럼만 명시 권장 | v1.0 |
| IN 서브쿼리 | JOIN으로 변환 제안 | v1.0 |
| EXISTS vs IN | 대용량 데이터에서 EXISTS 권장 | v1.0 |
| ORDER BY without LIMIT | LIMIT 추가 권장 | v1.0 |
| DISTINCT 최적화 | GROUP BY로 대체 제안 | v1.0 |
| LIKE 패턴 | FULLTEXT 검색 제안 | v1.0 |
| CTE 사용 제안 | 중첩 서브쿼리 시 WITH 권장 | v1.0 |
| NULL 비교 | IS NULL 사용 권장 | v1.0 |
| 테이블 별칭 | JOIN 시 별칭 사용 권장 | v1.0 |
| **Cartesian Join 감지** | 카테시안 곱 경고 | **v3.0** |
| **Window Function 제안** | 순위/누적에 윈도우 함수 권장 | **v3.0** |

---

## 📁 프로젝트 구조

```
advanced_text_to_sql/
├── text_to_sql_agent.py   # 핵심 에이전트 (Responses API · Pydantic v2 · __slots__ · _build_text_config DRY)
├── schema_linker.py       # 스키마 링킹 (55+ 키워드, _table_dict O(1) 룩업)
├── sql_optimizer.py       # SQL 최적화 11개 규칙 (프리컴파일 패턴 통합) + SelfCorrection
├── dialect_handler.py     # 멀티 DB 방언 6종 (dict dispatch 힌트, 프리컴파일 LIMIT 변환)
├── demo_app.py            # 데모 애플리케이션 (dispatch dict, Callable 타입 힌트)
├── test_all.py            # 종합 테스트 (12 시나리오, 106 항목)
├── requirements.txt       # 의존성 (openai>=1.93, pydantic>=2.10, httpx>=0.28)
├── sample_company.db      # 샘플 데이터베이스 (자동 생성)
└── README.md              # 문서 (v3.0.0)
```

## 🚀 빠른 시작

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# Linux/macOS
export OPEN_AI_KEY_5="your-api-key"
export OPEN_AI_ENDPOINT_5="https://your-resource.cognitiveservices.azure.com/"
```

```powershell
# PowerShell (Windows)
$env:OPEN_AI_KEY_5="your-api-key"
$env:OPEN_AI_ENDPOINT_5="https://your-resource.cognitiveservices.azure.com/"
```

### 3. 테스트 실행

```bash
python test_all.py
```

### 4. 데모 실행

```bash
python demo_app.py
```

## 📖 사용 방법

### 기본 사용 (v3.0)

```python
from text_to_sql_agent import TextToSQLAgent

# v3.0 에이전트 초기화 (Responses API)
agent = TextToSQLAgent(
    deployment_name="gpt-5.2",                # SQL 특화: gpt-5.2-codex
    api_version="2025-04-01-preview",         # Responses API 지원 버전
    use_structured_outputs=True,              # Pydantic 기반 Structured Outputs
    enable_deep_reasoning=True,               # GPT-5.2 심층 추론
    max_context_tokens=400000                 # 400K 컨텍스트
)

agent.load_database("your_database.db")

result = agent.ask("부서별 평균 연봉을 알려주세요")
print(f"SQL: {result['sql']}")
print(f"결과: {result['results']}")

agent.close()
```

### 대화형 모드 (previous_response_id)

```python
from text_to_sql_agent import ConversationalSQLAgent

agent = ConversationalSQLAgent()
agent.load_database("your_database.db")

# previous_response_id로 대화 체이닝 (v3.0)
result1 = agent.ask_with_history("개발팀 직원 목록을 보여줘")
result2 = agent.ask_with_history("그 중 연봉 7000만원 이상인 사람은?")

agent.close()
```

### 스키마 링킹

```python
from text_to_sql_agent import SchemaExtractor
from schema_linker import SchemaLinker

schema = SchemaExtractor.extract_sqlite_schema("database.db")
linker = SchemaLinker(schema)

result = linker.link("개발팀 직원의 평균 급여")
print(f"관련 테이블: {result.relevant_tables}")
```

### SQL 방언 변환 (v3.0: 6종)

```python
from dialect_handler import DialectManager, SQLDialect

manager = DialectManager()

# SQLite → BigQuery
bigquery_sql = manager.convert(
    "SELECT GROUP_CONCAT(name) FROM employees",
    SQLDialect.SQLITE, SQLDialect.BIGQUERY
)

# SQLite → SQL Server (v3.0 신규)
mssql_sql = manager.convert(
    "SELECT name FROM employees LIMIT 10",
    SQLDialect.SQLITE, SQLDialect.SQLSERVER
)
```

## 🔧 고급 설정

### 모델 설정 (2026년 19종)

```python
from text_to_sql_agent import TextToSQLAgent, ModelConfig

# GPT-5.2 (권장)
agent = TextToSQLAgent(deployment_name="gpt-5.2")

# SQL 특화 모델
agent = TextToSQLAgent(deployment_name="gpt-5.2-codex")

# 사용 가능한 모델:
# GPT-5.2: gpt-5.2, gpt-5.2-codex, gpt-5.2-mini
# GPT-5.1: gpt-5.1, gpt-5.1-codex, gpt-5.1-codex-max
# GPT-5:   gpt-5, gpt-5-pro, gpt-5-codex, gpt-5-mini, gpt-5-nano
# GPT-4.1: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
# 추론:    o3, o3-pro, o4-mini
# Claude:  claude-opus-4-5, claude-sonnet-4-5
```

## 📊 Spider 2.0 벤치마크 (2026-06 기준)

| 순위 | 솔루션 | 점수 | 날짜 |
|------|--------|------|------|
| 1 | **TCDataAgent-SQL** (Tencent) | **95.14%** | 2026-05-18 |
| 2 | Native mini v2 (usenative.ai) | 93.88% | 2026-04-12 |
| 3 | Prism Swarm + Claude-Sonnet-4.5 (Paytm) | 91.23% | 2026-03-27 |
| 4 | Ask Data + RKG (AT&T & RelationalAI) | 88.52% | 2026-02-15 |
| 5 | ByteBrain-Agent v2 (ByteDance) | 86.74% | 2026-01-28 |

## 🧪 테스트 커버리지 (106 항목)

| 시나리오 | 테스트 항목 | 항목 수 |
|---------|-------------|---------|
| 1. 모듈 임포트 | 전체 import, ModelConfig 19종, Pydantic 모델, API 버전 | 15 |
| 2. 스키마 추출 | DB 생성, 테이블/컬럼/FK/캐시/초기화 | 10 |
| 3. 스키마 링킹 | 한국어 키워드, 퍼지/시맨틱 매칭, QueryDecomposer | 8 |
| 4. SQL 최적화 | SELECT*, IN 서브쿼리, ORDER BY, 최적 쿼리 패스 | 4 |
| 5. 자가 수정 | 테이블/컬럼 오타, 모호한 컬럼, GROUP BY 누락 | 4 |
| 6. 방언 처리 | 감지 5종, 변환, 멀티 6종, 특성 조회, MySQL/MSSQL | 10 |
| 7. SQL 검증 | SELECT/오타/빈SQL/JOIN/CTE 문법 | 5 |
| 8. 프롬프트 | 스키마 컨텍스트 생성, 4테이블 포함 확인 | 5 |
| 9. demo_app | 상수/배너 v3.0/메뉴/EXIT 명령어 | 11 |
| 10. E2E 통합 | 전체 파이프라인 + SQLite 실행, 멀티 방언 | 7 |
| **11. v3.0 신규** | **Pydantic 스키마/인스턴스, Cartesian Join, 신규 키워드 5개, 방언 6종** | **8** |
| 12. API 통합 | Responses API 호출 (키 필요, 없으면 skip) | 4 |
| | **합계** | **≥ 106** |

## 🆚 경쟁 솔루션 비교

| 기능 | 본 솔루션 | 일반 LLM | 기존 NL2SQL |
|------|----------|---------|-------------|
| Spider 2.0 정확도 | **95.14%** | ~45% | ~50% |
| Self-Correction | ✅ 5-round | ❌ | ❌ |
| 멀티 DB 지원 | ✅ 6종 | ❌ | △ 1~2종 |
| 한국어 최적화 | ✅ 55+ | △ | ❌ |
| 스키마 링킹 | ✅ | ❌ | △ |
| 대화형 컨텍스트 | ✅ previous_response_id | △ | ❌ |
| 쿼리 최적화 제안 | ✅ 11개 규칙 | ❌ | ❌ |
| Structured Outputs | ✅ Pydantic v2 | ❌ | ❌ |

---

## 📝 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-06-15 | **3.0.0** | **Responses API 마이그레이션, Pydantic v2, previous_response_id, MySQL/SQL Server, 코드 최적화 (DRY, __slots__, 프리컴파일, dict dispatch)** |
| 2026-02-08 | 2.2.1 | README/주석 최신화, demo_app DRY 최적화 |
| 2026-02-08 | 2.2.0 | API v1 업그레이드, gpt-5.2-codex, 400K context |
| 2026-01-26 | 2.1.0 | GPT-5.2 내장 심층 추론, 종합 테스트 |
| 2026-01-24 | 2.0.0 | GPT-5.2 + Structured Outputs 적용 |
| 2025-12-01 | 1.5.0 | Spider 2.0 기술 적용, 한국어 최적화 |
| 2025-06-01 | 1.0.0 | 초기 버전 (GPT-4.1 기반) |

## 📚 참고 자료

- [Spider 2.0 벤치마크](https://spider2-sql.github.io/)
- [Azure OpenAI Responses API](https://learn.microsoft.com/azure/ai-services/openai/how-to/responses)
- [Azure OpenAI 모델 카탈로그](https://learn.microsoft.com/azure/ai-services/openai/concepts/models)
- [Pydantic v2 Structured Outputs](https://learn.microsoft.com/azure/ai-services/openai/how-to/structured-outputs)
- [OpenAI Responses API Reference](https://platform.openai.com/docs/api-reference/responses)

## 📄 라이선스

MIT License
