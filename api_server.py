"""
REST API Server for Advanced Text-to-SQL Agent (v3.1.3)

QueryWeaver 참조 FastAPI 기반 REST API 서버.
SSE(Server-Sent Events) 스트리밍, 파괴적 SQL 확인, 모호한 질문 후속 질문 기능 포함.

엔드포인트:
- GET  /databases              → 로드된 데이터베이스 목록
- GET  /databases/{db_id}/schema → 스키마 정보 (그래프 형태 포함)
- POST /databases               → 데이터베이스 업로드/연결
- POST /databases/{db_id}/query  → Text-to-SQL 질의 (스트리밍)
- POST /databases/{db_id}/query/sync → Text-to-SQL 질의 (동기)
- GET  /databases/{db_id}/graph  → 스키마 그래프 시각화 데이터
- POST /confirm/{confirmation_id} → 파괴적 SQL 실행 확인
- GET  /health                  → 헬스 체크

v3.1.3 변경:
- 동기 모드: sqlite_sql (실행용) / response_sql (응답용) 변수 분리 (방언 변환 SQL 실행 방지)
- 스트리밍 모드: dialect 파라미터 추가, display_sql / sqlite_sql 분리
- 스트리밍 호출부에 dialect=request.dialect 전달

실행:
    uvicorn api_server:app --host 0.0.0.0 --port 5000 --reload

Author: Azure OpenAI Sample
Date: 2026-06-15
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from text_to_sql_agent import (
    TextToSQLAgent,
    ConversationalSQLAgent,
    SchemaExtractor,
    DatabaseSchema,
    DatabaseType,
    SQLValidator,
    create_sample_database,
)
from schema_linker import SchemaLinker
from sql_optimizer import SQLOptimizer
from dialect_handler import SQLDialect, MultiDatabaseQuery
from query_guard import QueryGuard, ConfirmationStore
from ambiguity_detector import AmbiguityDetector
from schema_graph import SchemaGraphBuilder

logger = logging.getLogger(__name__)

# ── 글로벌 상태 ──

_databases: Dict[str, Dict[str, Any]] = {}      # db_id → {path, schema, agent, linker}
_conversations: Dict[str, ConversationalSQLAgent] = {}  # session_id → agent
_confirmation_store = ConfirmationStore()


# ── Pydantic 요청/응답 모델 ──

class QueryRequest(BaseModel):
    """질의 요청"""
    question: str = Field(..., description="자연어 질문")
    session_id: Optional[str] = Field(None, description="대화 세션 ID (멀티턴용)")
    instructions: Optional[str] = Field(None, description="추가 지시사항 (방언 등)")
    dialect: str = Field("sqlite", description="SQL 방언 (sqlite, postgresql, mysql, bigquery, snowflake, sqlserver)")
    execute: bool = Field(True, description="SQL 실행 여부")
    stream: bool = Field(False, description="스트리밍 응답 여부")


class QueryResponse(BaseModel):
    """질의 응답"""
    question: str
    sql: str
    explanation: str
    confidence: float
    results: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = None
    requires_confirmation: bool = False
    confirmation_id: Optional[str] = None
    confirmation_message: Optional[str] = None
    follow_up_questions: Optional[List[str]] = None
    ambiguity_detected: bool = False
    ambiguity_reason: Optional[str] = None


class ConfirmRequest(BaseModel):
    """파괴적 SQL 실행 확인 요청"""
    confirmed: bool = Field(..., description="실행 확인 여부")


class DatabaseUploadRequest(BaseModel):
    """데이터베이스 업로드 요청"""
    database_name: str = Field(..., description="데이터베이스 이름")
    db_path: Optional[str] = Field(None, description="로컬 DB 파일 경로")


class SchemaGraphResponse(BaseModel):
    """스키마 그래프 응답"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# ── 스트리밍 헬퍼 ──

STREAM_BOUNDARY = "|||TEXT2SQL_BOUNDARY|||"


async def _stream_sql_generation(
    agent: TextToSQLAgent,
    question: str,
    linker: SchemaLinker,
    optimizer: SQLOptimizer,
    guard: QueryGuard,
    ambiguity_detector: AmbiguityDetector,
    schema: DatabaseSchema,
    execute: bool = True,
    dialect: str = "sqlite",
) -> AsyncGenerator[str, None]:
    """SSE 스트리밍으로 SQL 생성 과정을 단계별 전송"""

    # Step 1: 모호성 감지
    yield json.dumps({
        "step": "ambiguity_check",
        "message": "질문 분석 중...",
        "status": "in_progress"
    }, ensure_ascii=False) + STREAM_BOUNDARY

    ambiguity = ambiguity_detector.detect(question, schema)
    if ambiguity["is_ambiguous"]:
        yield json.dumps({
            "step": "ambiguity_detected",
            "message": ambiguity["reason"],
            "follow_up_questions": ambiguity["suggestions"],
            "status": "needs_clarification"
        }, ensure_ascii=False) + STREAM_BOUNDARY

    # Step 2: 스키마 링킹
    yield json.dumps({
        "step": "schema_linking",
        "message": "관련 테이블/컬럼 식별 중...",
        "status": "in_progress"
    }, ensure_ascii=False) + STREAM_BOUNDARY

    link_result = linker.link(question)
    yield json.dumps({
        "step": "schema_linking",
        "message": f"관련 테이블: {', '.join(link_result.relevant_tables)}",
        "tables": list(link_result.relevant_tables),
        "status": "completed"
    }, ensure_ascii=False) + STREAM_BOUNDARY

    await asyncio.sleep(0.1)  # 스트리밍 간격

    # Step 3: SQL 생성
    yield json.dumps({
        "step": "sql_generation",
        "message": "SQL 생성 중...",
        "status": "in_progress"
    }, ensure_ascii=False) + STREAM_BOUNDARY

    try:
        result = agent.ask(question, execute=False)
        sqlite_sql = result["sql"]  # 원본 SQLite SQL (실행용)
        display_sql = sqlite_sql      # 응답용 SQL

        # 방언 변환 (v3.1.3: 스트리밍에서도 방언 변환 지원)
        if dialect and dialect.lower() != "sqlite":
            from dialect_handler import DialectManager as _DM, SQLDialect as _SD
            _dialect_map = {d.value: d for d in _SD}
            _target = _dialect_map.get(dialect.lower())
            if _target:
                display_sql = _DM().convert(sqlite_sql, _SD.SQLITE, _target)

        sql = sqlite_sql  # 실행에는 원본 사용

        yield json.dumps({
            "step": "sql_generation",
            "sql": display_sql,
            "explanation": result["explanation"],
            "confidence": result["confidence"],
            "status": "completed"
        }, ensure_ascii=False) + STREAM_BOUNDARY

    except Exception as e:
        yield json.dumps({
            "step": "sql_generation",
            "message": f"SQL 생성 실패: {str(e)}",
            "status": "error"
        }, ensure_ascii=False) + STREAM_BOUNDARY
        return

    # Step 4: 최적화 분석
    yield json.dumps({
        "step": "optimization",
        "message": "쿼리 최적화 분석 중...",
        "status": "in_progress"
    }, ensure_ascii=False) + STREAM_BOUNDARY

    opt_result = optimizer.optimize(sql)
    if opt_result.optimizations_applied:
        yield json.dumps({
            "step": "optimization",
            "suggestions": opt_result.optimizations_applied,
            "status": "completed"
        }, ensure_ascii=False) + STREAM_BOUNDARY

    # Step 5: 파괴적 SQL 확인
    if guard.is_destructive(sql):
        confirmation_id = _confirmation_store.store(sql, agent)
        yield json.dumps({
            "step": "confirmation_required",
            "message": guard.get_warning_message(sql),
            "confirmation_id": confirmation_id,
            "sql": sql,
            "status": "awaiting_confirmation"
        }, ensure_ascii=False) + STREAM_BOUNDARY
        return

    # Step 6: SQL 실행
    if execute and agent.db_connection:
        yield json.dumps({
            "step": "execution",
            "message": "SQL 실행 중...",
            "status": "in_progress"
        }, ensure_ascii=False) + STREAM_BOUNDARY

        try:
            columns, rows = agent.execute_query(sql)
            results = [dict(zip(columns, row)) for row in rows]
            yield json.dumps({
                "step": "execution",
                "columns": columns,
                "results": results[:100],  # 최대 100행
                "row_count": len(rows),
                "status": "completed"
            }, ensure_ascii=False) + STREAM_BOUNDARY
        except Exception as e:
            yield json.dumps({
                "step": "execution",
                "message": f"실행 오류: {str(e)}",
                "status": "error"
            }, ensure_ascii=False) + STREAM_BOUNDARY

    # 최종 완료
    yield json.dumps({
        "step": "done",
        "status": "completed"
    }, ensure_ascii=False) + STREAM_BOUNDARY


# ── FastAPI 앱 ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 수명주기 관리"""
    # 샘플 DB 자동 로드
    db_path = create_sample_database()
    _register_database("sample_company", db_path)
    logger.info("샘플 데이터베이스 로드 완료")
    yield
    # 정리
    for db_info in _databases.values():
        agent = db_info.get("agent")
        if agent:
            agent.close()


app = FastAPI(
    title="Advanced Text-to-SQL API",
    description="Spider 2.0 #1 기술 기반 Text-to-SQL REST API (QueryWeaver 참조)",
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _register_database(db_id: str, db_path: str) -> Dict[str, Any]:
    """데이터베이스 등록"""
    schema = SchemaExtractor.extract_sqlite_schema(db_path)
    linker = SchemaLinker(schema)

    # 에이전트는 API 키가 있을 때만 생성
    agent = None
    api_key = os.getenv("OPEN_AI_KEY_5") or os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        agent = TextToSQLAgent(deployment_name="gpt-5.2")
        agent.load_database(db_path)

    _databases[db_id] = {
        "path": db_path,
        "schema": schema,
        "linker": linker,
        "optimizer": SQLOptimizer(),
        "guard": QueryGuard(),
        "ambiguity_detector": AmbiguityDetector(schema),
        "agent": agent,
    }
    return _databases[db_id]


def _get_db(db_id: str) -> Dict[str, Any]:
    """데이터베이스 정보 조회"""
    if db_id not in _databases:
        raise HTTPException(status_code=404, detail=f"데이터베이스 '{db_id}'를 찾을 수 없습니다.")
    return _databases[db_id]


# ── 엔드포인트 ──

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "ok", "version": "3.1.0", "databases": len(_databases)}


@app.get("/databases")
async def list_databases():
    """로드된 데이터베이스 목록"""
    result = []
    for db_id, info in _databases.items():
        schema: DatabaseSchema = info["schema"]
        result.append({
            "id": db_id,
            "name": schema.database_name,
            "type": schema.database_type.value,
            "tables": len(schema.tables),
            "table_names": [t.name for t in schema.tables],
        })
    return {"databases": result}


@app.post("/databases")
async def upload_database(
    file: Optional[UploadFile] = File(None),
    database_name: Optional[str] = Query(None),
    db_path: Optional[str] = Query(None),
):
    """데이터베이스 업로드 또는 경로 지정"""
    if file:
        # 파일 업로드
        upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, file.filename)
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        db_id = database_name or file.filename.replace(".", "_")
        _register_database(db_id, save_path)
    elif db_path:
        if not os.path.exists(db_path):
            raise HTTPException(status_code=400, detail=f"파일이 존재하지 않습니다: {db_path}")
        db_id = database_name or os.path.basename(db_path).replace(".", "_")
        _register_database(db_id, db_path)
    else:
        raise HTTPException(status_code=400, detail="file 또는 db_path를 제공해주세요.")

    return {"message": f"데이터베이스 '{db_id}' 등록 완료", "id": db_id}


@app.get("/databases/{db_id}/schema")
async def get_schema(db_id: str):
    """데이터베이스 스키마 정보"""
    db_info = _get_db(db_id)
    schema: DatabaseSchema = db_info["schema"]

    tables = []
    for table in schema.tables:
        tables.append({
            "name": table.name,
            "columns": table.columns,
            "primary_keys": table.primary_keys,
            "foreign_keys": table.foreign_keys,
            "sample_data": table.sample_data,
        })

    return {
        "database_name": schema.database_name,
        "database_type": schema.database_type.value,
        "tables": tables,
    }


@app.get("/databases/{db_id}/graph")
async def get_schema_graph(db_id: str):
    """스키마 그래프 시각화 데이터 (nodes + edges)"""
    db_info = _get_db(db_id)
    schema: DatabaseSchema = db_info["schema"]
    graph = SchemaGraphBuilder.build(schema)
    return graph


@app.post("/databases/{db_id}/query", response_model=None)
async def query_database(db_id: str, request: QueryRequest):
    """
    Text-to-SQL 질의

    stream=True → SSE 스트리밍 응답
    stream=False → 동기 JSON 응답
    """
    db_info = _get_db(db_id)
    agent: Optional[TextToSQLAgent] = db_info.get("agent")

    if not agent:
        raise HTTPException(
            status_code=503,
            detail="API 키가 설정되지 않았습니다. OPEN_AI_KEY_5 또는 AZURE_OPENAI_API_KEY를 설정하세요."
        )

    # 스트리밍 모드
    if request.stream:
        return StreamingResponse(
            _stream_sql_generation(
                agent=agent,
                question=request.question,
                linker=db_info["linker"],
                optimizer=db_info["optimizer"],
                guard=db_info["guard"],
                ambiguity_detector=db_info["ambiguity_detector"],
                schema=db_info["schema"],
                execute=request.execute,
                dialect=request.dialect,
            ),
            media_type="text/event-stream",
        )

    # 동기 모드
    linker: SchemaLinker = db_info["linker"]
    guard: QueryGuard = db_info["guard"]
    ambiguity_detector: AmbiguityDetector = db_info["ambiguity_detector"]
    optimizer: SQLOptimizer = db_info["optimizer"]

    # 모호성 감지
    ambiguity = ambiguity_detector.detect(request.question, db_info["schema"])

    # SQL 생성 (instructions를 추가 컨텍스트로 전달)
    try:
        result = agent.ask(request.question, execute=False)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=f"SQL 생성 실패: {str(e)}")
    sqlite_sql = result["sql"]  # 원본 SQLite SQL (실행용)
    response_sql = sqlite_sql   # 응답용 SQL (방언 변환 가능)

    # 방언 변환 (dialect가 sqlite가 아닌 경우 — 응답용만 변환)
    if request.dialect and request.dialect.lower() != "sqlite":
        from dialect_handler import DialectManager, SQLDialect
        dialect_map = {d.value: d for d in SQLDialect}
        target = dialect_map.get(request.dialect.lower())
        if target:
            mgr = DialectManager()
            response_sql = mgr.convert(sqlite_sql, SQLDialect.SQLITE, target)

    response = QueryResponse(
        question=request.question,
        sql=response_sql,
        explanation=result["explanation"],
        confidence=result["confidence"],
        ambiguity_detected=ambiguity["is_ambiguous"],
        ambiguity_reason=ambiguity.get("reason"),
        follow_up_questions=ambiguity.get("suggestions"),
    )

    # 파괴적 SQL 확인 (원본 SQL로 판단)
    if guard.is_destructive(sqlite_sql):
        confirmation_id = _confirmation_store.store(sqlite_sql, agent)
        response.requires_confirmation = True
        response.confirmation_id = confirmation_id
        response.confirmation_message = guard.get_warning_message(sqlite_sql)
        return response

    # SQL 실행 (원본 SQLite SQL로 실행)
    if request.execute and agent.db_connection:
        try:
            columns, rows = agent.execute_query(sqlite_sql)
            response.results = [dict(zip(columns, row)) for row in rows[:100]]
            response.row_count = len(rows)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SQL 실행 오류: {str(e)}")

    return response


@app.post("/confirm/{confirmation_id}")
async def confirm_destructive_sql(confirmation_id: str, request: ConfirmRequest):
    """파괴적 SQL 실행 확인"""
    pending = _confirmation_store.get(confirmation_id)
    if not pending:
        raise HTTPException(status_code=404, detail="확인 요청을 찾을 수 없거나 만료되었습니다.")

    if not request.confirmed:
        _confirmation_store.remove(confirmation_id)
        return {"message": "실행이 취소되었습니다.", "confirmed": False}

    sql = pending["sql"]
    agent: TextToSQLAgent = pending["agent"]

    try:
        columns, rows = agent.execute_query(sql, force=True)
        _confirmation_store.remove(confirmation_id)
        return {
            "message": "SQL 실행 완료",
            "confirmed": True,
            "sql": sql,
            "columns": columns,
            "results": [dict(zip(columns, row)) for row in rows[:100]],
            "row_count": len(rows),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL 실행 오류: {str(e)}")


# ── 실행 ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=5000, reload=True)
