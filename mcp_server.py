"""
MCP (Model Context Protocol) Server for Text-to-SQL Agent (v3.1.4)

QueryWeaver 참조 MCP 서버 구현.
AI 에이전트가 데이터베이스를 탐색하고 자연어로 질의할 수 있는 표준 MCP 인터페이스.

MCP Operations:
- list_databases     → 사용 가능한 데이터베이스 목록
- connect_database   → 데이터베이스 연결
- database_schema    → 데이터베이스 스키마 조회
- query_database     → 자연어 Text-to-SQL 질의 (v3.1.3: dialect 파라미터 실제 적용)
- disconnect_database → 데이터베이스 연결 해제

v3.1.4 변경:
- 서버 메타데이터/버전 표기를 REST API 및 README와 일치하도록 정리

Usage:
    # FastAPI MCP 엔드포인트로 통합 (/mcp)
    # 또는 독립 실행:
    python mcp_server.py

mcp.json client 설정 예시:
    {
        "servers": {
            "text2sql": {
                "type": "http",
                "url": "http://127.0.0.1:5000/mcp",
                "headers": {
                    "Authorization": "Bearer your_token_here"
                }
            }
        }
    }

Author: Azure OpenAI Sample
Date: 2026-03-17
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from text_to_sql_agent import (
    TextToSQLAgent,
    SchemaExtractor,
    DatabaseSchema,
    DatabaseType,
    create_sample_database,
)
from schema_linker import SchemaLinker
from schema_graph import SchemaGraphBuilder

logger = logging.getLogger(__name__)


# ── MCP 요청/응답 모델 ──

class MCPRequest(BaseModel):
    """MCP 표준 요청"""
    method: str = Field(..., description="MCP 메서드명")
    params: Dict[str, Any] = Field(default_factory=dict, description="메서드 파라미터")
    id: Optional[str] = Field(None, description="요청 ID")


class MCPResponse(BaseModel):
    """MCP 표준 응답"""
    id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPToolDefinition(BaseModel):
    """MCP 도구 정의"""
    name: str
    description: str
    parameters: Dict[str, Any]


# ── MCP Server ──

class MCPTextToSQLServer:
    """
    MCP 서버 — Text-to-SQL 기능을 MCP 프로토콜로 노출

    QueryWeaver의 MCP 서버 패턴을 참조하여 구현.
    AI 에이전트가 데이터베이스를 탐색하고 질의할 수 있습니다.
    """

    def __init__(self):
        self._databases: Dict[str, Dict[str, Any]] = {}
        self._tools = self._register_tools()

    def _register_tools(self) -> Dict[str, MCPToolDefinition]:
        """MCP 도구 등록"""
        return {
            "list_databases": MCPToolDefinition(
                name="list_databases",
                description="사용 가능한 데이터베이스 목록을 반환합니다.",
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            "connect_database": MCPToolDefinition(
                name="connect_database",
                description="데이터베이스에 연결합니다. SQLite 파일 경로를 지정하세요.",
                parameters={
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string", "description": "SQLite DB 파일 경로"},
                        "db_name": {"type": "string", "description": "데이터베이스 식별 이름 (선택)"},
                    },
                    "required": ["db_path"],
                },
            ),
            "database_schema": MCPToolDefinition(
                name="database_schema",
                description="데이터베이스의 스키마(테이블, 컬럼, 외래키, 관계)를 반환합니다.",
                parameters={
                    "type": "object",
                    "properties": {
                        "db_name": {"type": "string", "description": "데이터베이스 이름"},
                        "include_graph": {"type": "boolean", "description": "그래프 형태 포함 여부 (기본: false)"},
                        "include_samples": {"type": "boolean", "description": "샘플 데이터 포함 여부 (기본: true)"},
                    },
                    "required": ["db_name"],
                },
            ),
            "query_database": MCPToolDefinition(
                name="query_database",
                description="자연어 질문을 SQL로 변환하고 실행합니다. Text-to-SQL 엔진 사용.",
                parameters={
                    "type": "object",
                    "properties": {
                        "db_name": {"type": "string", "description": "데이터베이스 이름"},
                        "question": {"type": "string", "description": "자연어 질문"},
                        "execute": {"type": "boolean", "description": "SQL 실행 여부 (기본: true)"},
                        "dialect": {"type": "string", "description": "SQL 방언 (기본: sqlite)"},
                    },
                    "required": ["db_name", "question"],
                },
            ),
            "disconnect_database": MCPToolDefinition(
                name="disconnect_database",
                description="데이터베이스 연결을 해제합니다.",
                parameters={
                    "type": "object",
                    "properties": {
                        "db_name": {"type": "string", "description": "데이터베이스 이름"},
                    },
                    "required": ["db_name"],
                },
            ),
        }

    # ── MCP 프로토콜 메서드 ──

    def handle_request(self, request: MCPRequest) -> MCPResponse:
        """MCP 요청 라우팅"""
        method = request.method
        params = request.params

        # 표준 MCP 메서드
        if method == "initialize":
            return self._handle_initialize(request.id)
        elif method == "tools/list":
            return self._handle_tools_list(request.id)
        elif method == "tools/call":
            return self._handle_tools_call(params, request.id)
        else:
            return MCPResponse(
                id=request.id,
                error={"code": -32601, "message": f"Unknown method: {method}"},
            )

    def _handle_initialize(self, request_id: Optional[str]) -> MCPResponse:
        """MCP 초기화"""
        return MCPResponse(
            id=request_id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                },
                "serverInfo": {
                    "name": "text2sql-mcp-server",
                    "version": "3.1.4",
                    "description": "Advanced Text-to-SQL Agent — Spider 2.0 #1 기술 기반",
                },
            },
        )

    def _handle_tools_list(self, request_id: Optional[str]) -> MCPResponse:
        """도구 목록 반환"""
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            }
            for tool in self._tools.values()
        ]
        return MCPResponse(id=request_id, result={"tools": tools})

    def _handle_tools_call(self, params: Dict[str, Any], request_id: Optional[str]) -> MCPResponse:
        """도구 호출 처리"""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler_map = {
            "list_databases": self._tool_list_databases,
            "connect_database": self._tool_connect_database,
            "database_schema": self._tool_database_schema,
            "query_database": self._tool_query_database,
            "disconnect_database": self._tool_disconnect_database,
        }

        handler = handler_map.get(tool_name)
        if not handler:
            return MCPResponse(
                id=request_id,
                error={"code": -32602, "message": f"Unknown tool: {tool_name}"},
            )

        try:
            result = handler(arguments)
            return MCPResponse(
                id=request_id,
                result={"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]},
            )
        except Exception as e:
            logger.error(f"MCP tool error ({tool_name}): {e}")
            return MCPResponse(
                id=request_id,
                error={"code": -32000, "message": str(e)},
            )

    # ── 도구 구현 ──

    def _tool_list_databases(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """데이터베이스 목록"""
        databases = []
        for name, info in self._databases.items():
            schema: DatabaseSchema = info["schema"]
            databases.append({
                "name": name,
                "type": schema.database_type.value,
                "tables": len(schema.tables),
                "table_names": [t.name for t in schema.tables],
            })
        return {"databases": databases, "count": len(databases)}

    def _tool_connect_database(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """데이터베이스 연결"""
        db_path = args["db_path"]
        db_name = args.get("db_name", os.path.basename(db_path).replace(".", "_"))

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {db_path}")

        schema = SchemaExtractor.extract_sqlite_schema(db_path)
        linker = SchemaLinker(schema)

        # 에이전트 생성 (API 키 필요)
        agent = None
        api_key = os.getenv("OPEN_AI_KEY_5") or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key:
            agent = TextToSQLAgent(deployment_name="gpt-5.4")
            agent.load_database(db_path)

        self._databases[db_name] = {
            "path": db_path,
            "schema": schema,
            "linker": linker,
            "agent": agent,
        }

        return {
            "message": f"데이터베이스 '{db_name}' 연결 완료",
            "name": db_name,
            "tables": len(schema.tables),
            "table_names": [t.name for t in schema.tables],
            "has_agent": agent is not None,
        }

    def _tool_database_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """스키마 조회"""
        db_name = args["db_name"]
        include_graph = args.get("include_graph", False)
        include_samples = args.get("include_samples", True)

        if db_name not in self._databases:
            raise ValueError(f"데이터베이스 '{db_name}'을(를) 찾을 수 없습니다. 먼저 connect_database를 호출하세요.")

        schema: DatabaseSchema = self._databases[db_name]["schema"]

        tables = []
        for table in schema.tables:
            t = {
                "name": table.name,
                "columns": table.columns,
                "primary_keys": table.primary_keys,
                "foreign_keys": table.foreign_keys,
            }
            if include_samples and table.sample_data:
                t["sample_data"] = table.sample_data
            tables.append(t)

        result: Dict[str, Any] = {
            "database_name": schema.database_name,
            "database_type": schema.database_type.value,
            "tables": tables,
        }

        if include_graph:
            result["graph"] = SchemaGraphBuilder.build(schema)

        return result

    def _tool_query_database(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """자연어 질의"""
        db_name = args["db_name"]
        question = args["question"]
        execute = args.get("execute", True)
        dialect = args.get("dialect", "sqlite")

        if db_name not in self._databases:
            raise ValueError(f"데이터베이스 '{db_name}'을(를) 찾을 수 없습니다.")

        agent: Optional[TextToSQLAgent] = self._databases[db_name].get("agent")
        if not agent:
            raise RuntimeError("API 키가 설정되지 않아 SQL 생성을 수행할 수 없습니다.")

        result = agent.ask(question, execute=execute)

        # 방언 변환 (v3.1.3: MCP에서도 dialect 파라미터 반영)
        if dialect and dialect.lower() != "sqlite" and "sql" in result:
            from dialect_handler import DialectManager, SQLDialect
            dialect_map = {d.value: d for d in SQLDialect}
            target = dialect_map.get(dialect.lower())
            if target:
                result["sql"] = DialectManager().convert(
                    result["sql"], SQLDialect.SQLITE, target
                )

        return result

    def _tool_disconnect_database(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """데이터베이스 연결 해제"""
        db_name = args["db_name"]

        if db_name not in self._databases:
            raise ValueError(f"데이터베이스 '{db_name}'을(를) 찾을 수 없습니다.")

        agent = self._databases[db_name].get("agent")
        if agent:
            agent.close()

        del self._databases[db_name]
        return {"message": f"데이터베이스 '{db_name}' 연결 해제 완료"}


# ── FastAPI MCP 라우터 (api_server에서 import) ──

def create_mcp_router():
    """FastAPI 라우터에 MCP 엔드포인트 등록"""
    from fastapi import APIRouter

    router = APIRouter(prefix="/mcp", tags=["MCP"])
    mcp_server = MCPTextToSQLServer()

    # 샘플 DB 자동 로드
    sample_db_path = create_sample_database()
    mcp_server._tool_connect_database({"db_path": sample_db_path, "db_name": "sample_company"})

    @router.post("")
    @router.post("/")
    async def mcp_endpoint(request: MCPRequest):
        """MCP 표준 엔드포인트"""
        return mcp_server.handle_request(request)

    @router.get("/tools")
    async def list_tools():
        """MCP 도구 목록 (편의용)"""
        return {
            "tools": [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in mcp_server._tools.values()
            ]
        }

    return router


# ── 독립 실행 ──

if __name__ == "__main__":
    """MCP 서버 독립 실행"""
    import uvicorn
    from fastapi import FastAPI as FA

    mcp_app = FA(title="Text-to-SQL MCP Server", version="3.1.4")
    mcp_router = create_mcp_router()
    mcp_app.include_router(mcp_router)

    print("=" * 60)
    print("  Text-to-SQL MCP Server v3.1.4")
    print("  Endpoint: http://localhost:5001/mcp")
    print("=" * 60)

    uvicorn.run(mcp_app, host="0.0.0.0", port=5001)
