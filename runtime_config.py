"""Runtime configuration helpers for Advanced Text-to-SQL.

2026-03 incremental modernization goals:
- Centralize env alias resolution
- Keep backwards compatibility with existing OPEN_AI_* aliases
- Expose a single settings object for API/demo/MCP layers
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, Optional


def _first_non_empty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def get_azure_openai_api_key() -> Optional[str]:
    return _first_non_empty(
        os.getenv("OPEN_AI_KEY_5"),
        os.getenv("AZURE_OPENAI_API_KEY"),
    )


def get_azure_openai_endpoint() -> Optional[str]:
    return _first_non_empty(
        os.getenv("OPEN_AI_ENDPOINT_5"),
        os.getenv("AZURE_OPENAI_ENDPOINT"),
    )


def get_deployment_name(default: str = "gpt-5.4") -> str:
    return _first_non_empty(
        os.getenv("MODEL_DEPLOYMENT_GPT5_4"),
        os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        default,
    ) or default


def get_api_version(default: str = "2025-04-01-preview") -> str:
    return _first_non_empty(
        os.getenv("AZURE_OPENAI_API_VERSION"),
        default,
    ) or default


def _get_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed >= minimum else default


# ── v3.2.0: Responses API 네이티브 파라미터 기본값 ────────────────────────

_VALID_REASONING_EFFORT = {"none", "minimal", "low", "medium", "high", "xhigh"}
_VALID_VERBOSITY = {"low", "medium", "high"}
_VALID_CACHE_RETENTION = {"in-memory", "24h"}


def _get_enum(name: str, default: str, allowed: set[str]) -> str:
    raw = (os.getenv(name) or "").strip().lower()
    return raw if raw in allowed else default


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    deployment_name: str
    api_version: str
    conversation_ttl_seconds: int
    default_max_rows: int
    max_result_window: int
    query_audit_limit: int
    cors_allow_origins: tuple[str, ...]
    azure_openai_configured: bool
    # v3.2.0 — Responses API native tuning knobs
    default_reasoning_effort: str           # none/minimal/low/medium/high/xhigh
    deep_reasoning_effort: str              # 복잡 질문용 effort
    verbosity: str                          # low/medium/high
    prompt_cache_retention: str             # in-memory/24h
    enable_execution_feedback: bool         # Self-Correction 실행 결과 피드백
    # v3.2.0 — 임베딩 기반 Schema Retriever
    enable_embedding_retrieval: bool        # 대규모 스키마에서 의미검색 보강
    embedding_deployment: str               # Azure OpenAI 임베딩 배포명
    embedding_top_k: int                    # 검색기 보강 상위 K
    embedding_min_score: float              # 코사인 유사도 하한

    def to_public_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["cors_allow_origins"] = list(self.cors_allow_origins)
        return data


@lru_cache(maxsize=1)
def get_runtime_settings() -> RuntimeSettings:
    cors_raw = _first_non_empty(os.getenv("TEXT2SQL_CORS_ORIGINS"), "*") or "*"
    cors_allow_origins = tuple(part.strip() for part in cors_raw.split(",") if part.strip()) or ("*",)
    enable_feedback_raw = (os.getenv("TEXT2SQL_ENABLE_EXECUTION_FEEDBACK") or "1").strip().lower()
    enable_feedback = enable_feedback_raw not in {"0", "false", "no", "off"}
    # v3.2.0 — 임베딩 검색기 플래그 (기본 비활성, 샘플 DB 환경 회귀 방지)
    enable_embed_raw = (os.getenv("TEXT2SQL_ENABLE_EMBEDDING_RETRIEVAL") or "0").strip().lower()
    enable_embed = enable_embed_raw in {"1", "true", "yes", "on"}
    embed_deployment = (os.getenv("TEXT2SQL_EMBEDDING_DEPLOYMENT") or "text-embedding-3-small").strip()
    try:
        embed_top_k = max(1, int(os.getenv("TEXT2SQL_EMBEDDING_TOP_K") or "5"))
    except ValueError:
        embed_top_k = 5
    try:
        embed_min_score = float(os.getenv("TEXT2SQL_EMBEDDING_MIN_SCORE") or "0.25")
    except ValueError:
        embed_min_score = 0.25
    return RuntimeSettings(
        deployment_name=get_deployment_name(),
        api_version=get_api_version(),
        conversation_ttl_seconds=_get_int("TEXT2SQL_CONVERSATION_TTL_SECONDS", 1800),
        default_max_rows=_get_int("TEXT2SQL_DEFAULT_MAX_ROWS", 100),
        max_result_window=_get_int("TEXT2SQL_MAX_RESULT_WINDOW", 1000),
        query_audit_limit=_get_int("TEXT2SQL_QUERY_AUDIT_LIMIT", 200),
        cors_allow_origins=cors_allow_origins,
        azure_openai_configured=bool(get_azure_openai_api_key() and get_azure_openai_endpoint()),
        default_reasoning_effort=_get_enum(
            "TEXT2SQL_REASONING_EFFORT", "low", _VALID_REASONING_EFFORT
        ),
        deep_reasoning_effort=_get_enum(
            "TEXT2SQL_DEEP_REASONING_EFFORT", "high", _VALID_REASONING_EFFORT
        ),
        verbosity=_get_enum(
            "TEXT2SQL_VERBOSITY", "low", _VALID_VERBOSITY
        ),
        prompt_cache_retention=_get_enum(
            "TEXT2SQL_PROMPT_CACHE_RETENTION", "24h", _VALID_CACHE_RETENTION
        ),
        enable_execution_feedback=enable_feedback,
        enable_embedding_retrieval=enable_embed,
        embedding_deployment=embed_deployment,
        embedding_top_k=embed_top_k,
        embedding_min_score=embed_min_score,
    )