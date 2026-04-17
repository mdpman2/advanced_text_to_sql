"""Embedding-based Schema Retriever (v3.2.0).

대규모 스키마(수십~수백 테이블)에서 자연어 질문과 의미적으로 유사한
테이블을 상위 K개 선별하여 LLM 프롬프트에 주입할 컨텍스트를 줄여줍니다.

동작 개요:
1. `index(schema)`: 각 테이블을 "테이블명 + 컬럼 + FK + 설명"으로 직렬화해 임베딩
2. `retrieve(question, top_k)`: 질문 임베딩과 코사인 유사도 계산 → 상위 K 테이블
3. 결과는 기존 `SchemaLinker` 키워드 매칭을 보강하는 신호로 사용

설계 원칙:
- numpy 없이도 pure Python으로 동작 (대규모일 때만 numpy 권장)
- Azure OpenAI Embedding API 미설정 시 조용히 비활성화 (raise 대신 is_ready=False)
- 임베딩 비용 절감을 위해 테이블 인덱스는 1회만 빌드 후 캐시 (스키마 해시 기반)
- 질문 임베딩은 LRU 캐시로 반복 호출 비용 절감

Author: Azure OpenAI Sample
Date: 2026-04-17
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from text_to_sql_agent import DatabaseSchema, TableSchema

logger = logging.getLogger(__name__)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """코사인 유사도 (numpy 없이 pure Python)"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / denom if denom else 0.0


@dataclass(slots=True)
class TableEmbedding:
    """테이블별 임베딩 항목"""
    table_name: str
    text: str                           # 임베딩 대상 텍스트
    vector: List[float] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalHit:
    """검색 결과 항목"""
    table_name: str
    score: float                        # 코사인 유사도 (-1.0 ~ 1.0)


class EmbeddingSchemaRetriever:
    """Azure OpenAI Embedding 기반 스키마 검색기.

    사용 예:
        retriever = EmbeddingSchemaRetriever(
            api_key=...,
            endpoint=...,
            deployment_name="text-embedding-3-small",
        )
        retriever.index(schema)
        hits = retriever.retrieve("지난달 VIP 고객 매출", top_k=3)
    """

    __slots__ = (
        "_client", "_deployment", "_api_version",
        "_embeddings", "_schema_fingerprint",
        "_question_cache", "_enabled",
    )

    # 기본 임베딩 배포명 (env로 오버라이드 가능)
    DEFAULT_DEPLOYMENT = "text-embedding-3-small"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: str = "2024-10-21",
    ) -> None:
        self._deployment = deployment_name or os.getenv(
            "TEXT2SQL_EMBEDDING_DEPLOYMENT", self.DEFAULT_DEPLOYMENT
        )
        self._api_version = api_version
        self._embeddings: List[TableEmbedding] = []
        self._schema_fingerprint: Optional[str] = None
        self._question_cache: Dict[str, List[float]] = {}
        self._enabled = False
        self._client = None

        resolved_key = api_key or os.getenv("OPEN_AI_KEY_5") or os.getenv("AZURE_OPENAI_API_KEY")
        resolved_endpoint = endpoint or os.getenv("OPEN_AI_ENDPOINT_5") or os.getenv("AZURE_OPENAI_ENDPOINT")

        if not (resolved_key and resolved_endpoint):
            logger.info("EmbeddingSchemaRetriever 비활성화: Azure OpenAI 자격증명 미설정")
            return

        try:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=resolved_key,
                azure_endpoint=resolved_endpoint,
                api_version=api_version,
            )
            self._enabled = True
        except Exception as exc:  # 패키지/네트워크 문제 — 조용히 비활성화
            logger.warning(f"EmbeddingSchemaRetriever 초기화 실패: {exc}")
            self._client = None
            self._enabled = False

    @property
    def is_ready(self) -> bool:
        """검색 준비 완료 여부 (자격증명 OK + 인덱싱 완료)"""
        return self._enabled and bool(self._embeddings)

    @property
    def deployment_name(self) -> str:
        return self._deployment

    @property
    def indexed_table_count(self) -> int:
        return len(self._embeddings)

    # ── 스키마 직렬화 ──

    @staticmethod
    def _serialize_table(table: "TableSchema") -> str:
        """테이블을 의미있는 임베딩 텍스트로 직렬화."""
        parts: List[str] = [f"Table: {table.name}"]
        if getattr(table, "description", None):
            parts.append(f"Description: {table.description}")
        col_parts = []
        for col in table.columns:
            col_parts.append(f"{col.get('name')}:{col.get('type', 'UNKNOWN')}")
        if col_parts:
            parts.append("Columns: " + ", ".join(col_parts))
        if table.primary_keys:
            parts.append("PrimaryKeys: " + ", ".join(table.primary_keys))
        if table.foreign_keys:
            fk_parts = [
                f"{fk.get('column')}->{fk.get('references_table')}.{fk.get('references_column')}"
                for fk in table.foreign_keys
            ]
            parts.append("ForeignKeys: " + ", ".join(fk_parts))
        return " | ".join(parts)

    @staticmethod
    def _schema_hash(texts: List[str], deployment: str) -> str:
        """테이블 직렬화 + 배포명 기반 해시 (인덱스 무효화 감지용)"""
        digest = hashlib.sha256()
        digest.update(deployment.encode("utf-8"))
        for text in texts:
            digest.update(b"\x1f")
            digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    # ── 임베딩 호출 ──

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 한 번에 임베딩."""
        if not self._enabled or not self._client or not texts:
            return []
        try:
            response = self._client.embeddings.create(
                model=self._deployment,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.warning(f"임베딩 호출 실패: {exc}")
            return []

    def _embed_single(self, text: str) -> List[float]:
        """단일 텍스트 임베딩 (LRU 캐시)."""
        cached = self._question_cache.get(text)
        if cached is not None:
            return cached
        vectors = self._embed_batch([text])
        vec = vectors[0] if vectors else []
        if vec:
            # 캐시 크기 제한 (200개 초과 시 FIFO 제거)
            if len(self._question_cache) >= 200:
                first_key = next(iter(self._question_cache))
                self._question_cache.pop(first_key, None)
            self._question_cache[text] = vec
        return vec

    # ── 인덱싱 ──

    def index(self, schema: "DatabaseSchema", force: bool = False) -> bool:
        """스키마를 인덱싱.

        Args:
            schema: DatabaseSchema
            force: True이면 기존 인덱스 캐시와 무관하게 재빌드
        Returns:
            인덱스가 실제로 빌드되었으면 True
        """
        if not self._enabled:
            return False

        texts = [self._serialize_table(t) for t in schema.tables]
        fingerprint = self._schema_hash(texts, self._deployment)
        if not force and fingerprint == self._schema_fingerprint and self._embeddings:
            return True  # 이미 동일 스키마 인덱싱됨

        vectors = self._embed_batch(texts)
        if len(vectors) != len(texts):
            logger.warning(
                f"임베딩 개수 불일치: 요청 {len(texts)} vs 응답 {len(vectors)} — 인덱싱 실패"
            )
            return False

        self._embeddings = [
            TableEmbedding(table_name=schema.tables[i].name, text=texts[i], vector=vectors[i])
            for i in range(len(texts))
        ]
        self._schema_fingerprint = fingerprint
        self._question_cache.clear()  # 스키마 변경 시 질문 캐시 무효화
        logger.info(
            f"EmbeddingSchemaRetriever 인덱스 빌드 완료: {len(self._embeddings)} 테이블, "
            f"deployment={self._deployment}"
        )
        return True

    # ── 검색 ──

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.25,
    ) -> List[RetrievalHit]:
        """질문과 의미적으로 유사한 테이블을 상위 top_k개 반환.

        Args:
            question: 자연어 질문
            top_k: 최대 반환 개수
            min_score: 유사도 하한 (미만이면 제외)
        """
        if not self.is_ready or not question.strip():
            return []
        q_vec = self._embed_single(question)
        if not q_vec:
            return []

        scored: List[Tuple[str, float]] = []
        for item in self._embeddings:
            score = _cosine(q_vec, item.vector)
            if score >= min_score:
                scored.append((item.table_name, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [RetrievalHit(table_name=name, score=score) for name, score in scored[:top_k]]

    def clear(self) -> None:
        """인덱스 및 캐시 초기화."""
        self._embeddings.clear()
        self._schema_fingerprint = None
        self._question_cache.clear()


__all__ = ["EmbeddingSchemaRetriever", "RetrievalHit", "TableEmbedding"]
