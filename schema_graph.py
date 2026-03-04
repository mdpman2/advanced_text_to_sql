"""
Schema Graph Builder — 스키마 그래프 시각화 데이터 생성 모듈 (v3.1.0)

QueryWeaver 참조: 데이터베이스 스키마를 그래프(nodes + edges) 형태로 변환하여
프론트엔드에서 시각적으로 테이블 관계를 표시할 수 있게 합니다.

기능:
- 테이블 → Node 변환 (타입: table)
- 컬럼 → Node 변환 (타입: column, PK/FK 표시)
- 외래키 관계 → Edge 변환 (관계 방향 포함)
- 테이블-컬럼 소속 관계 → Edge
- Mermaid ER 다이어그램 자동 생성
- NetworkX 호환 인접 리스트 (adjacency) 출력

출력 형식:
    {
        "nodes": [
            {"id": "employees", "type": "table", "label": "employees", "columns": [...]},
            {"id": "employees.emp_id", "type": "column", "label": "emp_id", ...},
        ],
        "edges": [
            {"source": "employees.dept_id", "target": "departments.dept_id",
             "type": "foreign_key", "label": "FK"},
        ],
        "metadata": {"table_count": 4, "edge_count": 3, ...}
    }

Author: Azure OpenAI Sample
Date: 2026-06-15
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from text_to_sql_agent import DatabaseSchema, TableSchema


class SchemaGraphBuilder:
    """
    스키마를 그래프(nodes + edges) 형태로 변환

    QueryWeaver의 /graphs/{graph_id}/data 엔드포인트 참조.
    프론트엔드에서 D3.js, vis.js, Mermaid 등으로 시각화 가능.
    """

    @staticmethod
    def build(schema: 'DatabaseSchema') -> Dict[str, Any]:
        """
        스키마를 그래프 데이터로 변환

        Args:
            schema: DatabaseSchema 객체

        Returns:
            {"nodes": [...], "edges": [...], "metadata": {...}}
        """
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        fk_count = 0

        for table in schema.tables:
            # 테이블 노드
            table_node = {
                "id": table.name,
                "type": "table",
                "label": table.name,
                "columns": [col["name"] for col in table.columns],
                "primary_keys": table.primary_keys,
                "column_count": len(table.columns),
            }
            nodes.append(table_node)

            # 컬럼 노드
            for col in table.columns:
                col_id = f"{table.name}.{col['name']}"
                is_pk = col["name"] in table.primary_keys
                is_fk = any(
                    fk["column"] == col["name"] for fk in table.foreign_keys
                )

                col_node = {
                    "id": col_id,
                    "type": "column",
                    "label": col["name"],
                    "table": table.name,
                    "data_type": col.get("type", "UNKNOWN"),
                    "nullable": col.get("nullable", True),
                    "is_primary_key": is_pk,
                    "is_foreign_key": is_fk,
                }
                nodes.append(col_node)

                # 테이블-컬럼 소속 에지
                edges.append({
                    "source": table.name,
                    "target": col_id,
                    "type": "has_column",
                    "label": "HAS",
                })

            # 외래키 에지
            for fk in table.foreign_keys:
                source_id = f"{table.name}.{fk['column']}"
                target_id = f"{fk['references_table']}.{fk['references_column']}"
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "type": "foreign_key",
                    "label": "FK",
                    "source_table": table.name,
                    "target_table": fk["references_table"],
                })
                fk_count += 1

        # 메타데이터
        metadata = {
            "database_name": schema.database_name,
            "database_type": schema.database_type.value,
            "table_count": len(schema.tables),
            "total_columns": sum(len(t.columns) for t in schema.tables),
            "foreign_key_count": fk_count,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata,
        }

    @staticmethod
    def build_table_only(schema: 'DatabaseSchema') -> Dict[str, Any]:
        """
        테이블 수준 그래프만 생성 (간소화 — 컬럼 노드 제외)

        테이블 간 FK 관계만 edge로 표시.
        """
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        for table in schema.tables:
            nodes.append({
                "id": table.name,
                "type": "table",
                "label": table.name,
                "columns": [
                    {
                        "name": col["name"],
                        "type": col.get("type", ""),
                        "pk": col["name"] in table.primary_keys,
                    }
                    for col in table.columns
                ],
            })

            # 외래키: 테이블 간 에지
            for fk in table.foreign_keys:
                edges.append({
                    "source": table.name,
                    "target": fk["references_table"],
                    "type": "foreign_key",
                    "label": f"{fk['column']} → {fk['references_column']}",
                })

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def to_mermaid(schema: 'DatabaseSchema') -> str:
        """
        Mermaid ER 다이어그램 코드 생성

        SchemaExtractor.to_mermaid()에 위임합니다.
        """
        from text_to_sql_agent import SchemaExtractor
        return SchemaExtractor.to_mermaid(schema)

    @staticmethod
    def to_adjacency_list(schema: 'DatabaseSchema') -> Dict[str, List[str]]:
        """
        NetworkX 호환 인접 리스트 (테이블 수준)

        Returns:
            {"employees": ["departments"], "departments": [], ...}
        """
        adj: Dict[str, List[str]] = {t.name: [] for t in schema.tables}

        for table in schema.tables:
            for fk in table.foreign_keys:
                target = fk["references_table"]
                if target not in adj[table.name]:
                    adj[table.name].append(target)

        return adj
