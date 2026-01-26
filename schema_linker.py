"""
Schema Linker - 스키마 연결 및 관계 분석 모듈

Spider 2.0 벤치마크의 핵심 기술 중 하나인 스키마 링킹을 구현합니다.
자연어 질문에서 관련 테이블과 컬럼을 식별합니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from text_to_sql_agent import DatabaseSchema, TableSchema

# 미리 컴파일된 정규식 패턴
_WORD_PATTERN = re.compile(r'\b[가-힣a-zA-Z_]+\b')
_SUBQUERY_PATTERNS = tuple(re.compile(p) for p in [
    r"가장.+[은는이가]",  # 최대/최소
    r"평균.+보다",  # 비교
    r".+별로.+하고.+",  # 다중 집계
    r".+[은는이가].+[보다보단]",  # 비교 구문
])


@dataclass(slots=True)
class SchemaLink:
    """스키마 링크 정보"""
    mention: str  # 자연어에서의 언급
    table_name: str
    column_name: Optional[str] = None
    confidence: float = 0.0
    link_type: str = "exact"  # exact, fuzzy, semantic


@dataclass(slots=True)
class SchemaLinkingResult:
    """스키마 링킹 결과"""
    links: List[SchemaLink]
    relevant_tables: Set[str]
    relevant_columns: Dict[str, Set[str]]  # table -> columns
    inferred_joins: List[Tuple[str, str, str, str]]  # (table1, col1, table2, col2)


class SchemaLinker:
    """
    스키마 링커

    자연어 질문과 데이터베이스 스키마를 연결합니다.
    """

    __slots__ = ('schema', 'table_index', 'column_index', 'table_columns',
                 '_link_cache', '_entity_patterns_frozen')

    # 한국어 키워드 매핑 (클래스 레벨 상수) - 2026년 확장
    KOREAN_KEYWORDS: Dict[str, List[str]] = {
        # 집계 함수
        "평균": ["AVG", "average", "mean"],
        "합계": ["SUM", "total", "sum_of"],
        "개수": ["COUNT", "number", "count_of"],
        "총": ["SUM", "COUNT", "total"],
        "최대": ["MAX", "maximum", "highest", "top"],
        "최소": ["MIN", "minimum", "lowest", "bottom"],
        "가장 많은": ["MAX", "top"],
        "가장 적은": ["MIN", "bottom"],

        # 정렬
        "오름차순": ["ASC", "ascending"],
        "내림차순": ["DESC", "descending"],
        "높은순": ["DESC", "descending"],
        "낮은순": ["ASC", "ascending"],
        "최신순": ["DESC"],
        "오래된순": ["ASC"],

        # 조건
        "이상": [">=", "greater_or_equal"],
        "이하": ["<=", "less_or_equal"],
        "초과": [">", "greater_than"],
        "미만": ["<", "less_than"],
        "같은": ["=", "equal"],
        "다른": ["!=", "<>", "not_equal"],
        "포함": ["IN", "LIKE", "contains"],
        "제외": ["NOT IN", "NOT LIKE", "excludes"],

        # 시간
        "이번달": ["current_month", "this_month"],
        "지난달": ["last_month", "previous_month"],
        "올해": ["current_year", "this_year"],
        "작년": ["last_year", "previous_year"],
        "오늘": ["today", "current_date"],
        "어제": ["yesterday"],
        "이번주": ["this_week"],
        "지난주": ["last_week"],

        # 일반
        "전체": ["all", "total"],
        "별로": ["by", "group_by"],
        "부서별": ["by_department", "per_department"],
        "직원별": ["by_employee", "per_employee"],
        "월별": ["by_month", "per_month"],
        "연도별": ["by_year", "per_year"],
        "프로젝트별": ["by_project", "per_project"],
        "고객별": ["by_customer", "per_customer"],

        # 논리 연산
        "그리고": ["AND"],
        "또는": ["OR"],
        "아닌": ["NOT"],

        # 조인
        "포함한": ["INNER JOIN"],
        "모두": ["LEFT JOIN", "FULL JOIN"],
    }

    # 일반적인 엔티티-테이블 매핑 - 2026년 확장
    ENTITY_PATTERNS = {
        "직원": ["employee", "employees", "staff", "worker", "workers", "emp", "user", "users", "member"],
        "사원": ["employee", "employees", "staff", "worker", "workers", "emp"],
        "부서": ["department", "departments", "dept", "division", "team"],
        "팀": ["department", "departments", "team", "group"],
        "프로젝트": ["project", "projects", "proj", "task", "tasks"],
        "고객": ["customer", "customers", "client", "clients", "account"],
        "주문": ["order", "orders", "purchase", "purchases", "transaction"],
        "제품": ["product", "products", "item", "items", "goods"],
        "판매": ["sale", "sales", "transaction", "transactions", "revenue"],
        "매출": ["revenue", "sales", "income", "earning"],
        "급여": ["salary", "salaries", "wage", "wages", "pay", "compensation"],
        "연봉": ["salary", "annual_salary", "yearly_pay", "compensation"],
        "계약": ["contract", "contracts", "agreement"],
        "송장": ["invoice", "invoices", "bill", "billing"],
        "결제": ["payment", "payments", "transaction"],
        "배송": ["shipping", "delivery", "shipment"],
        "재고": ["inventory", "stock", "warehouse"],
        "리뷰": ["review", "reviews", "rating", "feedback"],
        "로그": ["log", "logs", "history", "audit"],
        "설정": ["setting", "settings", "config", "configuration"],
        "권한": ["permission", "permissions", "role", "access"],
    }

    def __init__(self, schema: 'DatabaseSchema'):
        self.schema = schema
        self._link_cache: Dict[str, SchemaLinkingResult] = {}
        self._entity_patterns_frozen: FrozenSet[Tuple[str, ...]] = frozenset()
        self._build_index()

    def _build_index(self) -> None:
        """검색을 위한 인덱스 구축"""
        self.table_index: Dict[str, str] = {}  # lowercase -> original
        self.column_index: Dict[str, List[Tuple[str, str]]] = {}  # lowercase -> [(table, column)]
        self.table_columns: Dict[str, List[str]] = {}  # table -> [columns]

        for table in self.schema.tables:
            table_lower = table.name.lower()
            self.table_index[table_lower] = table.name
            self.table_columns[table.name] = []

            for col in table.columns:
                col_lower = col["name"].lower()
                self.table_columns[table.name].append(col["name"])

                if col_lower not in self.column_index:
                    self.column_index[col_lower] = []
                self.column_index[col_lower].append((table.name, col["name"]))

    @staticmethod
    @lru_cache(maxsize=1024)
    def _fuzzy_match(s1: str, s2: str, threshold: float = 0.6) -> float:
        """퍼지 매칭 점수 계산"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def _find_table_mentions(self, question: str) -> List[SchemaLink]:
        """질문에서 테이블 언급 찾기"""
        links: List[SchemaLink] = []
        question_lower = question.lower()
        found_tables: Set[str] = set()

        # 정확한 매칭
        for table_lower, table_name in self.table_index.items():
            if table_lower in question_lower:
                links.append(SchemaLink(
                    mention=table_lower,
                    table_name=table_name,
                    confidence=1.0,
                    link_type="exact"
                ))
                found_tables.add(table_name)

        # 엔티티 패턴 매칭
        for korean_entity, patterns in self.ENTITY_PATTERNS.items():
            if korean_entity not in question:
                continue
            for table_lower, table_name in self.table_index.items():
                if table_name in found_tables:
                    continue
                if any(pattern in table_lower for pattern in patterns):
                    links.append(SchemaLink(
                        mention=korean_entity,
                        table_name=table_name,
                        confidence=0.9,
                        link_type="semantic"
                    ))
                    found_tables.add(table_name)
                    break

        # 퍼지 매칭 (미리 컴파일된 정규식 사용)
        words = _WORD_PATTERN.findall(question)
        for word in words:
            if len(word) < 2:
                continue

            for table_lower, table_name in self.table_index.items():
                if table_name in found_tables:
                    continue
                score = self._fuzzy_match(word, table_lower)
                if score >= 0.7:
                    links.append(SchemaLink(
                        mention=word,
                        table_name=table_name,
                        confidence=score,
                        link_type="fuzzy"
                    ))
                    found_tables.add(table_name)

        return links

    def _find_column_mentions(self, question: str,
                             relevant_tables: Set[str]) -> List[SchemaLink]:
        """질문에서 컬럼 언급 찾기"""
        links: List[SchemaLink] = []
        question_lower = question.lower()
        found_columns: Set[Tuple[str, str]] = set()  # (table, column) 쌍

        # 미리 컴파일된 정규식으로 단어 추출
        words = _WORD_PATTERN.findall(question)

        # 관련 테이블의 컬럼에서 검색
        for table_name in relevant_tables:
            columns = self.table_columns.get(table_name)
            if not columns:
                continue

            for col_name in columns:
                col_lower = col_name.lower()

                # 정확한 매칭
                if col_lower in question_lower:
                    if (table_name, col_name) not in found_columns:
                        links.append(SchemaLink(
                            mention=col_lower,
                            table_name=table_name,
                            column_name=col_name,
                            confidence=1.0,
                            link_type="exact"
                        ))
                        found_columns.add((table_name, col_name))
                    continue

                # 퍼지 매칭
                for word in words:
                    if (table_name, col_name) in found_columns:
                        break
                    score = self._fuzzy_match(word, col_lower)
                    if score >= 0.7:
                        links.append(SchemaLink(
                            mention=word,
                            table_name=table_name,
                            column_name=col_name,
                            confidence=score,
                            link_type="fuzzy"
                        ))
                        found_columns.add((table_name, col_name))
                        break

        return links

    def _infer_joins(self, relevant_tables: Set[str]) -> List[Tuple[str, str, str, str]]:
        """관련 테이블 간의 조인 관계 추론"""
        joins = []
        tables_list = list(relevant_tables)

        for i, table1 in enumerate(tables_list):
            table1_schema = next((t for t in self.schema.tables if t.name == table1), None)
            if not table1_schema:
                continue

            # 외래키 기반 조인
            for fk in table1_schema.foreign_keys:
                if fk["references_table"] in relevant_tables:
                    joins.append((
                        table1, fk["column"],
                        fk["references_table"], fk["references_column"]
                    ))

            # 컬럼명 기반 조인 추론 (id 패턴)
            for col in table1_schema.columns:
                col_name = col["name"].lower()

                # table_id 패턴 (예: dept_id, employee_id)
                for table2 in tables_list[i+1:]:
                    table2_lower = table2.lower()
                    if f"{table2_lower}_id" in col_name or f"{table2_lower[:-1]}_id" in col_name:
                        table2_schema = next((t for t in self.schema.tables if t.name == table2), None)
                        if table2_schema:
                            pk = table2_schema.primary_keys[0] if table2_schema.primary_keys else "id"
                            joins.append((table1, col["name"], table2, pk))

        return list(set(joins))  # 중복 제거

    def link(self, question: str) -> SchemaLinkingResult:
        """
        질문과 스키마 연결 수행

        Args:
            question: 자연어 질문

        Returns:
            SchemaLinkingResult: 스키마 링킹 결과
        """
        # 테이블 언급 찾기
        table_links = self._find_table_mentions(question)
        relevant_tables = {link.table_name for link in table_links}

        # 테이블이 없으면 모든 테이블 고려
        if not relevant_tables:
            relevant_tables = {t.name for t in self.schema.tables}

        # 컬럼 언급 찾기
        column_links = self._find_column_mentions(question, relevant_tables)

        # 결과 구성
        all_links = table_links + column_links

        relevant_columns: Dict[str, Set[str]] = {t: set() for t in relevant_tables}
        for link in column_links:
            if link.column_name and link.table_name in relevant_columns:
                relevant_columns[link.table_name].add(link.column_name)

        # 조인 관계 추론
        inferred_joins = self._infer_joins(relevant_tables) if len(relevant_tables) > 1 else []

        return SchemaLinkingResult(
            links=all_links,
            relevant_tables=relevant_tables,
            relevant_columns=relevant_columns,
            inferred_joins=inferred_joins
        )

    def get_focused_schema(self, question: str) -> str:
        """
        질문에 관련된 스키마만 추출하여 컨텍스트 생성

        대규모 스키마에서 토큰 절약을 위해 사용
        """
        linking_result = self.link(question)

        lines = ["### 관련 스키마:"]

        for table_name in linking_result.relevant_tables:
            table = next((t for t in self.schema.tables if t.name == table_name), None)
            if not table:
                continue

            lines.append(f"\n#### {table.name}")

            for col in table.columns:
                col_marker = ""
                if col["name"] in table.primary_keys:
                    col_marker = " [PK]"
                if any(fk["column"] == col["name"] for fk in table.foreign_keys):
                    col_marker += " [FK]"

                lines.append(f"  - {col['name']}: {col['type']}{col_marker}")

            if table.foreign_keys:
                lines.append("  외래키:")
                for fk in table.foreign_keys:
                    lines.append(f"    {fk['column']} -> {fk['references_table']}.{fk['references_column']}")

        if linking_result.inferred_joins:
            lines.append("\n### 추론된 조인 관계:")
            for join in linking_result.inferred_joins:
                lines.append(f"  {join[0]}.{join[1]} = {join[2]}.{join[3]}")

        return "\n".join(lines)


class QueryDecomposer:
    """
    복잡한 질문을 서브 쿼리로 분해하는 모듈

    다단계 추론을 위한 핵심 컴포넌트
    """

    DECOMPOSITION_KEYWORDS = {
        "그리고": "AND",
        "또는": "OR",
        "하지만": "EXCEPT",
        "제외하고": "EXCEPT",
        "중에서": "INTERSECT",
        "비교해서": "COMPARE",
    }

    def __init__(self):
        pass

    def is_complex_query(self, question: str) -> bool:
        """복잡한 쿼리인지 판단"""
        # 다중 조건 검사
        condition_count = sum(
            1 for keyword in self.DECOMPOSITION_KEYWORDS
            if keyword in question
        )

        # 서브쿼리가 필요한 패턴 검사
        subquery_patterns = [
            r"가장.+[은는이가]",  # 최대/최소
            r"평균.+보다",  # 비교
            r".+별로.+하고.+",  # 다중 집계
            r".+[은는이가].+[보다보단]",  # 비교 구문
        ]

        has_subquery_pattern = any(
            re.search(pattern, question)
            for pattern in subquery_patterns
        )

        return condition_count > 0 or has_subquery_pattern

    def decompose(self, question: str) -> List[str]:
        """
        복잡한 질문을 서브 질문으로 분해

        Args:
            question: 원본 질문

        Returns:
            List[str]: 서브 질문 목록
        """
        if not self.is_complex_query(question):
            return [question]

        sub_questions = []

        # 키워드로 분할
        for keyword, operation in self.DECOMPOSITION_KEYWORDS.items():
            if keyword in question:
                parts = question.split(keyword)
                sub_questions.extend([p.strip() for p in parts if p.strip()])

        if not sub_questions:
            sub_questions = [question]

        return sub_questions


# 사용 예시
if __name__ == "__main__":
    from text_to_sql_agent import SchemaExtractor, create_sample_database

    # 샘플 DB 로드
    db_path = create_sample_database()
    schema = SchemaExtractor.extract_sqlite_schema(db_path)

    # 스키마 링커 테스트
    linker = SchemaLinker(schema)

    test_questions = [
        "개발팀 직원들의 평균 연봉을 알려주세요",
        "프로젝트에 참여하는 직원 목록",
        "부서별 직원 수는?",
        "가장 많은 예산을 가진 프로젝트",
    ]

    print("=" * 60)
    print("Schema Linker 테스트")
    print("=" * 60)

    for question in test_questions:
        print(f"\n질문: {question}")
        result = linker.link(question)
        print(f"  관련 테이블: {result.relevant_tables}")
        print(f"  관련 컬럼: {dict(result.relevant_columns)}")
        print(f"  추론된 조인: {result.inferred_joins}")
        print(f"  링크 수: {len(result.links)}")

    # 집중 스키마 출력
    print("\n" + "=" * 60)
    print("집중 스키마 예시:")
    print(linker.get_focused_schema("개발팀 직원들의 평균 연봉"))
