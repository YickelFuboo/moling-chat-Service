import asyncio
import logging
import copy
from abc import ABC
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import nest_asyncio
from app.infrastructure.vector_store import VECTOR_STORE_CONN
from app.infrastructure.vector_store.base import (
    VectorStoreConnection, SearchRequest, MatchExpr, SortField,
    SortOrder, SortFieldType, SortMode, 
)

class OrderByExpr(ABC):
    def __init__(self):
        self.fields = list()
    def asc(self, field: str):
        self.fields.append((field, 0))
        return self
    def desc(self, field: str):
        self.fields.append((field, 1))
        return self
    def fields(self):
        return self.fields


class DocVectorStoreService:
    """
    文档RAG服务 - 处理文档相关的RAG业务逻辑
    包括租户隔离、知识库隔离、搜索融合等业务逻辑
    """

    def __init__(self, store_conn: VectorStoreConnection):
        """
        初始化文档RAG服务
        Args:
            store_conn: 向量存储连接
        """
        self.store_conn = store_conn

    # 索引管理
    async def createIdx(self, tenant_id: str, kb_id: str, vector_size: int) -> bool:
        """
        创建空间
        Args:
            tenant_id: 租户ID
            kb_id: 知识库ID
            vector_size: 向量维度
        """
        # 1. ES和OpenSearch：同一租户使用同一数据空间，kb_id作为数据主键的一部分
        # 2. infinity：没有数据空间的概念，通过table管理数据，则每个知识库创建一张表
        if self.store_conn.get_db_type() == "infinity":
            space_name = f"{tenant_id}_{kb_id}"   # table_name
        else: 
            space_name = f"{tenant_id}"   # index_name

        return await self.store_conn.create_space(space_name, vector_size)

    async def deleteIdx(self, tenant_id: str, kb_id: str = None) -> bool:
        """
        删除索引    
        Args:
            tenant_id: 租户ID
            kb_id: 知识库ID（可选）
        """
        # 1. ES和OpenSearch：同一租户使用同一数据空间，kb_id作为数据主键的一部分
        # 2. infinity：没有数据空间的概念，通过table管理数据，则每个知识库创建一张表
        if self.store_conn.get_db_type() == "infinity":
            space_name = f"{tenant_id}_{kb_id}"
        else:
            if (kb_id.lower() != "all" and kb_id != None):
                # 如果是指定知识库id删除，则啥都不用做，因为一个租户的多个知识库共享index，所以index还不能删除
                return
            space_name = f"{tenant_id}"

        return await self.store_conn.delete_space(space_name)

    async def indexExist(self, tenant_id: str, kb_id: str = None) -> bool:
        """
        检查索引是否存在
        Args:
            tenant_id: 租户ID
            kb_id: 知识库ID（可选）
        """    
        # 1. ES和OpenSearch：同一租户使用同一数据空间，kb_id作为数据主键的一部分
        # 2. infinity：没有数据空间的概念，通过table管理数据，则每个知识库创建一张表    
        if self.store_conn.get_db_type() == "infinity":
            space_name = f"{tenant_id}_{kb_id}"
        else:
            space_name = f"{tenant_id}"

        return await self.store_conn.space_exists(space_name)

    # 文档CRUD操作
    async def insert(self, chunks: List[dict[str, Any]], tenant_id: str, kb_id: str) -> List[str]:
        """
        插入文档
        Args:
            tenant_id: 租户ID
            kb_id: 知识库ID
            documents: 文档列表
        """
        if not chunks:
            return []

        # 1. ES和OpenSearch：同一租户使用同一数据空间，kb_id作为数据主键的一部分
        # 2. infinity：没有数据空间的概念，通过table管理数据，则每个知识库创建一张表 
        new_chunks = []
        if self.store_conn.get_db_type() == "infinity":
            space_name = f"{tenant_id}_{kb_id}"
            new_chunks = chunks
        else:
            space_name = f"{tenant_id}"
            for doc in chunks:
                d_copy = copy.deepcopy(doc)
                if not d_copy.get("kb_id"):
                    d_copy["kb_id"] = kb_id
                new_chunks.append(d_copy)
        
        return await self.store_conn.insert_records(space_name, new_chunks)

    async def update(self, condition: Dict[str, Any], new_value: Dict[str, Any], tenant_id: str, kb_id: str) -> bool:
        """
        更新文档
        Args:
            tenant_id: 租户ID
            kb_id: 知识库ID
            condition: 更新条件
            new_value: 新值
        """
        # 1. ES和OpenSearch：同一租户使用同一数据空间，kb_id作为数据主键的一部分
        # 2. infinity：没有数据空间的概念，通过table管理数据，则每个知识库创建一张表
        new_condition = None
        fields_to_remove = []
        
        if self.store_conn.get_db_type() == "infinity":
            space_name = f"{tenant_id}_{kb_id}"
            new_condition = condition
        else:
            space_name = f"{tenant_id}"
            new_condition = copy.deepcopy(condition)
            new_condition["kb_id"] = kb_id

            # 业务层决定需要删除的字段：排名特征字段（以_feas结尾）
            for field_name in new_value.keys():
                if field_name.endswith('_feas'):
                    fields_to_remove.append(field_name)
        
        return await self.store_conn.update_records(space_name, new_condition, new_value, fields_to_remove=fields_to_remove)

    async def delete(self, condition: Dict[str, Any], tenant_id: str, kb_id: str) -> int:
        """
        删除文档
        Args:
            condition: 删除条件
            index_name: 索引名称
            kb_id: 知识库ID
        Returns:
            删除的文档数量
        """
        # 1. ES和OpenSearch：同一租户使用同一数据空间，kb_id作为数据主键的一部分
        # 2. infinity：没有数据空间的概念，通过table管理数据，则每个知识库创建一张表
        new_condition = None
        if self.store_conn.get_db_type() == "infinity":
            space_name = f"{tenant_id}_{kb_id}"
            new_condition = condition
        else:
            space_name = f"{tenant_id}"
            # 为条件添加kb_id
            new_condition = copy.deepcopy(condition)
            new_condition["kb_id"] = kb_id
    
        return await self.store_conn.delete_records(space_name, new_condition)

    async def get(self, chunk_id: str, tenant_id: str, kb_ids: list[str]) -> Optional[dict[str, Any]]:
        """
        获取单个文档
        Args:
            document_id: 文档ID
            index_name: 索引名称
            kb_ids: 知识库ID列表
        """   
        # 1. ES和OpenSearch：同一租户使用同一数据空间，kb_id作为数据主键的一部分
        # 2. infinity：没有数据空间的概念，通过table管理数据，则每个知识库创建一张表     
        space_names = []
        if self.store_conn.get_db_type() == "infinity":
            for kb_id in kb_ids:
                space_names.append(f"{tenant_id}_{kb_id}")
        else:
            space_names.append(f"{tenant_id}")

        return await self.store_conn.get_record(space_names, chunk_id)

    # 搜索功能
    async def search(self, 
            selectFields: list[str],
            highlightFields: list[str],
            condition: dict,
            matchExprs: list[MatchExpr],
            orderBy: OrderByExpr,
            offset: int,
            limit: int,
            tenant_ids: str|list[str],
            kb_ids: list[str],
            aggFields: list[str] = [],
            rank_feature: dict | None = None) -> dict[str, Any]:
        """
        搜索文档
        Args:
            tenant_id: 租户ID
            kb_ids: 知识库ID列表
            request: 搜索请求
            kb_ids: 知识库ID列表
        """
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")
        assert isinstance(tenant_ids, list) and len(tenant_ids) > 0

        if isinstance(kb_ids, str):
            kb_ids = kb_ids.split(",")

        # 1. ES和OpenSearch：同一租户使用同一数据空间，kb_id作为数据主键的一部分
        # 2. infinity：没有数据空间的概念，通过table管理数据，则每个知识库创建一张表 
        new_condition = copy.deepcopy(condition)

        space_names = []
        if self.store_conn.get_db_type() == "infinity":
            for tenant_id in tenant_ids:
                for kb_id in kb_ids:
                    space_names.append(f"{tenant_id}_{kb_id}")
        else:
            space_names.append(tenant_ids)  
            new_condition["kb_id"] = kb_ids

        # 转换请求
        order_fields = [] 
        if orderBy:
            for field, order in orderBy.fields:
                sort_order=SortOrder.ASC if order == 0 else SortOrder.DESC

                if field in ["page_num_int", "top_int"]:
                    sort_field = SortField.multi_value_field(
                        sort_field=field, 
                        sort_order=sort_order,
                        mode=SortMode.AVG,
                        unmapped_type=SortFieldType.FLOAT,
                        numeric_type=SortFieldType.DOUBLE
                    )
                elif field.endswith("_int") or field.endswith("_flt"):
                    sort_field = SortField.simple_field(
                        sort_field=field, 
                        sort_order=sort_order,
                        unmapped_type=SortFieldType.FLOAT
                    )
                else:
                    sort_field = SortField.simple_field(
                        sort_field=field, 
                        sort_order=sort_order,
                        unmapped_type=SortFieldType.TEXT
                    )

                order_fields.append(sort_field)

        request = SearchRequest(
                    select_fields=selectFields,
                    highlight_fields=highlightFields,
                    condition=new_condition,
                    match_exprs=matchExprs,
                    order_by=order_fields,
                    offset=offset,
                    limit=limit,
                    agg_fields=aggFields,
                    rank_feature=rank_feature
                )

        return await self.store_conn.search(space_names, request)

    # 健康检查
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return await self.store_conn.health_check()

    def get_db_type(self) -> str:
        """获取数据库类型"""
        return self.store_conn.get_db_type()

    """
    Helper functions for search result
    """
    def getTotal(self, result):
        """获取搜索结果总数"""
        return self.store_conn.get_total(result)

    def getChunkIds(self, result):
        """获取搜索结果中的Chunk IDs"""
        return self.store_conn.get_chunk_ids(result)

    def getFields(self, result, fields: list[str]) -> dict[str, dict]:
        """获取搜索结果中指定字段的数据"""
        return self.store_conn.get_fields(result, fields)

    def getHighlight(self, result, keywords: list[str], fieldnm: str):
        """获取搜索结果中的高亮信息"""
        return self.store_conn.get_highlight(result, keywords, fieldnm)

    def getAggregation(self, result, fieldnm: str):
        """获取搜索结果中的聚合信息"""
        return self.store_conn.get_aggregation(result, fieldnm)

    """
    SQL
    """
    async def sql(self, sql: str, fetch_size: int, format: str):
        """
        执行由text-to-sql生成的SQL查询
        Args:
            sql: SQL查询语句
            fetch_size: 获取结果数量限制
            format: 返回格式 (json, csv, tsv, txt, yaml, cbor, smile)
        """
        return await self.store_conn.sql(sql, fetch_size, format)

DOC_STORE_CONN = DocVectorStoreService(VECTOR_STORE_CONN)       