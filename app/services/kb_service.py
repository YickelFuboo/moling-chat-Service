from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import json
from sqlalchemy import select, and_, func, desc, asc, or_
import uuid
from app.models import KB
from app.constants.common import KBConstants
from app.infrastructure.database import get_db
from app.infrastructure.llm.llms import embedding_factory
from app.infrastructure.llm.llms import rerank_factory
from app.rag_core.utils import ParserType
from app.services.common.doc_vector_store_service import DOC_STORE_CONN
from app.rag_core.constants import PAGERANK_FLD


class KBService:
    """知识库服务类"""

    @staticmethod
    async def create_kb(
        session: AsyncSession,
        name: str,
        owner_id: str,
        description: str = None,        
        language: str = "Chinese",
        tenant_id: str = None,
        embd_provider_name: str = None,
        embd_model_name: str = None,
        rerank_provider_name: str = None,
        rerank_model_name: str = None,
        page_rank: int = 0,
        parser_id: str = ParserType.GENERAL,
        parser_config: Optional[Dict[str, Any]] = None
    ) -> KB:
        """创建知识库"""
        try:
            # 检查关键信息
            if not tenant_id:
                logging.error("缺少租户信息")
                raise ValueError("缺少租户信息")

            if not owner_id:
                logging.error("缺少Owner信息")
                raise ValueError("缺少Owner信息")
            
            # 检查租户下知识库名称是否重复
            if await KBService._check_kb_name_exists_in_tenant(session, name, tenant_id):
                raise ValueError("知识库名称已存在")
            
            # 如果没有指定嵌入模型，从工厂获取默认值
            if not embd_provider_name or not embd_model_name:
                try:
                    embd_provider_name, embd_model_name = embedding_factory.get_default_model()
                except Exception as e:
                    logging.error(f"获取默认嵌入模型失败: {e}")
                    raise
            
            # 如果没有指定重排序模型，从工厂获取默认值
            if not rerank_provider_name or not rerank_model_name:
                try:
                    rerank_provider_name, rerank_model_name = rerank_factory.get_default_model()
                except Exception as e:
                    logging.error(f"获取默认重排序模型失败: {e}")
                    raise
            
            # 如果没有指定解析器配置，使用默认配置
            if not parser_config:
                parser_config = KBConstants.DEFAULT_PARSER_CONFIG
            
            kb_id = str(uuid.uuid4()).replace("-", "")
            kb = KB(
                id=kb_id,
                name=name,
                description=description,
                language=language,
                owner_id=owner_id,
                tenant_id=tenant_id,
                embd_provider_name=embd_provider_name,
                embd_model_name=embd_model_name,
                rerank_provider_name=rerank_provider_name,
                rerank_model_name=rerank_model_name,
                page_rank=page_rank,
                parser_id=parser_id,
                parser_config=parser_config
            )
            
            session.add(kb)
            await session.commit()
            await session.refresh(kb)
            
            logging.info(f"知识库创建成功: {kb_id}")
            return kb
            
        except Exception as e:
            await session.rollback()
            logging.error(f"创建知识库失败: {e}")
            raise
    
    @staticmethod
    async def get_kb_by_id(
        session: AsyncSession,
        kb_id: str
    ) -> Optional[KB]:
        """根据ID获取知识库"""
        try:
            stmt = select(KB).where(
                and_(
                    KB.id == kb_id,
                    KB.status == "1"
                )
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
            
        except Exception as e:
            logging.error(f"获取知识库失败: {e}")
            raise

    @staticmethod
    async def get_kb_by_ids(
        session: AsyncSession,
        kb_ids: List[str]
    ) -> Optional[List[KB]]:
        """根据ID获取知识库"""
        try:
            stmt = select(KB).where(
                and_(
                    KB.id.in_(kb_ids),
                    KB.status == "1"
                )
            )
            result = await session.execute(stmt)
            return result.scalars().all()
            
        except Exception as e:
            logging.error(f"获取知识库失败: {e}")
            raise
    
    @staticmethod
    async def get_kb_detail(
        session: AsyncSession,
        kb_id: str
    ) -> Optional[Dict[str, Any]]:
        """获取知识库详情"""
        try:
            kb = await KBService.get_kb_by_id(session, kb_id)
            if not kb:
                return None
            
            return {
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "language": kb.language,
                "owner_id": kb.owner_id,
                "tenant_id": kb.tenant_id,
                "doc_num": kb.doc_num,
                "embd_provider_name": kb.embd_provider_name,
                "embd_model_name": kb.embd_model_name,
                "rerank_provider_name": kb.rerank_provider_name,
                "rerank_model_name": kb.rerank_model_name,
                "page_rank": kb.page_rank,
                "parser_id": kb.parser_id,
                "parser_config": kb.parser_config,
                "created_at": kb.created_at.isoformat() if kb.created_at else None,
                "updated_at": kb.updated_at.isoformat() if kb.updated_at else None
            }
            
        except Exception as e:
            logging.error(f"获取知识库详情失败: {e}")
            raise
    
    @staticmethod
    async def list_kbs(
        session: AsyncSession,
        owner_id: str,
        page_number: int = 1,
        items_per_page: int = 20,
        order_by: str = "created_at",
        desc_order: bool = True,
        keywords: str = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取知识库列表"""
        try:
            # 构建查询条件
            conditions = [
                KB.status == "1",
                KB.owner_id == owner_id
            ]
            
            # 关键词搜索
            if keywords:
                conditions.append(
                    func.lower(KB.name).contains(keywords.lower())
                )
            
            # 构建查询语句
            stmt = select(KB).where(and_(*conditions))
            
            # 排序
            if hasattr(KB, order_by):
                order_field = getattr(KB, order_by)
                if desc_order:
                    stmt = stmt.order_by(desc(order_field))
                else:
                    stmt = stmt.order_by(asc(order_field))
            else:
                # 默认按创建时间排序
                if desc_order:
                    stmt = stmt.order_by(desc(KB.created_at))
                else:
                    stmt = stmt.order_by(asc(KB.created_at))
            
            # 获取总数
            count_stmt = select(func.count()).select_from(stmt.subquery())
            count_result = await session.execute(count_stmt)
            total_count = count_result.scalar()
            
            # 分页
            if page_number and items_per_page:
                offset = (page_number - 1) * items_per_page
                stmt = stmt.offset(offset).limit(items_per_page)
            
            # 执行查询
            result = await session.execute(stmt)
            knowledgebases = result.scalars().all()
            
            # 转换为字典列表
            kb_list = []
            for kb in knowledgebases:
                kb_dict = {
                    "id": kb.id,
                    "name": kb.name,
                    "description": kb.description,
                    "language": kb.language,
                    "owner_id": kb.owner_id,
                    "tenant_id": kb.tenant_id,
                    "doc_num": kb.doc_num,
                    "embd_provider_name": kb.embd_provider_name,
                    "embd_model_name": kb.embd_model_name,
                    "rerank_provider_name": kb.rerank_provider_name,
                    "rerank_model_name": kb.rerank_model_name,
                    "page_rank": kb.page_rank,
                    "parser_id": kb.parser_id,
                    "parser_config": kb.parser_config,
                    "created_at": kb.created_at,
                    "updated_at": kb.updated_at
                }
                kb_list.append(kb_dict)
            
            return kb_list, total_count
            
        except Exception as e:
            logging.error(f"获取知识库列表失败: {e}")
            raise
    
    @staticmethod
    async def update_kb(
        session: AsyncSession,
        kb_id: str,
        update_data: Dict[str, Any],
        owner_id: str
    ) -> Optional[KB]:
        """更新知识库"""
        try:
            # 获取知识库
            kb = await KBService.get_kb_by_id(session, kb_id)
            if not kb:
                raise ValueError("知识库不存在")
            
            # 检查权限
            if kb.owner_id != owner_id:
                raise ValueError("不是知识库Owner")
            
            # 如果更新租户ID，验证租户权限

            # 如果更新名称，检查租户下名称是否重复
            if "name" in update_data:
                tenant_id = update_data.get("tenant_id", kb.tenant_id)
                if await KBService._check_kb_name_exists_in_tenant(session, update_data["name"], tenant_id, kb_id):
                    raise ValueError("知识库名称已存在")

            if "parser_id" in update_data:
                if update_data["parser_id"] not in [member.value for member in ParserType]:
                    raise ValueError("解析器类型不存在")
            
            # 记录修改前的page_rank
            old_page_rank = kb.page_rank

            # 更新字段
            for field, value in update_data.items():
                if hasattr(kb, field) and value is not None:
                    setattr(kb, field, value)

            # 检查并纠错模型信息
            kb = KBService._check_and_update_model_info(kb)

            await session.commit()
            await session.refresh(kb)
            
            # 如果PageRank发生变化，同步更新向量存储库
            if old_page_rank != kb.page_rank:
                try:
                    success = await KBService._update_kb_pagerank_in_vector_store(kb)
                    if success:
                        logging.info(f"知识库 {kb_id} 的PageRank向量存储更新成功")
                    else:
                        logging.warning(f"知识库 {kb_id} 的PageRank向量存储更新失败，但数据库更新成功")
                except Exception as e:
                    logging.error(f"更新知识库PageRank向量存储失败: {e}")
            
            logging.info(f"知识库更新成功: {kb_id}")
            return kb
            
        except Exception as e:
            await session.rollback()
            logging.error(f"更新知识库失败: {e}")
            raise

    @ staticmethod
    def _check_and_update_model_info(kb: KB) -> KB:
        """
        验证并纠错模型配置
        """
        if not embedding_factory.if_model_support(kb.embd_provider_name, kb.embd_model_name):
            kb.embd_provider_name, kb.embd_model_name = embedding_factory.get_default_model()
            logging.warning(f"嵌入模型不支持，使用默认模型: {kb.embd_provider_name}/{kb.embd_model_name}")

        if not rerank_factory.if_model_support(kb.rerank_provider_name, kb.rerank_model_name):
            kb.rerank_provider_name, kb.rerank_model_name = rerank_factory.get_default_model()
            logging.warning(f"重排序模型不支持，使用默认模型: {kb.rerank_provider_name}/{kb.rerank_model_name}")

        return kb
    
    @staticmethod
    async def delete_kb(
        session: AsyncSession,
        kb_id: str,
        owner_id: str
    ):
        """删除知识库"""
        try:
            kb = await KBService.get_kb_by_id(session, kb_id)
            if not kb:
                raise ValueError("知识库不存在")
            
            # 检查权限
            if kb.owner_id != owner_id:
                raise ValueError("不是知识库Owner")

            # 删除KB下所有文档
            from app.services.doc_service import DocumentService
            documents, total = await DocumentService.get_documents_by_kb_id(session, kb_id)
            for document in documents:
                await DocumentService.delete_document_by_id(session, document.id)
            
            # 硬删除知识库
            await session.delete(kb)
            await session.commit()
            
            logging.info(f"知识库删除成功: {kb_id}")
            
        except Exception as e:
            await session.rollback()
            logging.error(f"删除知识库失败: {e}")
            raise
    
    @staticmethod
    async def update_parser_config(kb_id: str, config: Dict[str, Any]) -> bool:
        """
        更新知识库解析器配置
        
        Args:
            kb_id: 知识库ID
            config: 要更新的配置字典
            
        Returns:
            bool: 更新是否成功
            
        Raises:
            ValueError: 配置为空或无效
            LookupError: 知识库不存在
            Exception: 其他数据库操作错误
        """
        if not config:
            logging.warning(f"尝试更新知识库 {kb_id} 的解析器配置，但配置为空")
            return False
            
        if not isinstance(config, dict):
            raise ValueError("配置必须是字典类型")
        
        db_gen = get_db()
        session = await db_gen.__anext__()
        try:
            try:
                # 获取知识库
                kb = await KBService.get_kb_by_id(session, kb_id)
                if not kb:
                    raise LookupError(f"知识库 {kb_id} 不存在")
                
                # 记录原始配置用于日志
                original_config = kb.parser_config.copy() if kb.parser_config else {}
                
                # 深度合并配置
                def deep_merge_config(old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
                    """
                    深度合并配置字典
                    
                    Args:
                        old_config: 原始配置字典
                        new_config: 新配置字典
                    """
                    for key, value in new_config.items():
                        if key not in old_config:
                            old_config[key] = value
                            continue
                            
                        if isinstance(value, dict) and isinstance(old_config[key], dict):
                            deep_merge_config(old_config[key], value)
                        else:
                            old_config[key] = value
                
                # 执行配置合并
                if not kb.parser_config:
                    kb.parser_config = {}
                deep_merge_config(kb.parser_config, config)
                
                # 特殊处理：如果新配置中没有raptor，则删除原有的raptor配置
                if not config.get("raptor") and kb.parser_config.get("raptor"):
                    del kb.parser_config["raptor"]
                    logging.info(f"已删除知识库 {kb_id} 的raptor配置")
                
                # 提交更改
                await session.commit()
                await session.refresh(kb)
                
                # 记录更新日志
                logging.info(f"知识库 {kb_id} 解析器配置更新成功")
                logging.debug(f"原始配置: {original_config}")
                logging.debug(f"新配置: {kb.parser_config}")
                
                return True
                
            except Exception as e:
                await session.rollback()
                logging.error(f"更新知识库 {kb_id} 解析器配置失败: {e}")
                raise
        finally:
            try:
                await db_gen.aclose()
            except Exception as e:
                logging.warning(f"关闭数据库会话失败: {e}")

    @staticmethod
    async def _check_kb_name_exists_in_tenant(
        session: AsyncSession,
        name: str,
        tenant_id: str,
        exclude_kb_id: str = None
    ) -> bool:
        """检查租户下知识库名称是否已存在"""
        try:
            conditions = [
                KB.name == name,
                KB.tenant_id == tenant_id,
                KB.status == "1"
            ]
            
            # 排除指定的知识库ID（用于更新时检查）
            if exclude_kb_id:
                conditions.append(KB.id != exclude_kb_id)
            
            stmt = select(KB.id).where(and_(*conditions))
            result = await session.execute(stmt)
            return result.scalar_one_or_none() is not None
            
        except Exception as e:
            logging.error(f"检查知识库名称是否存在失败: {e}")
            raise

    @staticmethod
    async def _update_kb_pagerank_in_vector_store(kb: KB) -> bool:
        """更新知识库在向量存储库中的PageRank数据"""
        try:
            # 构建更新条件：匹配指定知识库的所有文档
            condition = {"kb_id": kb.id}
            
            if kb.page_rank > 0:
                logging.info(f"更新知识库 {kb.id} 的PageRank为 {kb.page_rank}")
                new_value = {PAGERANK_FLD: kb.page_rank}
            else:
                logging.info(f"删除知识库 {kb.id} 的PageRank字段")
                condition["exists"] = PAGERANK_FLD    
                new_value = {"remove": PAGERANK_FLD}

            # 执行批量更新
            result = await DOC_STORE_CONN.update(
                condition=condition,
                new_value=new_value,
                tenant_id=kb.tenant_id,
                kb_id=kb.id
            )
            
            if result:
                logging.info(f"知识库 {kb.id} 的PageRank字段更新成功")
                return True
            else:
                logging.error(f"知识库 {kb.id} 的PageRank字段更新失败")
                return False
                
        except Exception as e:
            logging.error(f"更新知识库PageRank失败: {e}")
            return False

    @staticmethod
    async def get_field_map(session: AsyncSession, kb_ids: List[str]) -> Optional[dict]:
        """
        获取知识库的字段映射，用于SQL查询
        
        Args:
            session: 数据库会话
            kb_ids: 知识库ID列表
            
        Returns:
            dict: 字段映射字典，如果知识库不支持SQL查询则返回None
        """
        try:
            # 获取知识库信息
            kbs = await KBService.get_kb_by_ids(session, kb_ids)
            if not kbs:
                return None
            
            # 构建字段映射配置
            conf = {}
            for kb in kbs:
                if kb.parser_config:
                    try:
                        # parser_config已经是字典对象，直接使用
                        if isinstance(kb.parser_config, dict) and "field_map" in kb.parser_config:
                            conf.update(kb.parser_config["field_map"])
                        elif isinstance(kb.parser_config, str):
                            # 如果是字符串，则解析JSON
                            parser_config = json.loads(kb.parser_config)
                            if "field_map" in parser_config:
                                conf.update(parser_config["field_map"])
                    except (json.JSONDecodeError, TypeError) as e:
                        logging.warning(f"解析知识库 {kb.id} 的parser_config失败: {e}")
                        continue
            
            # 如果没有任何字段映射，返回None
            return conf if conf else None
            
        except Exception as e:
            logging.error(f"获取知识库字段映射失败: {e}")
            return None