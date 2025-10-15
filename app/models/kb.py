import os
from enum import StrEnum
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin
from app.constants.common import KBConstants, PDFParser, KnowledgeGraphMethod
from app.rag_core.utils import ParserType


class KB(Base, TimestampMixin):
    """知识库模型"""
    __tablename__ = "knowledgebase"
    
    # 主键
    id = Column(String(32), primary_key=True)
    
    # 知识库名称
    name = Column(String(128), nullable=False, index=True, comment="知识库名称")

    # 知识库描述
    description = Column(Text, nullable=True, comment="知识库描述")

    # 知识库语言，默认为中文或英文
    language = Column(String(32), nullable=True, default="Chinese" if "zh_CN" in os.getenv("LANG", "") else "English", comment="English|Chinese", index=True)
    
    # 所有者ID
    owner_id = Column(String(32), nullable=False, index=True, comment="所有者用户ID")
    
    # 租户ID
    tenant_id = Column(String(32), nullable=True, index=True, comment="所属租户ID")
    
    # 文档数量
    doc_num = Column(Integer, default=0, index=True)
    
    # 默认嵌入模型配置
    embd_provider_name = Column(String(32), nullable=True, index=True, comment="默认嵌入模型供应商名称")
    embd_model_name = Column(String(32), nullable=True, index=True, comment="默认嵌入模型名称")
    
    # 默认重排序模型配置
    rerank_provider_name = Column(String(32), nullable=True, index=True, comment="默认重排序模型供应商名称")
    rerank_model_name = Column(String(32), nullable=True, index=True, comment="默认重排序模型名称")
    
    # ==================== 基础配置 ====================
    # 默认解析器类型
    parser_id = Column(String(32), nullable=False, default=ParserType.GENERAL, comment="解析器类型")
    
    # ==================== 解析器配置 ====================
    # 解析器配置（JSON格式）
    parser_config = Column(JSON, nullable=False, default=KBConstants.DEFAULT_PARSER_CONFIG, comment="解析器配置(JSON格式)")
    
    # 页面排名
    page_rank = Column(Integer, default=0, index=False, comment="页面排名算法强度，0表示禁用，1-100表示启用且强度递增")

    # 知识库状态，1表示有效，0表示无效
    status = Column(String(1), nullable=True, default="1", index=True, comment="状态(0:无效, 1:有效)")
    
    # 关联关系
    documents = relationship("Document", back_populates="knowledgebase")
    
    def __str__(self):
        return self.name 