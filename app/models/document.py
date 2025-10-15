from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.models.base import Base
from app.rag_core.utils import ParserType
from app.constants.common import FileType, FileSource


class Document(Base):
    """文档表"""
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, comment="文档ID")
    kb_id = Column(String(36), ForeignKey("knowledgebase.id"), nullable=False, comment="知识库ID")
    name = Column(String(255), nullable=False, comment="文档名称")
    description = Column(Text, comment="文档描述")
    type = Column(String(20), nullable=False, default=FileType.PDF, comment="文档类型")
    suffix = Column(String(10), comment="文件扩展名")
    file_id = Column(String(500), comment="文件唯一标识符")
    size = Column(Integer, default=0, comment="文件大小(字节)")
    parser_id = Column(String(50), nullable=False, default=ParserType.GENERAL, comment="解析器类型")
    parser_config = Column(JSON, comment="解析器配置(JSON)")
    thumbnail_id = Column(String(500), comment="缩略图ID")
    source_type = Column(String(20), default=FileSource.UPLOAD, comment="文件来源")
    created_by = Column(String(36), nullable=False, comment="创建者ID")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    # 关联关系
    knowledgebase = relationship("KB", back_populates="documents")


class File2Document(Base):
    """文件与文档关联表"""
    __tablename__ = "file2document"
    
    id = Column(String(36), primary_key=True, comment="关联ID")
    file_id = Column(String(36), nullable=False, comment="文件ID")
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False, comment="文档ID")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    
    # 关联关系
    document = relationship("Document") 