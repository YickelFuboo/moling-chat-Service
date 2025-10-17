from datetime import datetime
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from app.models.document import ProcessStatus


class FileUploadResult(BaseModel):
    """单个文件上传结果"""
    filename: str                    # 原始文件名
    success: bool                    # 是否成功
    document_id: Optional[str] = None    # 成功时返回文档ID
    error: Optional[str] = None          # 失败时返回错误信息

# 请求模型
class UpdateDocumentRequest(BaseModel):
    """更新文档请求模型"""
    description: Optional[str] = Field(None, description="文档描述")

class UpdateDocumentMetaFieldsRequest(BaseModel):
    """更新文档元数据字段请求模型"""
    meta_fields: Optional[Any] = Field(None, description="文档元数据字段(JSON格式)")

# 响应模型
class DocumentResponse(BaseModel):
    """文档响应模型"""
    id: str
    kb_id: str
    name: str
    description: Optional[str]
    type: str
    suffix: Optional[str]
    file_id: Optional[str]
    size: int
    thumbnail: Optional[str]
    parser_id: str
    parser_config: Optional[Any]
    meta_fields: Optional[Any] = None
    source_type: str
    process_status: ProcessStatus
    chunk_count: int = 0
    created_by: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ListDocumentResponse(BaseModel):
    """文档列表响应模型"""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class ParserResult(BaseModel):
    """解析结果模型"""
    doc_id: str = Field(..., description="文档ID")
    success: bool = Field(..., description="解析是否成功")

class DocumentChunksResponse(BaseModel):
    """文档切片列表响应模型"""
    chunks: List[Any]
    total: int
    page: int
    page_size: int

