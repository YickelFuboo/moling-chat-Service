from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


# 请求模型
class UploadDocumentRequest(BaseModel):
    """上传文档请求模型"""
    kb_id: str = Field(..., description="知识库ID")


class CreateDocumentRequest(BaseModel):
    """创建虚拟文档请求模型"""
    kb_id: str = Field(..., description="知识库ID")
    name: str = Field(..., description="文档名称", max_length=255)
    description: Optional[str] = Field(None, description="文档描述")


class ListDocumentRequest(BaseModel):
    """获取文档列表请求模型"""
    kb_id: str = Field(..., description="知识库ID")
    page_number: int = Field(1, description="页码", ge=1)
    items_per_page: int = Field(20, description="每页数量", ge=1, le=100)
    order_by: str = Field("created_at", description="排序字段")
    desc: bool = Field(True, description="是否降序")
    keywords: Optional[str] = Field(None, description="搜索关键词")
    status: Optional[str] = Field(None, description="文档状态过滤")


class UpdateDocumentRequest(BaseModel):
    """更新文档请求模型"""
    name: Optional[str] = Field(None, description="文档名称", max_length=255)
    description: Optional[str] = Field(None, description="文档描述")


class ChangeStatusRequest(BaseModel):
    """更改文档状态请求模型"""
    doc_ids: List[str] = Field(..., description="文档ID列表")
    status: str = Field(..., description="目标状态")


class RunDocumentRequest(BaseModel):
    """运行文档解析请求模型"""
    doc_ids: List[str] = Field(..., description="文档ID列表")
    run: str = Field(..., description="运行操作")
    delete: Optional[bool] = Field(False, description="是否删除")


class WebCrawlRequest(BaseModel):
    """网页爬取请求模型"""
    kb_id: str = Field(..., description="知识库ID")
    name: str = Field(..., description="文档名称", max_length=255)
    url: str = Field(..., description="网页URL")
    description: Optional[str] = Field(None, description="文档描述")


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
    parser_config: Optional[str]
    source_type: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class TaskResponse(BaseModel):
    """任务响应模型"""
    id: str
    doc_id: str
    status: str
    progress: int
    error_message: Optional[str]
    result: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DocumentDetailResponse(BaseModel):
    """文档详情响应模型"""
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
    parser_config: Optional[str]
    source_type: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CreateDocumentResponse(BaseModel):
    """创建文档响应模型"""
    document: DocumentResponse


class ListDocumentResponse(BaseModel):
    """文档列表响应模型"""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class FileUploadResult(BaseModel):
    """单个文件上传结果"""
    filename: str                    # 原始文件名
    success: bool                    # 是否成功
    document_id: Optional[str] = None    # 成功时返回文档ID
    task_id: Optional[str] = None        # 成功时返回任务ID
    error: Optional[str] = None          # 失败时返回错误信息


class UploadDocumentResponse(BaseModel):
    """上传文档响应模型"""
    results: List[FileUploadResult]  # 每个文件的上传结果


class ChangeStatusResponse(BaseModel):
    """更改状态响应模型"""
    success_count: int
    failed_count: int
    errors: List[str]


class ParseTaskResponse(BaseModel):
    """解析任务响应模型"""
    id: str
    doc_id: str
    status: str
    progress: int
    error_message: Optional[str]
    result: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class TaskStatusResponse(BaseModel):
    """任务状态响应模型"""
    id: str
    doc_id: str
    status: str
    progress: int
    error_message: Optional[str]
    result: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# 更新前向引用
UploadDocumentResponse.model_rebuild() 