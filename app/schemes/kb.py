from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from app.constants.common import PDFParser, KnowledgeGraphMethod


class CreateKBRequest(BaseModel):
    """创建知识库请求模型"""
    name: str = Field(..., description="知识库名称", max_length=128)
    description: Optional[str] = Field(None, description="知识库描述")
    tenant_id: Optional[str] = Field(None, description="所属团队ID")
    embd_provider_name: Optional[str] = Field(None, description="默认嵌入模型供应商名称")
    embd_model_name: Optional[str] = Field(None, description="默认嵌入模型名称")
    rerank_provider_name: Optional[str] = Field(None, description="默认重排序模型供应商名称")
    rerank_model_name: Optional[str] = Field(None, description="默认重排序模型名称")
    language: Optional[str] = Field("Chinese", description="知识库语言")
    page_rank: Optional[int] = Field(0, description="页面排名算法强度，0表示禁用，1-100表示启用且强度递增", ge=0, le=100)


class UpdateKBRequest(BaseModel):
    """更新知识库请求模型"""
    name: Optional[str] = Field(None, description="知识库名称", max_length=128)
    description: Optional[str] = Field(None, description="知识库描述")
    language: Optional[str] = Field(None, description="知识库语言")
    tenant_id: Optional[str] = Field(None, description="所属团队ID")
    embd_provider_name: Optional[str] = Field(None, description="默认嵌入模型供应商名称")
    embd_model_name: Optional[str] = Field(None, description="默认嵌入模型名称")
    rerank_provider_name: Optional[str] = Field(None, description="默认重排序模型供应商名称")
    rerank_model_name: Optional[str] = Field(None, description="默认重排序模型名称")
    page_rank: Optional[int] = Field(None, description="页面排名算法强度，0表示禁用，1-100表示启用且强度递增", ge=0, le=100)

class KBResponse(BaseModel):
    """知识库响应模型"""
    id: str
    name: str
    description: Optional[str]
    owner_id: str
    tenant_id: Optional[str]
    doc_num: int
    embd_provider_name: Optional[str]
    embd_model_name: Optional[str]
    rerank_provider_name: Optional[str]
    rerank_model_name: Optional[str]
    language: str
    page_rank: int


class KBDetailResponse(BaseModel):
    """知识库详情响应模型"""
    id: str
    name: str
    description: Optional[str]
    owner_id: str
    tenant_id: Optional[str]
    doc_num: int
    embd_provider_name: Optional[str]
    embd_model_name: Optional[str]
    rerank_provider_name: Optional[str]
    rerank_model_name: Optional[str]
    language: str
    page_rank: int
    created_at: datetime
    updated_at: datetime


class ListKBRequest(BaseModel):
    """知识库列表请求模型"""
    page_number: int = Field(1, ge=1, description="页码")
    items_per_page: int = Field(20, ge=1, le=100, description="每页数量")
    order_by: str = Field("created_at", description="排序字段")
    desc: bool = Field(True, description="是否降序")
    keywords: Optional[str] = Field(None, description="搜索关键词")


class ListKBResponse(BaseModel):
    """知识库列表响应模型"""
    items: List[KBResponse]
    total: int
    page_number: int
    items_per_page: int


class CreateKBResponse(BaseModel):
    """创建知识库响应模型"""
    kb_id: str



class ParserConfigRequest(BaseModel):
    """解析器配置请求"""
    parser_id: Optional[str] = Field("general", description="解析器类型", pattern="^(general|semantic|presentation|laws|manual|paper|resume|book|qa|table|naive|picture|one|audio|email|knowledge_graph|tag)$")
    parser_config: Optional[Dict[str, Any]] = Field(None, description="解析器配置")


class ParserConfigResponse(BaseModel):
    """解析器配置响应模型"""
    parser_id: str
    parser_config: Dict[str, Any]


class ParserConfigTemplateResponse(BaseModel):
    """解析器配置模板响应模型"""
    template: Dict[str, Any]
    field_descriptions: Dict[str, Any] 