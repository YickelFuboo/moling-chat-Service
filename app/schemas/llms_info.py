from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class LLMInfo(BaseModel):
    """LLM模型信息"""
    id: Optional[str] = None
    name: str = Field(..., description="模型名称", max_length=128)
    display_name: str = Field(..., description="模型显示名称", max_length=128)
    model_type: str = Field(..., description="模型类型 (chat, embedding, speech2text, image2text, rerank, tts)")
    provider: str = Field(..., description="模型提供商 (openai, anthropic, google, azure, aws, local, custom, silicon)")
    version: Optional[str] = Field(None, description="模型版本", max_length=32)
    description: Optional[str] = Field(None, description="模型描述")
    temperature: Optional[float] = Field(0.7, description="温度参数")
    max_tokens: Optional[int] = Field(4096, description="最大token数")
    max_context_tokens: Optional[int] = Field(None, description="最大上下文token数")
    api_base: Optional[str] = Field(None, description="API基础URL")
    api_key: Optional[str] = Field(None, description="API密钥")
    status: Optional[str] = Field("1", description="状态(0:无效, 1:有效)")
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class SetDefaultModelRequest(BaseModel):
    """设置默认模型请求模型"""
    model_id: str = Field(..., description="模型ID")
    model_type: str = Field(..., description="模型类型 (chat, embedding, speech2text, image2text, rerank, tts)")

class ModelTypeResponse(BaseModel):
    """模型类型响应模型"""
    type: str
    display_name: str
    description: str

class ProviderResponse(BaseModel):
    """提供商响应模型"""
    provider: str
    display_name: str
    description: str

class ModelsByTypeResponse(BaseModel):
    """按类型获取模型响应模型"""
    model_type: str
    models: List[LLMInfo]

class DefaultModelResponse(BaseModel):
    """默认模型响应模型"""
    model_type: str
    model: Optional[LLMInfo] 