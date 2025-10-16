from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色: user, assistant, system")
    content: str = Field(..., description="消息内容")
    doc_ids: Optional[List[str]] = Field(None, description="关联的文档ID列表")

class SingleQaRequest(BaseModel):
    """单次问答请求模型"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    kb_ids: List[str] = Field(..., description="知识库ID列表", min_items=1)


class ChatRequest(BaseModel):
    """多轮对话请求模型"""
    messages: List[ChatMessage] = Field(..., description="对话消息列表", min_items=1)
    kb_ids: List[str] = Field(..., description="知识库ID列表", min_items=1)
    is_stream: bool = Field(True, description="是否流式返回")
    doc_ids: Optional[List[str]] = Field(None, description="指定检索的文档ID列表")
    # 对话配置参数
    top_n: int = Field(12, description="检索文档数量", ge=1, le=50)
    similarity_threshold: float = Field(0.1, description="相似度阈值", ge=0.0, le=1.0)
    vector_similarity_weight: float = Field(0.3, description="向量相似度权重", ge=0.0, le=1.0)
    top_k: int = Field(5, description="重排序后保留的文档数量", ge=1, le=20)
    # 提示词配置
    system_prompt: Optional[str] = Field(None, description="系统提示词")
    enable_quote: bool = Field(True, description="是否启用引用")
    enable_multiturn_refine: bool = Field(False, description="是否启用多轮对话优化")
    target_language: Optional[str] = Field(None, description="目标语言代码，如：zh、en、ja等")
    enable_keyword_extraction: bool = Field(False, description="是否启用关键词提取")
    use_kg: bool = Field(False, description="是否启用知识图谱检索")
    tavily_api_key: Optional[str] = Field(None, description="Tavily API密钥，用于外部知识源检索")
    # LLM配置
    temperature: float = Field(0.1, description="生成温度", ge=0.0, le=2.0)

class DocumentChunk(BaseModel):
    """文档片段模型"""
    doc_id: str = Field(..., description="文档ID")
    chunk_id: str = Field(..., description="片段ID")
    content: str = Field(..., description="片段内容")
    content_ltks: str = Field(..., description="分词后的内容")
    similarity: float = Field(..., description="相似度分数")
    docnm_kwd: Optional[str] = Field(None, description="文档名称关键词")

class DocumentReference(BaseModel):
    """文档引用模型"""
    doc_id: str = Field(..., description="文档ID")
    doc_name: str = Field(..., description="文档名称")
    count: int = Field(..., description="引用次数")


class QaReference(BaseModel):
    """问答引用信息模型"""
    total: int = Field(..., description="总检索数量")
    chunks: List[DocumentChunk] = Field(..., description="检索到的文档片段")
    doc_aggs: List[DocumentReference] = Field(..., description="文档聚合信息")


class QaResponse(BaseModel):
    """问答响应模型"""
    answer: str = Field(..., description="回答内容")
    reference: QaReference = Field(..., description="引用信息")
    prompt: Optional[str] = Field(None, description="使用的提示词")
    audio_binary: Optional[str] = Field(None, description="音频二进制数据(十六进制)")
    created_at: float = Field(..., description="创建时间戳")


class StreamQaResponse(BaseModel):
    """流式问答响应模型"""
    answer: str = Field(..., description="回答内容")
    reference: Optional[Dict[str, Any]] = Field(None, description="引用信息")
    prompt: Optional[str] = Field(None, description="使用的提示词")
    audio_binary: Optional[str] = Field(None, description="音频二进制数据(十六进制)")
    created_at: Optional[float] = Field(None, description="创建时间戳")


