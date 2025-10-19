from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色: user, assistant, system")
    content: str = Field(..., description="消息内容")
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式"""
        return {"role": self.role, "content": self.content}

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
    enable_deep_research: bool = Field(False, description="是否启用深度研究")
    use_kg: bool = Field(False, description="是否启用知识图谱检索")
    tavily_api_key: Optional[str] = Field(None, description="Tavily API密钥，用于外部知识源检索")
    # LLM配置
    temperature: float = Field(0.1, description="生成温度", ge=0.0, le=2.0)


class KbQueryRequest(BaseModel):
    """知识库查询请求模型"""
    question: str = Field(..., description="用户问题")
    history_messages: Optional[List[ChatMessage]] = Field(None, description="历史对话消息列表")
    kb_ids: List[str] = Field(..., description="知识库ID列表", min_items=1)
    doc_ids: Optional[List[str]] = Field(None, description="指定检索的文档ID列表")
    # 功能开关
    enable_quote: bool = Field(True, description="是否启用引用")
    enable_multi_questions: bool = Field(False, description="是否启用多问题生成")
    enable_keyword_extraction: bool = Field(False, description="是否启用关键词提取")
    enable_deep_research: bool = Field(False, description="是否启用深度研究")
    enable_web_search: bool = Field(False, description="是否启用网络搜索")
    enable_knowledge_graph: bool = Field(False, description="是否启用知识图谱检索")
    target_language: Optional[str] = Field(None, description="目标语言代码，如：zh、en、ja等")

class DocumentReference(BaseModel):
    """文档引用模型"""
    doc_id: str = Field(..., description="文档ID")
    doc_name: str = Field(..., description="文档名称")
    count: int = Field(..., description="引用次数")


class ChunkInfo(BaseModel):
    """文档片段信息模型 - 与chunks_format函数输出格式一致"""
    id: Optional[str] = Field(None, description="片段ID")
    content: Optional[str] = Field(None, description="片段内容")
    document_id: Optional[str] = Field(None, description="文档ID")
    document_name: Optional[str] = Field(None, description="文档名称")
    dataset_id: Optional[Any] = Field(None, description="数据集ID")
    image_id: Optional[Any] = Field(None, description="图像ID")
    positions: Optional[Any] = Field(None, description="位置信息")
    url: Optional[str] = Field(None, description="URL")
    similarity: Optional[float] = Field(None, description="相似度分数")
    vector_similarity: Optional[float] = Field(None, description="向量相似度")
    term_similarity: Optional[float] = Field(None, description="词汇相似度")
    doc_type: Optional[str] = Field(None, description="文档类型")


class QaReference(BaseModel):
    """问答引用信息模型"""
    total: Optional[int] = Field(0, description="总检索数量")
    chunks: Optional[List[ChunkInfo]] = Field(default_factory=list, description="检索到的文档片段")
    doc_aggs: Optional[List[DocumentReference]] = Field(default_factory=list, description="文档聚合信息")


class QaResponse(BaseModel):
    """问答响应模型"""
    answer: str = Field(..., description="回答内容")
    reference: Optional[QaReference] = Field(None, description="引用信息")
    prompt: Optional[str] = Field(None, description="使用的提示词")
    created_at: Optional[float] = Field(None, description="创建时间戳")


class StreamQaResponse(BaseModel):
    """流式问答响应模型"""
    answer: str = Field(..., description="回答内容")
    reference: Optional[QaReference] = Field(None, description="引用信息")
    prompt: Optional[str] = Field(None, description="使用的提示词")
    created_at: Optional[float] = Field(None, description="创建时间戳")
