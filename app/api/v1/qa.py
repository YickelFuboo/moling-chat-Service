import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
import json
from sqlalchemy.ext.asyncio import AsyncSession
from app.infrastructure.database import get_db
from app.schemes.qa import SingleQaRequest, QaResponse, StreamQaResponse, ChatRequest
from app.services.qa_service import QAService

router = APIRouter(prefix="/api/qa", tags=["问答服务"])


@router.post("/single", response_model=QaResponse)
async def single_qa(
    request: SingleQaRequest,
    user_id: str = Query(..., description="用户ID"),
    tenant_id: str = Query(..., description="租户ID"),
    session: AsyncSession = Depends(get_db)
):
    """单次问答接口"""
    try:
        # 验证请求参数
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "问题不能为空"}
            )
        
        if not request.kb_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "知识库ID列表不能为空"}
            )
        
        # 执行问答
        response_generator = QAService.single_ask(
            session=session,
            request=request,
            user_id=user_id,
            tenant_id=tenant_id,
            is_stream=False
        )
        
        # 获取最终响应
        final_response = None
        async for response in response_generator:
            final_response = response
        
        if not final_response:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"message": "问答服务返回空结果"}
            )
        
        return QaResponse(**final_response)
        
    except Exception as e:
        logging.error(f"单次问答失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"问答失败: {str(e)}"}
        )


@router.post("/single/stream")
async def single_qa_stream(
    request: SingleQaRequest,
    user_id: str = Query(..., description="用户ID"),
    tenant_id: str = Query(..., description="租户ID"),
    session: AsyncSession = Depends(get_db)
):
    """单次问答流式接口"""
    try:
        # 验证请求参数
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "问题不能为空"}
            )
        
        if not request.kb_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "知识库ID列表不能为空"}
            )
        
        # 执行问答
        response_generator = QAService.single_ask(
            session=session,
            request=request,
            user_id=user_id,
            tenant_id=tenant_id,
            is_stream=True
        )
        
        async def generate_stream():
            try:
                async for response in response_generator:
                    # 将响应转换为JSON字符串
                    json_str = json.dumps(response, ensure_ascii=False)
                    # 使用Server-Sent Events格式
                    yield f"data: {json_str}\n\n"
            except Exception as e:
                logging.error(f"流式问答失败: {e}")
                error_response = {
                    "answer": f"抱歉，处理您的问题时出现了错误：{str(e)}",
                    "reference": {"total": 0, "chunks": [], "doc_aggs": []},
                    "prompt": "",
                    "created_at": None
                }
                json_str = json.dumps(error_response, ensure_ascii=False)
                yield f"data: {json_str}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except Exception as e:
        logging.error(f"流式问答失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"流式问答失败: {str(e)}"}
        )


@router.post("/chat", response_model=QaResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Query(..., description="用户ID"),
    tenant_id: str = Query(..., description="租户ID"),
    session: AsyncSession = Depends(get_db)
):
    """多轮对话接口"""
    try:
        # 验证请求参数
        if not request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "消息列表不能为空"}
            )
        
        if not request.kb_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "知识库ID列表不能为空"}
            )
        
        # 验证最后一条消息必须是用户消息
        if request.messages[-1].role != "user":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "最后一条消息必须是用户消息"}
            )
        
        # 执行多轮对话
        response_generator = QAService.chat(
            session=session,
            messages=request.messages,
            user_id=user_id,
            kb_ids=request.kb_ids,
            doc_ids=request.doc_ids,
            system_prompt=request.system_prompt,
            top_n=request.top_n,
            similarity_threshold=request.similarity_threshold,
            vector_similarity_weight=request.vector_similarity_weight,
            top_k=request.top_k,
            enable_quote=request.enable_quote,
            enable_multiturn_refine=request.enable_multiturn_refine,
            target_language=request.target_language,
            enable_keyword_extraction=request.enable_keyword_extraction,
            use_kg=request.use_kg,
            tavily_api_key=request.tavily_api_key,
            temperature=request.temperature,
            is_stream=request.is_stream
        )
        
        # 获取最终响应
        final_response = None
        async for response in response_generator:
            final_response = response
        
        if not final_response:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"message": "对话服务返回空结果"}
            )
        
        return QaResponse(**final_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"多轮对话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"多轮对话失败: {str(e)}"}
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user_id: str = Query(..., description="用户ID"),
    tenant_id: str = Query(..., description="租户ID"),
    session: AsyncSession = Depends(get_db)
):
    """多轮对话流式接口"""
    try:
        # 验证请求参数
        if not request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "消息列表不能为空"}
            )
        
        if not request.kb_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "知识库ID列表不能为空"}
            )
        
        # 验证最后一条消息必须是用户消息
        if request.messages[-1].role != "user":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "最后一条消息必须是用户消息"}
            )
        
        # 执行多轮对话（强制流式）
        response_generator = QAService.chat(
            session=session,
            messages=request.messages,
            user_id=user_id,
            kb_ids=request.kb_ids,
            doc_ids=request.doc_ids,
            system_prompt=request.system_prompt,
            top_n=request.top_n,
            similarity_threshold=request.similarity_threshold,
            vector_similarity_weight=request.vector_similarity_weight,
            top_k=request.top_k,
            enable_quote=request.enable_quote,
            enable_multiturn_refine=request.enable_multiturn_refine,
            target_language=request.target_language,
            enable_keyword_extraction=request.enable_keyword_extraction,
            use_kg=request.use_kg,
            tavily_api_key=request.tavily_api_key,
            temperature=request.temperature,
            is_stream=True
        )
        
        async def generate_stream():
            try:
                async for response in response_generator:
                    # 将响应转换为JSON字符串
                    json_str = json.dumps(response, ensure_ascii=False)
                    # 使用Server-Sent Events格式
                    yield f"data: {json_str}\n\n"
            except Exception as e:
                logging.error(f"流式对话失败: {e}")
                error_response = {
                    "answer": f"抱歉，处理您的问题时出现了错误：{str(e)}",
                    "reference": {"total": 0, "chunks": [], "doc_aggs": []},
                    "prompt": "",
                    "created_at": None
                }
                json_str = json.dumps(error_response, ensure_ascii=False)
                yield f"data: {json_str}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"流式对话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"流式对话失败: {str(e)}"}
        )
