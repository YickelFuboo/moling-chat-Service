import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
import json
from sqlalchemy.ext.asyncio import AsyncSession
from app.infrastructure.database import get_db
from app.schemas.qa import SingleQaRequest, QaResponse, StreamQaResponse
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
        response_generator = QAService.qa(
            session=session,
            request=request,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        # 获取最终响应
        final_response = None
        for response in response_generator:
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
        
        # 强制设置为流式
        request.is_stream = True
        
        # 执行问答
        response_generator = QAService.qa(
            session=session,
            request=request,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        def generate_stream():
            try:
                for response in response_generator:
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
                    "audio_binary": None,
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
