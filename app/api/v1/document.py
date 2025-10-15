import os
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status
from app.infrastructure.database import get_db
from app.schemas.document import (
    FileUploadResult,
    ParseTaskResponse,
)
from app.services.document_service import DocumentService
from app.services.kb_service import KBService
from app.tasks.document_tasks import parse_document_task


router = APIRouter(prefix="/api/documents", tags=["文档管理"])

@router.post("/upload", response_model=list[FileUploadResult])
async def upload_documents(
    kb_id: str = Form(..., description="知识库ID"),
    files: List[UploadFile] = File(..., description="上传的文件"),
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """上传文档到知识库"""
    try:
        # 验证知识库是否存在
        kb = await KBService.get_kb_by_id(session, kb_id)
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "知识库不存在"}
            )
        
        # 检查权限
        if kb.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"message": "无权限操作此知识库"}
            )
        
        # 批量处理文件上传
        results = []
        for file in files:
            result = await DocumentService.upload_document_to_kb(
                session=session,
                kb=kb,
                file=file,
                created_by=user_id
            )
            results.append(result)
        
        return results
        
    except Exception as e:
        logging.error(f"上传文档失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "上传文档失败"}
        )

@router.post("/{doc_id}/parse", response_model=ParseTaskResponse)
async def parse_document(
    doc_id: str,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db),
):
    """解析文档内容"""
    try:
        # 验证文档是否存在且用户有权限
        document = await DocumentService.get_document_by_id(session, doc_id)
        if not document:
            raise ValueError("文档不存在")
        
        if document.created_by != user_id:
            raise ValueError("不是文档Owner")
        
        # 创建解析任务
        task = await DocumentService.create_parse_task(session, doc_id, user_id)
        
        # 使用Celery异步执行解析任务
        celery_task = parse_document_task.delay(
            doc_id=doc_id,
            task_id=task.id,
            user_id=user_id
        )
        
        # 更新任务的Celery任务ID
        await DocumentService.update_task_celery_id(session, task.id, celery_task.id)
        
        return ParseTaskResponse(
            task_id=task.id,
            celery_task_id=celery_task.id,
            status="pending",
            message="文档解析任务已创建并开始执行"
        )
        
    except Exception as e:
        logging.error(f"解析文档失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "解析文档失败"}
        )
        raise
