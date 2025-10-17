import os
import urllib.parse
from datetime import datetime
from typing import List, Optional, Union
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status
from fastapi.responses import Response
from app.infrastructure.database import get_db
from app.schemes.document import (
    FileUploadResult,
    DocumentResponse,
    ListDocumentResponse,
    DocumentChunksResponse,
    UpdateDocumentRequest,
    UpdateDocumentMetaFieldsRequest,
    ParserResult,
)
from app.models.document import ProcessStatus
from app.services.kb_service import KBService
from app.services.doc_service import DocumentService
from app.services.doc_parser_service import DocParserService
from app.services.common.file_service import FileService, FileUsage


router = APIRouter(prefix="/api/documents", tags=["文档管理"])

# 上传文件
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
            detail={"message": f"上传文档失败: {str(e)}"}
        )

@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document_detail(
    doc_id: str,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """获取文档详情"""
    try:
        # 获取文档信息
        document = await DocumentService.get_document_by_id(session, doc_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "文档不存在"}
            )
        
        # 获取chunk数量
        chunk_count = await DocumentService.get_document_chunk_count(session, doc_id)
        
        # 构建响应
        response_data = {
            "id": document.id,
            "kb_id": document.kb_id,
            "name": document.name,
            "description": document.description,
            "type": document.type,
            "suffix": document.suffix,
            "file_id": document.file_id,
            "size": document.size,
            "thumbnail": document.thumbnail_id,
            "parser_id": document.parser_id,
            "parser_config": document.parser_config,
            "source_type": document.source_type,
            "process_status": document.process_status,
            "chunk_count": chunk_count,
            "created_by": document.created_by,
            "created_at": document.created_at,
            "updated_at": document.updated_at
        }
        
        return DocumentResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"获取文档详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"获取文档详情失败: {str(e)}"}
        )

@router.put("/{doc_id}", response_model=DocumentResponse)
async def update_document(
    doc_id: str,
    request: UpdateDocumentRequest,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """更新文档信息（名称和描述）"""
    try:
        # 获取文档信息
        document = await DocumentService.get_document_by_id(session, doc_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "文档不存在"}
            )
        
        # 验证权限
        if document.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"message": "无权限更新此文档"}
            )
        
        # 更新文档信息
        updated_document = await DocumentService.update_document(
            session=session,
            doc_id=doc_id,
            description=request.description
        )
        
        # 获取chunk数量
        chunk_count = await DocumentService.get_document_chunk_count(session, doc_id)
        
        # 构建响应
        response_data = {
            "id": updated_document.id,
            "kb_id": updated_document.kb_id,
            "name": updated_document.name,
            "description": updated_document.description,
            "type": updated_document.type,
            "suffix": updated_document.suffix,
            "file_id": updated_document.file_id,
            "size": updated_document.size,
            "thumbnail": updated_document.thumbnail_id,
            "parser_id": updated_document.parser_id,
            "parser_config": updated_document.parser_config,
            "source_type": updated_document.source_type,
            "process_status": updated_document.process_status,
            "chunk_count": chunk_count,
            "created_by": updated_document.created_by,
            "created_at": updated_document.created_at,
            "updated_at": updated_document.updated_at
        }
        
        return DocumentResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"更新文档失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"更新文档失败: {str(e)}"}
        )

@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """删除文档及其相关数据"""
    try:
        # 获取文档信息
        document = await DocumentService.get_document_by_id(session, doc_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "文档不存在"}
            )
        
        # 验证权限
        if document.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"message": "无权限删除此文档"}
            )
        
        # 删除文档及相关数据
        await DocumentService.delete_document_by_id(session, doc_id)
        
        return {"message": "文档删除成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"删除文档失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"删除文档失败: {str(e)}"}
        )


@router.get("/{doc_id}/download")
async def download_document(
    doc_id: str,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """下载文档文件"""
    try:
        # 获取文档信息
        document = await DocumentService.get_document_by_id(session, doc_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "文档不存在"}
            )
        
        # 验证权限
        if document.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"message": "无权限下载此文档"}
            )
        
        # 检查文件是否存在
        if not document.file_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "文档文件不存在"}
            )
        
        # 从文件存储中获取文件内容
        file_content = await FileService.get_file_content(document.file_id, FileUsage.DOCUMENT)
        
        # 设置响应头，处理中文文件名编码
        encoded_filename = urllib.parse.quote(document.name.encode('utf-8'))
        headers = {
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}",
            "Content-Type": "application/octet-stream"
        }
        
        return Response(
            content=file_content,
            headers=headers,
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"下载文档失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"下载文档失败: {str(e)}"}
        )

# 替换文件
@router.post("/{doc_id}/reload", response_model=DocumentResponse)
async def reload_document(
    doc_id: str,
    file: UploadFile = File(..., description="新文件"),
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """重新加载文档（替换文件）"""
    try:
        # 获取原文档信息
        document = await DocumentService.get_document_by_id(session, doc_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "文档不存在"}
            )
        
        # 验证权限
        if document.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"message": "无权限重新加载此文档"}
            )
        
        # 重新加载文档（删除旧文档，创建新文档）
        reloaded_document = await DocumentService.update_document_file(
            session=session,
            doc_id=doc_id,
            file=file,
            created_by=user_id
        )
        
        # 构建响应
        response_data = {
            "id": reloaded_document.id,
            "kb_id": reloaded_document.kb_id,
            "name": reloaded_document.name,
            "description": reloaded_document.description,
            "type": reloaded_document.type,
            "suffix": reloaded_document.suffix,
            "file_id": reloaded_document.file_id,
            "size": reloaded_document.size,
            "thumbnail": reloaded_document.thumbnail_id,
            "parser_id": reloaded_document.parser_id,
            "parser_config": reloaded_document.parser_config,
            "source_type": reloaded_document.source_type,
            "process_status": reloaded_document.process_status,
            "chunk_count": 0, #初始状态，直接返回0
            "created_by": reloaded_document.created_by,
            "created_at": reloaded_document.created_at,
            "updated_at": reloaded_document.updated_at
        }
        
        return DocumentResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"重新加载文档失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"重新加载文档失败: {str(e)}"}
        )

@router.get("/kb/{kb_id}/list", response_model=ListDocumentResponse)
async def get_documents_by_kb(
    kb_id: str,
    page: int = Query(1, description="页码", ge=1),
    page_size: int = Query(20, description="每页数量", ge=1, le=100),
    keywords: Optional[str] = Query(None, description="搜索关键词"),
    order_by: str = Query("created_at", description="排序字段"),
    desc: bool = Query(True, description="是否降序"),
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """获取知识库下的文档列表"""
    try:        
        # 获取文档列表
        documents, total = await DocumentService.get_documents_by_kb_id(
            session=session,
            kb_id=kb_id,
            page=page,
            page_size=page_size,
            keywords=keywords or "",
            order_by=order_by,
            desc_order=desc
        )
        
        # 构建响应数据
        document_responses = []
        for doc in documents:
            chunk_count = await DocumentService.get_document_chunk_count(session, doc.id)
            doc_data = {
                "id": doc.id,
                "kb_id": doc.kb_id,
                "name": doc.name,
                "description": doc.description,
                "type": doc.type,
                "suffix": doc.suffix,
                "file_id": doc.file_id,
                "size": doc.size,
                "thumbnail": doc.thumbnail_id,
                "parser_id": doc.parser_id,
                "parser_config": doc.parser_config,
                "source_type": doc.source_type,
                "process_status": doc.process_status,
                "chunk_count": chunk_count,
                "created_by": doc.created_by,
                "created_at": doc.created_at,
                "updated_at": doc.updated_at
            }
            document_responses.append(doc_data)
        
        return ListDocumentResponse(
            documents=document_responses,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"获取文档列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "获取文档列表失败"}
        )

@router.post("/{doc_id}/parse", response_model=ParserResult)
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
        
        # 获取知识库信息
        kb = await KBService.get_kb_by_id(session, document.kb_id)
        if not kb:
            raise ValueError("知识库不存在")

        # 先删除Document对应的Chunk，然后更新Document状态为init
        await DocumentService.update_document_status(session, doc_id, ProcessStatus.INIT)

        # 启动Document解析任务
        docparser = DocParserService(session, kb, document, user_id)
        result = await docparser.parse_document()

        # 更新文档状态
        await DocumentService.update_document_status(session, doc_id, ProcessStatus.PARSED if result else ProcessStatus.FAILED)
        
        return ParserResult(doc_id=doc_id, success=True)
        
    except Exception as e:
        logging.error(f"解析文档失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"解析文档失败: {str(e)}"}
        )

@router.get("/{doc_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(
    doc_id: str,
    with_vector: bool = Query(False, description="是否返回向量"),
    page: int = Query(1, description="页码", ge=1),
    page_size: int = Query(20, description="每页数量", ge=1, le=100),
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """获取文档的切片列表"""
    try:        
        # 获取切片列表
        chunks, total = await DocumentService.get_document_chunks(
            session=session,
            doc_id=doc_id,
            with_vector=with_vector,
            page=page,
            page_size=page_size
        )
        
        return DocumentChunksResponse(
            chunks=chunks,
            total=total,
            page=page,
            page_size=page_size
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"获取文档切片列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"获取文档切片列表失败: {str(e)}"}
        )

@router.post("/chunks/batch", response_model=DocumentChunksResponse)
async def get_documents_chunks_batch(
    doc_ids: Union[str] = Form(..., description="文档ID列表，可以是逗号分隔的字符串或列表"),
    with_vector: bool = Query(False, description="是否返回向量"),
    page: int = Query(1, description="页码", ge=1),
    page_size: int = Query(20, description="每页数量", ge=1, le=100),
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """批量获取多个文档的切片列表"""
    try:
        if not doc_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "文档ID列表不能为空"}
            )

        # 解析逗号分隔的文档ID字符串
        doc_ids = [doc_id.strip() for doc_id in doc_ids.split(',') if doc_id.strip()]

        # 批量获取chunks
        chunks, total = await DocumentService.get_documents_chunks(
            session=session,
            doc_ids=doc_ids,
            with_vector=with_vector,
            page=page,
            page_size=page_size
        )
        
        return DocumentChunksResponse(
            chunks=chunks,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        # 处理业务逻辑错误（如文档不在同一知识库）
        logging.warning(f"批量获取文档切片列表业务错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": str(e)}
        )
    except Exception as e:
        logging.error(f"批量获取文档切片列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"批量获取文档切片列表失败: {str(e)}"}
        )


@router.put("/{doc_id}/meta-fields", response_model=DocumentResponse)
async def update_document_meta_fields(
    doc_id: str,
    request: UpdateDocumentMetaFieldsRequest,
    session: AsyncSession = Depends(get_db)
):
    """更新文档的元数据字段"""
    try:       
        # 更新meta_fields
        updated_document = await DocumentService.update_document_meta_fields(
            session, doc_id, request.meta_fields
        )
        return updated_document
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"更新文档元数据字段失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"更新文档元数据字段失败: {str(e)}"}
        )