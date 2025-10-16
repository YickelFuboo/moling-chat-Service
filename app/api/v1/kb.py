from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status, Request
from app.constants.common import KBConstants, PDFParser, KnowledgeGraphMethod
from app.infrastructure.database import get_db
from app.models import KB
from app.services.kb_service import KBService
from app.schemes.kb import (
    CreateKBRequest,
    UpdateKBRequest,
    ListKBRequest,
    KBResponse,
    KBDetailResponse,
    ListKBResponse,
    CreateKBResponse,
    ParserConfigRequest,
    ParserConfigResponse,
    ParserConfigTemplateResponse
)

 
router = APIRouter(prefix="/api/knowledgebase", tags=["知识库管理"])

# API接口
@router.post("/create", response_model=CreateKBResponse)
async def create_kb(
    request: CreateKBRequest,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """创建知识库"""
    try:
        # 验证知识库名称
        if not request.name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "知识库名称不能为空"}
            )
        
        if len(request.name.encode("utf-8")) > KBConstants.NAME_MAX_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "知识库名称长度不能超过128字节"}
            )
        
        # 验证知识库描述长度
        if request.description and len(request.description.encode("utf-8")) > KBConstants.DESCRIPTION_MAX_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "知识库描述长度不能超过1000字节"}
            )

        # 创建知识库
        kb = await KBService.create_kb(
            session=session,
            name=request.name.strip(),
            owner_id=user_id,
            description=request.description,
            language=request.language,
            tenant_id=request.tenant_id,  
            embd_provider_name=request.embd_provider_name,
            embd_model_name=request.embd_model_name,
            rerank_provider_name=request.rerank_provider_name,
            rerank_model_name=request.rerank_model_name,
            page_rank=request.page_rank
        )
        
        return CreateKBResponse(kb_id=kb.id)
        
    except Exception as e:
        logging.error(f"创建知识库失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"创建知识库失败: {str(e)}"}
        )

@router.post("/update")
async def update_kb(
    request: UpdateKBRequest,
    user_id: str = Query(..., description="用户ID"),
    kb_id: str = Query(..., description="知识库ID"),
    session: AsyncSession = Depends(get_db)
):
    """更新知识库"""
    try:
        # 验证知识库名称
        if request.name and not request.name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "知识库名称不能为空"}
            )
        
        if request.name and len(request.name.encode("utf-8")) > KBConstants.NAME_MAX_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "知识库名称长度不能超过128字节"}
            )
        
        # 验证知识库描述长度
        if request.description and len(request.description.encode("utf-8")) > KBConstants.DESCRIPTION_MAX_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "知识库描述长度不能超过1000字节"}
            )
        
        # 更新知识库
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name.strip()
        if request.description is not None:
            update_data["description"] = request.description        
        if request.language is not None:
            update_data["language"] = request.language
        if request.tenant_id is not None:
            update_data["tenant_id"] = request.tenant_id
        if request.embd_provider_name is not None:
            update_data["embd_provider_name"] = request.embd_provider_name
        if request.embd_model_name is not None:
            update_data["embd_model_name"] = request.embd_model_name
        if request.rerank_provider_name is not None:
            update_data["rerank_provider_name"] = request.rerank_provider_name
        if request.rerank_model_name is not None:
            update_data["rerank_model_name"] = request.rerank_model_name
        if request.page_rank is not None:
            update_data["page_rank"] = request.page_rank        
        
        await KBService.update_kb(
            session=session,
            kb_id=kb_id,
            update_data=update_data,
            owner_id=user_id
        )
        
        return {"message": "知识库更新成功"}
        
    except Exception as e:
        logging.error(f"更新知识库失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"更新知识库失败: {str(e)}"}
        )

@router.get("/detail", response_model=KBDetailResponse)
async def get_kb_detail(
    kb_id: str = Query(..., description="知识库ID"),
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """获取知识库详情"""
    try:
        kb_detail = await KBService.get_kb_detail(session, kb_id)
        
        if not kb_detail:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "知识库不存在"}
            )
        
        return KBDetailResponse(**kb_detail)
        
    except Exception as e:
        logging.error(f"获取知识库详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"获取知识库详情失败: {str(e)}"}
        )

@router.post("/list", response_model=ListKBResponse)
async def list_kbs(
    list_request: ListKBRequest,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """获取知识库列表"""
    try:
        kb_list, total_count = await KBService.list_kbs(
            session=session,
            owner_id=user_id,
            page_number=list_request.page_number,
            items_per_page=list_request.items_per_page,
            order_by=list_request.order_by,
            desc_order=list_request.desc,
            keywords=list_request.keywords
        )
        
        return ListKBResponse(
            items=kb_list,
            total=total_count,
            page_number=list_request.page_number,
            items_per_page=list_request.items_per_page
        )
        
    except Exception as e:
        logging.error(f"获取知识库列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"获取知识库列表失败: {str(e)}"}
        )

@router.post("/delete")
async def delete_kb(
    kb_id: str = Query(..., description="知识库ID"),
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """删除知识库"""
    try:
        # 删除知识库
        await KBService.delete_kb(
            session=session,
            kb_id=kb_id,
            owner_id=user_id
        )
        
        return {"message": "知识库删除成功"}
        
    except Exception as e:
        logging.error(f"删除知识库失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"删除知识库失败: {str(e)}"}
        )

@router.get("/{kb_id}/parser-config", response_model=ParserConfigResponse)
async def get_kb_parser_config(
    kb_id: str,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """获取知识库解析器配置"""
    try:
        # 检查知识库是否存在
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
        
        return ParserConfigResponse(
            parser_id=kb.parser_id,
            parser_config=kb.parser_config
        )
        
    except Exception as e:
        logging.error(f"获取知识库解析器配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"获取知识库解析器配置失败: {str(e)}"}
        )

@router.put("/{kb_id}/parser-config", response_model=ParserConfigResponse)
async def update_kb_parser_config(
    kb_id: str,
    request: ParserConfigRequest,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """更新知识库解析器配置"""
    try:        
        # 更新配置
        update_data = {}
        if request.parser_id is not None:
            update_data["parser_id"] = request.parser_id
        if request.parser_config is not None:
            update_data["parser_config"] = request.parser_config
        
        await KBService.update_kb(
            session=session,
            kb_id=kb_id,
            update_data=update_data,
            owner_id=user_id
        )
        
        # 返回更新后的配置
        updated_kb = await KBService.get_kb_by_id(session, kb_id)
        return ParserConfigResponse(
            parser_id=updated_kb.parser_id,
            parser_config=updated_kb.parser_config
        )
        
    except Exception as e:
        logging.error(f"更新知识库解析器配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"更新知识库解析器配置失败: {str(e)}"}
        )

@router.get("/parser-config-template", response_model=ParserConfigTemplateResponse)
async def get_parser_config_template():
    """获取解析器配置模板和字段说明"""
    return ParserConfigTemplateResponse(
        template=KBConstants.DEFAULT_PARSER_CONFIG,
        field_descriptions=KBConstants.PARSER_CONFIG_FIELD_DESCRIPTIONS
    ) 