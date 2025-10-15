from sqlalchemy.ext.asyncio import AsyncSession
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status, Request
from app.constants.common import TenantConstants
from app.infrastructure.database import get_db
from app.schemas.tenant import (
    TenantRequest,
    ListTenantRequest,
    MembersRequest,
    ChangeOwnerRequest,
    TenantResponse,
    TenantDetailResponse,
    ListTenantResponse,
    CreateTenantResponse,
    TenantOperationResponse
)
from app.services.tenant_service import TenantService

router = APIRouter(prefix="/api/tenants", tags=["租户管理"])

@router.post("/create", response_model=CreateTenantResponse)
async def create_tenant(
    request: TenantRequest,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """创建租户"""
    try:
        # 验证租户名称
        if not request.name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "租户名称不能为空"}
            )
        
        if len(request.name.encode("utf-8")) > TenantConstants.TENANT_NAME_MAX_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "租户名称长度不能超过128字节"}
            )

        if request.description and len(request.description.encode("utf-8")) > TenantConstants.TENANT_DESCRIPTION_MAX_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "租户描述长度不能超过1000字节"}
            )
        
        # 创建租户
        tenant = await TenantService.create_tenant(
            session=session,
            name=request.name.strip(),
            description=request.description,
            owner_id=user_id
        )
        
        return CreateTenantResponse(tenant_id=tenant.id)
        
    except Exception as e:
        logging.error(f"创建租户失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"创建租户失败: {str(e)}"}
        )

@router.post("/update/{tenant_id}")
async def update_tenant(
    tenant_id: str,
    request: TenantRequest,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """更新租户信息"""
    try:
        # 验证租户名称
        if not request.name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "租户名称不能为空"}
            )
        
        if len(request.name.encode("utf-8")) > TenantConstants.TENANT_NAME_MAX_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "租户名称长度不能超过128字节"}
            )
        
        if request.description and len(request.description.encode("utf-8")) > TenantConstants.TENANT_DESCRIPTION_MAX_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "租户描述长度不能超过1000字节"}
            )
        
        # 更新租户
        await TenantService.update_tenant(
            session=session,
            tenant_id=tenant_id,
            name=request.name.strip(),
            description=request.description,
            user_id=user_id
        )
        
        return {"message": "租户更新成功"}
        
    except Exception as e:
        logging.error(f"更新租户失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"更新租户失败: {str(e)}"}
        )

@router.post("/delete/{tenant_id}")
async def delete_tenant(
    tenant_id: str,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """删除租户"""
    try:
        # 删除租户
        await TenantService.delete_tenant(
            session=session,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        return {"message": "租户删除成功"}
    
    except Exception as e:
        logging.error(f"删除租户失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"删除租户失败: {str(e)}"}
        )

@router.post("/list", response_model=ListTenantResponse)
async def list_tenants(
    list_request: ListTenantRequest,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """获取租户列表"""
    try:
        # 获取租户列表
        tenants, total_count = await TenantService.list_tenants(
            session=session,
            owner_id=user_id,
            page_number=list_request.page_number,
            items_per_page=list_request.items_per_page,
            order_by=list_request.order_by,
            desc=list_request.desc,
            keywords=list_request.keywords
        )
        
        # 转换为响应模型
        tenant_responses = [TenantResponse.from_orm(tenant) for tenant in tenants]
        
        return ListTenantResponse(
            items=tenant_responses,
            total=total_count,
            page_number=list_request.page_number,
            items_per_page=list_request.items_per_page
        )
        
    except Exception as e:
        logging.error(f"获取租户列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"获取租户列表失败: {str(e)}"}
        )

@router.get("/detail/{tenant_id}", response_model=TenantDetailResponse)
async def get_tenant_detail(
    tenant_id: str,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """获取租户详情"""
    try:
        # 获取租户详情
        tenant = await TenantService.get_tenant_by_id(session, tenant_id)
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "租户不存在"}
            )
        
        # 检查权限
        if tenant.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"message": "无权限操作此租户"}
            )
        
        return TenantDetailResponse.from_orm(tenant)
        
    except Exception as e:
        logging.error(f"获取租户详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"获取租户详情失败: {str(e)}"}
        )

@router.post("/{tenant_id}/add-members", response_model=TenantOperationResponse)
async def add_tenant_members(
    tenant_id: str,
    request: MembersRequest,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """添加租户成员"""
    try:
        # 验证请求
        if not request.user_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "用户ID列表不能为空"}
            )
        
        success_count = 0
        failed_count = 0
        
        # 批量添加成员
        for member_id in request.user_ids:
            try:
                await TenantService.add_member(
                    session=session,
                    tenant_id=tenant_id,
                    member_id=member_id,
                    user_id=user_id
                )
                success_count += 1
            except Exception as e:
                failed_count += 1
                logging.error(f"添加租户成员失败: {tenant_id} - {user_id} - {e}")
        
        if failed_count == 0:
            message = f"成功添加 {success_count} 个成员"
        else:
            message = f"成功添加 {success_count} 个成员，失败 {failed_count} 个"
        
        return TenantOperationResponse(
            success=success_count > 0,
            message=message
        )
        
    except Exception as e:
        logging.error(f"添加租户成员失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"添加租户成员失败: {str(e)}"}
        )

@router.post("/{tenant_id}/remove-members", response_model=TenantOperationResponse)
async def remove_tenant_members(
    tenant_id: str,
    request: MembersRequest,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """移除租户成员"""
    try:
        # 验证请求
        if not request.user_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "用户ID列表不能为空"}
            )
        
        success_count = 0
        failed_count = 0
        
        # 批量移除成员
        for member_id in request.user_ids:
            try:
                await TenantService.remove_member(
                    session=session,
                    tenant_id=tenant_id,
                    member_id=member_id,
                    user_id=user_id
                )
                success_count += 1
            except Exception as e:
                failed_count += 1
                logging.error(f"移除租户成员失败: {tenant_id} - {user_id} - {e}")
        
        if failed_count == 0:
            message = f"成功移除 {success_count} 个成员"
        else:
            message = f"成功移除 {success_count} 个成员，失败 {failed_count} 个"
        
        return TenantOperationResponse(
            success=success_count > 0,
            message=message
        )
        
    except Exception as e:
        logging.error(f"移除租户成员失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"移除租户成员失败: {str(e)}"}
        )

@router.post("/{tenant_id}/change-owner", response_model=TenantOperationResponse)
async def change_tenant_owner(
    tenant_id: str,
    request: ChangeOwnerRequest,
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """修改租户Owner"""
    try:
        # 修改Owner
        await TenantService.change_owner(
            session=session,
            tenant_id=tenant_id,
            new_owner_id=request.new_owner_id,
            current_owner_id=user_id
        )
        
        return TenantOperationResponse(
            success=True,
            message="租户Owner修改成功"
        )
        
    except Exception as e:
        logging.error(f"修改租户Owner失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"修改租户Owner失败: {str(e)}"}
        )