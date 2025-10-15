from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# 请求模型
class TenantRequest(BaseModel):
    """租户请求模型"""
    name: str = Field(..., description="租户名称", max_length=128)
    description: Optional[str] = Field(None, description="租户描述")


class ListTenantRequest(BaseModel):
    """获取租户列表请求模型"""
    page_number: int = Field(1, description="页码", ge=1)
    items_per_page: int = Field(20, description="每页数量", ge=1, le=100)
    order_by: str = Field("created_at", description="排序字段")
    desc: bool = Field(True, description="是否降序")
    keywords: Optional[str] = Field(None, description="搜索关键词")


class MembersRequest(BaseModel):
    """租户成员操作请求模型"""
    user_ids: List[str] = Field(..., description="用户ID列表")


class ChangeOwnerRequest(BaseModel):
    """修改租户Owner请求模型"""
    new_owner_id: str = Field(..., description="新Owner用户ID")


# 响应模型
class TenantResponse(BaseModel):
    """租户响应模型"""
    id: str
    name: str
    description: Optional[str]
    owner_id: str
    member_count: int
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class TenantDetailResponse(BaseModel):
    """租户详情响应模型"""
    id: str
    name: str
    description: Optional[str]
    owner_id: str
    member_count: int
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ListTenantResponse(BaseModel):
    """租户列表响应模型"""
    items: List[TenantResponse]
    total: int
    page_number: int
    items_per_page: int


class CreateTenantResponse(BaseModel):
    """创建租户响应模型"""
    tenant_id: str


class TenantOperationResponse(BaseModel):
    """租户操作响应模型"""
    success: bool
    message: str 