from datetime import datetime
from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.orm import selectinload
import uuid
from app.models import Tenant, tenant_members

class TenantService:
    """租户管理服务"""
    
    @staticmethod
    async def create_tenant(
        session: AsyncSession,
        name: str,
        description: Optional[str],
        owner_id: str
    ) -> Tenant:
        """创建租户"""
        try:
            # 检查租户名称是否已存在
            existing_tenant = await session.execute(
                select(Tenant).where(
                    and_(
                        Tenant.name == name,
                        Tenant.status == "1"
                    )
                )
            )
            
            if existing_tenant.scalar_one_or_none():
                raise ValueError("租户名称已存在")
            
            tenant_id = str(uuid.uuid4()).replace("-", "")
            tenant = Tenant(
                id=tenant_id,
                name=name,
                description=description,
                owner_id=owner_id,
                member_count=1
            )
            
            session.add(tenant)
            await session.flush() 
            
            # 添加Owner为租户成员
            await session.execute(
                tenant_members.insert().values(
                    tenant_id=tenant_id,
                    user_id=owner_id,
                    joined_at=datetime.utcnow()
                )
            )
            
            await session.commit()
            
            logging.info(f"租户创建成功: {tenant_id}")
            return tenant
            
        except Exception as e:
            await session.rollback()
            logging.error(f"创建租户失败: {e}")
            raise
    
    @staticmethod
    async def get_tenant_by_id(session: AsyncSession, tenant_id: str) -> Optional[Tenant]:
        """根据ID获取租户"""
        try:
            result = await session.execute(
                select(Tenant).where(Tenant.id == tenant_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logging.error(f"获取租户失败: {e}")
            raise
    
    @staticmethod
    async def update_tenant(
        session: AsyncSession,
        tenant_id: str,
        name: str,
        description: Optional[str],
        user_id: str
    ) -> None:
        """更新租户信息"""
        try:
            # 检查权限
            tenant = await TenantService.get_tenant_by_id(session, tenant_id)
            if not tenant:
                raise ValueError("租户不存在")
            if tenant.owner_id != user_id:
                raise ValueError("不是租户Owner")
            
            # 检查租户名称是否已存在（排除当前租户）
            existing_tenant = await session.execute(
                select(Tenant).where(
                    and_(
                        Tenant.name == name,
                        Tenant.status == "1",
                        Tenant.id != tenant_id
                    )
                )
            )
            
            if existing_tenant.scalar_one_or_none():
                raise ValueError("租户名称已存在，请使用其他名字")
            
            # 更新租户基本信息
            await session.execute(
                update(Tenant)
                .where(Tenant.id == tenant_id)
                .values(
                    name=name,
                    description=description,
                    updated_at=datetime.utcnow()
                )
            )
            
            await session.commit()
            logging.info(f"租户更新成功: {tenant_id}")
            
        except Exception as e:
            await session.rollback()
            logging.error(f"更新租户失败: {e}")
            raise
    
    @staticmethod
    async def delete_tenant(session: AsyncSession, tenant_id: str, user_id: str) -> None:
        """删除租户"""
        try:
            # 检查权限
            tenant = await TenantService.get_tenant_by_id(session, tenant_id)
            if not tenant:
                raise ValueError("租户不存在")
            if tenant.owner_id != user_id:
                raise ValueError("不是租户Owner")
            
            # 删除租户成员关系
            await session.execute(
                delete(tenant_members).where(tenant_members.c.tenant_id == tenant_id)
            )
            
            # 删除租户
            await session.execute(
                delete(Tenant).where(Tenant.id == tenant_id)
            )
            
            await session.commit()
            logging.info(f"租户删除成功: {tenant_id}")
            
        except Exception as e:
            await session.rollback()
            logging.error(f"删除租户失败: {e}")
            raise
    
    @staticmethod
    async def list_tenants(
        session: AsyncSession,
        owner_id: str,
        page_number: int = 1,
        items_per_page: int = 20,
        order_by: str = "created_at",
        desc: bool = True,
        keywords: Optional[str] = None
    ) -> Tuple[List[Tenant], int]:
        """获取租户列表"""
        try:
            # 构建查询条件
            query = select(Tenant).where(Tenant.owner_id == owner_id)
            
            if keywords:
                query = query.where(
                    or_(
                        Tenant.name.contains(keywords),
                        Tenant.description.contains(keywords)
                    )
                )
            
            # 获取总数
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(count_query)
            total_count = total_result.scalar()
            
            # 排序和分页
            if desc:
                query = query.order_by(getattr(Tenant, order_by).desc())
            else:
                query = query.order_by(getattr(Tenant, order_by))
            
            query = query.offset((page_number - 1) * items_per_page).limit(items_per_page)
            
            result = await session.execute(query)
            tenants = result.scalars().all()
            
            return list(tenants), total_count
            
        except Exception as e:
            logging.error(f"获取租户列表失败: {e}")
            raise
    
    @staticmethod
    async def add_member(
        session: AsyncSession,
        tenant_id: str,
        member_id: str,
        user_id: str
    ) -> None:
        """添加租户成员"""
        try:
            # 检查权限
            tenant = await TenantService.get_tenant_by_id(session, tenant_id)
            if not tenant:
                raise ValueError("租户不存在")
            if tenant.owner_id != user_id:
                raise ValueError("不是租户Owner")
            
            # 检查用户是否已经是成员
            existing_member = await session.execute(
                select(tenant_members).where(
                    and_(
                        tenant_members.c.tenant_id == tenant_id,
                        tenant_members.c.user_id == member_id
                    )
                )
            )
            
            if existing_member.scalar_one_or_none():
                return False
            
            # 添加成员
            await session.execute(
                tenant_members.insert().values(
                    tenant_id=tenant_id,
                    user_id=member_id,
                    joined_at=datetime.utcnow()
                )
            )
            
            # 更新成员数量
            await session.execute(
                update(Tenant)
                .where(Tenant.id == tenant_id)
                .values(member_count=Tenant.member_count + 1)
            )
            
            await session.commit()
            logging.info(f"添加租户成员成功: {tenant_id} - {user_id}")
            
        except Exception as e:
            await session.rollback()
            logging.error(f"添加租户成员失败: {e}")
            raise
    
    @staticmethod
    async def remove_member(
        session: AsyncSession,
        tenant_id: str,
        member_id: str,
        user_id: str
    ) -> None:
        """移除租户成员"""
        try:
            # 检查权限
            tenant = await TenantService.get_tenant_by_id(session, tenant_id)
            if not tenant:
                raise ValueError("租户不存在")
            if tenant.owner_id != user_id:
                raise ValueError("不是租户Owner")
            
            # 不能移除Owner
            if member_id == tenant.owner_id:
                raise ValueError("不能移除租户Owner")
            
            # 移除成员
            result = await session.execute(
                delete(tenant_members).where(
                    and_(
                        tenant_members.c.tenant_id == tenant_id,
                        tenant_members.c.user_id == member_id
                    )
                )
            )
            
            if result.rowcount == 0:
                raise ValueError("用户不是租户成员")
            
            # 更新成员数量
            await session.execute(
                update(Tenant)
                .where(Tenant.id == tenant_id)
                .values(member_count=Tenant.member_count - 1)
            )
            
            await session.commit()
            logging.info(f"移除租户成员成功: {tenant_id} - {user_id}")
            
        except Exception as e:
            await session.rollback()
            logging.error(f"移除租户成员失败: {e}")
            raise
    
    @staticmethod
    async def change_owner(
        session: AsyncSession,
        tenant_id: str,
        new_owner_id: str,
        current_owner_id: str
    ) -> None:
        """修改租户Owner"""
        try:
            # 检查权限
            tenant = await TenantService.get_tenant_by_id(session, tenant_id)
            if not tenant:
                raise ValueError("租户不存在")
            if tenant.owner_id != current_owner_id:
                raise ValueError("不是租户Owner，无法执行修改操作")
            
            # 检查新Owner是否是租户成员
            if not await TenantService.is_tenant_member(session, tenant_id, new_owner_id):
                raise ValueError("新Owner不是租户成员")
            
            # 更新Owner
            await session.execute(
                update(Tenant)
                .where(Tenant.id == tenant_id)
                .values(owner_id=new_owner_id)
            )
            
            await session.commit()
            logging.info(f"修改租户Owner成功: {tenant_id} - {new_owner_id}")
            
        except Exception as e:
            await session.rollback()
            logging.error(f"修改租户Owner失败: {e}")
            raise
    
    @staticmethod
    async def get_tenant_members(session: AsyncSession, tenant_id: str) -> List[str]:
        """获取租户成员ID列表"""
        try:
            result = await session.execute(
                select(tenant_members.c.user_id).where(tenant_members.c.tenant_id == tenant_id)
            )
            return [row[0] for row in result.fetchall()]
        except Exception as e:
            logging.error(f"获取租户成员失败: {e}")
            raise
    
    @staticmethod
    async def is_tenant_member(session: AsyncSession, tenant_id: str, user_id: str) -> bool:
        """检查用户是否是租户成员"""
        try:
            result = await session.execute(
                select(tenant_members).where(
                    and_(
                        tenant_members.c.tenant_id == tenant_id,
                        tenant_members.c.user_id == user_id
                    )
                )
            )
            return result.scalar_one_or_none() is not None
        except Exception as e:
            logging.error(f"检查租户成员失败: {e}")
            raise 