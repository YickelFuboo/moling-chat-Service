from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin


# 租户成员关联表
tenant_members = Table(
    'tenant_members',
    Base.metadata,
    Column('tenant_id', String(32), ForeignKey('tenants.id', ondelete='CASCADE'), primary_key=True),
    Column('user_id', String(32), nullable=False, primary_key=True, comment="用户ID"),
    Column('joined_at', DateTime, nullable=False, comment="加入时间")
)


class Tenant(Base, TimestampMixin):
    """租户模型"""
    __tablename__ = "tenants"
    
    # 主键
    id = Column(String(32), primary_key=True)
    
    # 租户名称
    name = Column(String(128), nullable=False, index=True, comment="租户名称")
    
    # 租户描述
    description = Column(Text, nullable=True, comment="租户描述")
    
    # 租户Owner ID
    owner_id = Column(String(32), nullable=False, index=True, comment="租户Owner用户ID")
    
    # 租户成员数量
    member_count = Column(Integer, default=1, index=True, comment="租户成员数量")
    
    # 租户状态，1表示有效，0表示无效
    status = Column(String(1), nullable=False, default="1", index=True, comment="状态(0:无效, 1:有效)")
    
    def __str__(self):
        return self.name 