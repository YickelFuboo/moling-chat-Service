from datetime import datetime
from enum import Enum
from sqlalchemy import Column, String, DateTime, Integer, Float, Text, Boolean, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from .base import Base


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"      # 等待中
    RUNNING = "running"      # 执行中
    SUCCESS = "success"      # 成功
    FAILED = "failed"        # 失败
    RETRYING = "retrying"    # 重试中
    CANCELLED = "cancelled"  # 已取消


class TaskType(str, Enum):
    """任务类型枚举"""
    DOCUMENT_PARSE = "document_parse"      # 文档解析



class Task(Base):
    """通用任务表"""
    __tablename__ = "tasks"
    
    id = Column(String(36), primary_key=True, comment="任务ID")
    task_type = Column(String(50), nullable=False, comment="任务类型")
    parent_id = Column(String(36), nullable=False, comment="父级ID")
    task_id = Column(String(255), nullable=False, comment="Celery任务ID")
    status = Column(String(20), default=TaskStatus.PENDING, comment="任务状态")
    progress = Column(Float, default=0.0, comment="任务进度")
    retry_count = Column(Integer, default=0, comment="重试次数")
    max_retries = Column(Integer, default=3, comment="最大重试次数")
    error_message = Column(Text, comment="错误信息")
    result = Column(JSON, comment="任务结果")
    task_params = Column(JSON, comment="任务参数")
    depends_on = Column(String(36), ForeignKey("tasks.id"), nullable=True, comment="依赖任务ID")
    priority = Column(Integer, default=0, comment="任务优先级")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    # 关联关系
    dependent_tasks = relationship("Task", back_populates="depends_on_task")
    depends_on_task = relationship("Task", remote_side=[id], back_populates="dependent_tasks")
    
    # 索引
    __table_args__ = (
        # 复合索引，用于快速查询特定类型的任务
        Index('idx_task_type_parent', 'task_type', 'parent_id'),
        # 状态索引，用于查询特定状态的任务
        Index('idx_task_status', 'status'),
        # 创建时间索引，用于排序
        Index('idx_task_created_at', 'created_at'),
    ) 