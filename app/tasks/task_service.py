from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from sqlalchemy import select, and_, update
import uuid
from app.models.task import Task, TaskStatus, TaskType

class TaskService:
    """通用任务服务类"""
    
    @staticmethod
    async def create_task(
        session: AsyncSession,
        task_type: str,
        parent_id: str,
        task_data: Dict[str, Any]
    ) -> str:
        """创建单个任务"""
        try:
            task = Task(
                id=str(uuid.uuid4()),
                task_type=task_type,
                parent_id=parent_id,
                status=TaskStatus.PENDING,
                priority=task_data.get("priority", 0),
                depends_on=task_data.get("depends_on", None),
                task_params=task_data.get("task_params", {})
            )
            
            session.add(task)
            await session.commit()
            
            logging.info(f"为 {task_type} 类型的父级 {parent_id} 创建了任务: {task.id}")
            return task.id
            
        except Exception as e:
            await session.rollback()
            logging.error(f"创建任务失败: {e}")
            raise
    
    @staticmethod
    async def update_task_status(
        session: AsyncSession,
        task_type: str,
        parent_id: str,
        task_id: str,
        status: TaskStatus,
        progress: float = None,
        error_message: str = None,
        result: Dict[str, Any] = None
    ):
        """更新任务状态"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if progress is not None:
                update_data["progress"] = progress
            if error_message is not None:
                update_data["error_message"] = error_message
            if result is not None:
                update_data["result"] = result
            
            # 通过 task_type + parent_id + task_id 唯一标识任务
            await session.execute(
                update(Task)
                .where(
                    and_(
                        Task.task_type == task_type,
                        Task.parent_id == parent_id,
                        Task.id == task_id
                    )
                )
                .values(**update_data)
            )
            await session.commit()
            
        except Exception as e:
            logging.error(f"更新任务状态失败: {e}")
            await session.rollback()
            raise
    
    @staticmethod
    async def get_tasks_by_parent(
        session: AsyncSession,
        task_type: str,
        parent_id: str
    ) -> List[Task]:
        """获取指定父级的所有任务"""
        try:
            result = await session.execute(
                select(Task).where(
                    and_(
                        Task.task_type == task_type,
                        Task.parent_id == parent_id
                    )
                )
            )
            return result.scalars().all()
        except Exception as e:
            logging.error(f"获取父级任务失败: {e}")
            raise
    
    @staticmethod
    async def get_task_by_id(
        session: AsyncSession,
        task_id: str
    ) -> Optional[Task]:
        """根据ID获取任务"""
        try:
            result = await session.execute(
                select(Task).where(Task.id == task_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logging.error(f"获取任务失败: {e}")
            raise
    
    @staticmethod
    async def check_overall_progress(
        session: AsyncSession,
        task_type: str,
        parent_id: str
    ) -> float:
        """检查指定父级的整体处理进度"""
        try:
            tasks = await TaskService.get_tasks_by_parent(session, task_type, parent_id)
            
            if not tasks:
                return 0.0
            
            # 计算整体进度
            total_tasks = len(tasks)
            completed_tasks = len([t for t in tasks if t.status == TaskStatus.SUCCESS])
            overall_progress = completed_tasks / total_tasks
            
            logging.info(f"{task_type} 类型的父级 {parent_id} 整体进度: {overall_progress:.2%} ({completed_tasks}/{total_tasks})")
            return overall_progress
            
        except Exception as e:
            logging.error(f"检查整体进度失败: {e}")
            return 0.0
    
    @staticmethod
    async def increment_retry_count(
        session: AsyncSession,
        task_type: str,
        parent_id: str,
        task_id: str
    ):
        """增加任务重试次数"""
        try:
            await session.execute(
                update(Task)
                .where(
                    and_(
                        Task.task_type == task_type,
                        Task.parent_id == parent_id,
                        Task.id == task_id
                    )
                )
                .values(
                    retry_count=Task.retry_count + 1,
                    updated_at=datetime.utcnow()
                )
            )
            await session.commit()
            
        except Exception as e:
            logging.error(f"增加重试次数失败: {e}")
            await session.rollback()
            raise
    
    @staticmethod
    async def get_tasks_by_status(
        session: AsyncSession,
        status: TaskStatus,
        task_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Task]:
        """根据状态获取任务"""
        try:
            query = select(Task).where(Task.status == status)
            
            if task_type:
                query = query.where(Task.task_type == task_type)
            
            if limit:
                query = query.limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
        except Exception as e:
            logging.error(f"根据状态获取任务失败: {e}")
            raise
    
    @staticmethod
    async def cancel_tasks(
        session: AsyncSession,
        task_type: str,
        parent_id: str
    ):
        """取消指定父级的所有任务"""
        try:
            await session.execute(
                update(Task)
                .where(
                    and_(
                        Task.task_type == task_type,
                        Task.parent_id == parent_id,
                        Task.status.in_([TaskStatus.PENDING, TaskStatus.RUNNING])
                    )
                )
                .values(
                    status=TaskStatus.CANCELLED,
                    updated_at=datetime.utcnow()
                )
            )
            await session.commit()
            
            logging.info(f"已取消 {task_type} 类型的父级 {parent_id} 的所有任务")
            
        except Exception as e:
            logging.error(f"取消任务失败: {e}")
            await session.rollback()
            raise 