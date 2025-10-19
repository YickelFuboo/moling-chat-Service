#!/usr/bin/env python3
"""
QAService功能测试运行脚本
用于测试不同条件下的问答服务功能
"""

import asyncio
import sys
import os
import logging
from unittest.mock import AsyncMock, MagicMock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_qa_service import run_tests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    """主函数"""
    print("🚀 启动QAService功能测试...")
    print("=" * 60)
    print("测试说明:")
    print("1. 基础单次问答功能")
    print("2. 带关键词提取的单次问答")
    print("3. 基础多轮对话功能")
    print("4. 带目标语言转换的多轮对话")
    print("5. 带Tavily外部搜索的多轮对话")
    print("6. 带知识图谱搜索的多轮对话")
    print("7. 知识库不存在的错误处理")
    print("8. 没有找到相关知识的错误处理")
    print("9. SQL查询功能")
    print("10. 修复错误引用格式功能")
    print("=" * 60)
    
    try:
        # 运行异步测试
        asyncio.run(run_tests())
        print("\n🎉 所有测试完成!")
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n💥 测试运行出错: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
