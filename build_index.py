"""
build_index.py — 一次性索引构建脚本

运行方式：
  python build_index.py

作用：
  读取 knowledge_base/ 下所有 .txt 文件 → 切块 → 向量化 → 存入 Chroma

注意：
  - 只需要运行一次（或知识库有更新时重新运行）
  - 如果想完全重建，请先删除 chroma_db/ 目录再运行
  - 首次运行会下载嵌入模型（约 90MB），之后从缓存加载

时间参考（本地 CPU）：
  86 个文件 / ~1.7MB 文本 → 约 1~3 分钟
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
# 这样 from rag_system.xxx import yyy 才能正常工作
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.indexer import build_all

if __name__ == "__main__":
    build_all()
