"""
indexer.py — 知识库索引构建器

功能：
  1. 读取 knowledge_base/ 下各子文件夹的 .txt 文件
  2. 把每篇文档切成小块（chunk），让向量模型能处理
  3. 用 sentence-transformers 把每个 chunk 变成向量
  4. 把向量 + 原文 + 元数据存入对应的 Chroma collection

为什么要分 collection？
  - 每个知识域（FMS/SFMA、NASM CES、PRI、Red Flags）独立存储
  - 检索时只查相关领域，避免不同领域的内容互相干扰
  - 后期迁移到 pgvector 也是同样的分表逻辑

运行方式：只需在项目根目录执行 build_index.py，它会调用这里的 build_all()
"""

import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag_system.config import (
    KB_PATH,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    COLLECTIONS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ── 1. 文本分块 ────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    把一篇文档切成若干个有重叠的小块。

    策略：
      - 先按换行符分段（段落天然的边界）
      - 把段落拼接，当累积字符数超过 chunk_size 时切断
      - 下一个 chunk 回退 overlap 个字符，保留上下文衔接

    参数：
      text       — 完整文档文本
      chunk_size — 每块目标字符数（默认 450）
      overlap    — 相邻块重叠字符数（默认 80）

    返回：
      list[str]  — 切好的 chunk 列表，每个元素是一段文本
    """
    # 按连续空行（段落分隔符）拆分
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # 如果加上这段还没超过 chunk_size，直接拼接
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n" + para).strip()
        else:
            # 当前 chunk 已满，先保存
            if current:
                chunks.append(current)
            # 从上一个 chunk 的末尾取 overlap 个字符作为新 chunk 的开头
            # 这样即使切断处有重要信息，下一个 chunk 也能覆盖到
            tail = current[-overlap:] if len(current) > overlap else current
            current = (tail + "\n" + para).strip()

    # 别忘了最后剩余的部分
    if current:
        chunks.append(current)

    return chunks


# ── 2. 加载向量模型 ─────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    """
    加载 sentence-transformers 模型。

    all-MiniLM-L6-v2：
      - 首次运行会从 HuggingFace 自动下载（约 90MB）
      - 之后从本地缓存加载，很快
      - 输出 384 维向量，适合语义相似度检索
    """
    print(f"[Indexer] 加载嵌入模型：{EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


# ── 3. 构建单个 Collection ──────────────────────────────────────────────────

def build_collection(
    folder_name: str,
    collection_name: str,
    model: SentenceTransformer,
    chroma_client: chromadb.PersistentClient,
) -> int:
    """
    处理一个知识域文件夹，把所有文档切块、向量化、存入 Chroma。

    参数：
      folder_name     — knowledge_base/ 下的子文件夹名（如 "FMS_SFMA"）
      collection_name — 对应的 Chroma collection 名（如 "fms_sfma"）
      model           — 已加载的 SentenceTransformer 模型
      chroma_client   — 已连接的 Chroma 客户端

    返回：
      int — 本次索引的 chunk 数量
    """
    kb_dir = KB_PATH / folder_name
    if not kb_dir.exists():
        print(f"  [跳过] 目录不存在：{kb_dir}")
        return 0

    txt_files = list(kb_dir.glob("*.txt"))
    if not txt_files:
        print(f"  [跳过] 目录为空：{kb_dir}")
        return 0

    # get_or_create_collection：已存在就用旧的，不存在就新建
    # 注意：重复运行 build_index.py 时，旧数据会保留并追加
    # 如果想重建，需要先删除 chroma_db/ 目录
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        # 这里不需要传 embedding_function，因为我们自己手动传入向量
        metadata={"hnsw:space": "cosine"},  # 使用余弦相似度，适合文本语义匹配
    )

    total_chunks = 0
    existing_ids = set(collection.get()["ids"])  # 已存在的 chunk ID，用于去重

    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        # 为每个 chunk 生成唯一 ID：文件名 + chunk 序号
        # 格式：fms_sfma__Functional_Movement_Screen__0, __1, __2 ...
        stem = txt_path.stem[:50]  # 文件名（去掉 .txt 后缀），最多 50 字符
        ids = [f"{collection_name}__{stem}__{i}" for i in range(len(chunks))]

        # 过滤掉已经存在的 chunk（避免重复索引）
        new_ids, new_chunks = [], []
        for cid, chunk in zip(ids, chunks):
            if cid not in existing_ids:
                new_ids.append(cid)
                new_chunks.append(chunk)

        if not new_chunks:
            continue  # 这个文件已经全部索引过了

        # 向量化：把文本列表变成 numpy 矩阵
        # show_progress_bar=False 避免控制台输出太乱
        embeddings = model.encode(new_chunks, show_progress_bar=False).tolist()

        # 元数据：每个 chunk 记录来源文件和所属知识域
        # 检索到结果后可以告诉用户"这条来自哪篇文章"
        metadatas = [
            {
                "source_file": txt_path.name,
                "folder": folder_name,
                "chunk_index": i,
            }
            for i in range(len(new_chunks))
        ]

        # 批量写入 Chroma
        # documents — 原始文本（检索到后直接送给 Claude）
        # embeddings — 对应的向量（用于相似度搜索）
        # ids        — 唯一标识符（Chroma 要求必须有）
        # metadatas  — 额外字段（过滤、溯源用）
        collection.add(
            documents=new_chunks,
            embeddings=embeddings,
            ids=new_ids,
            metadatas=metadatas,
        )

        total_chunks += len(new_chunks)
        print(f"    {txt_path.name}: {len(chunks)} chunks → 新增 {len(new_chunks)}")

    return total_chunks


# ── 4. 构建所有 Collection ──────────────────────────────────────────────────

def build_all() -> None:
    """
    遍历 config.py 里定义的所有知识域，依次构建索引。

    调用顺序：
      1. 加载嵌入模型（只加载一次，供所有 collection 共用）
      2. 连接 Chroma（PersistentClient 会把数据持久化到 chroma_db/ 目录）
      3. 逐个处理文件夹
    """
    print("=" * 55)
    print("开始构建知识库索引")
    print(f"知识库目录：{KB_PATH}")
    print(f"Chroma 存储：{CHROMA_PATH}")
    print("=" * 55)

    # 加载向量模型（只做一次）
    model = load_embedding_model()

    # 连接 Chroma 持久化客户端
    # PersistentClient 会把所有数据写入磁盘，程序退出后仍然保留
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # 逐个处理知识域
    grand_total = 0
    for folder_name, collection_name in COLLECTIONS.items():
        print(f"\n[{folder_name}] → collection: {collection_name}")
        count = build_collection(folder_name, collection_name, model, client)
        grand_total += count
        print(f"  小计：{count} 个新 chunk")

    print("\n" + "=" * 55)
    print(f"索引构建完成！共写入 {grand_total} 个 chunk")
    print("=" * 55)
