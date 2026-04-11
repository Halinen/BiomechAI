"""
retriever.py — 向量检索器

功能：
  - 接收用户查询文本
  - 在指定的 Chroma collection 中搜索语义最相似的 chunk
  - 返回 top-k 个结果，带原文和来源元数据

核心思路（RAG 的 R 部分）：
  用户问"髋关节活动受限怎么纠正？"
  → 把这句话也变成向量
  → 在 FMS_SFMA collection 里找最接近的向量
  → 把找到的文本片段给 Claude 作为上下文
  → Claude 根据这些片段回答，而不是靠"背诵训练数据"
"""

from dataclasses import dataclass

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from rag_system.config import (
    CHROMA_PATH,
    CLAUDE_MODEL,
    EMBEDDING_MODEL,
    GROQ_BASE_URL,
    SIMILARITY_ACCEPT,
    SIMILARITY_REJECT,
    TOP_K,
)


# ── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    单个检索结果。

    dataclass 是 Python 的一种便捷类，
    自动帮你生成 __init__、__repr__ 等方法，
    相当于一个带类型注解的命名元组。
    """
    text: str        # chunk 的原始文本内容
    source_file: str # 来自哪个文件（如 "Functional_Movement_Screen.txt"）
    folder: str      # 来自哪个知识域（如 "FMS_SFMA"）
    chunk_index: int # 在原文件中是第几个 chunk
    score: float     # 相似度分数（余弦距离，越小越相似；0 = 完全相同）


# ── 检索器类 ────────────────────────────────────────────────────────────────

class Retriever:
    """
    向量检索器。

    设计为"惰性加载"：
      - __init__ 只保存配置，不立即加载模型
      - 第一次调用 query() 时才真正加载
      - 这样如果你只是 import retriever，不会触发模型下载
    """

    def __init__(self):
        self._model: SentenceTransformer | None = None
        self._client: chromadb.PersistentClient | None = None
        self._llm: OpenAI | None = None

    def _ensure_loaded(self):
        """确保模型和数据库已加载（如果还没加载就加载）。"""
        if self._model is None:
            print(f"[Retriever] 加载嵌入模型：{EMBEDDING_MODEL}")
            self._model = SentenceTransformer(EMBEDDING_MODEL)

        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(CHROMA_PATH))

        if self._llm is None:
            self._llm = OpenAI(
                api_key=__import__("os").environ.get("GROQ_API_KEY"),
                base_url=GROQ_BASE_URL,
            )

    def _llm_judge(self, query: str, chunk: RetrievedChunk) -> bool:
        """
        让 LLM 判断边界情况的 chunk 是否与 query 相关。

        只在 SIMILARITY_ACCEPT <= score < SIMILARITY_REJECT 时调用。
        使用最小化 prompt + max_tokens=10，只需要 yes/no，成本极低。

        返回：True = 保留，False = 丢弃
        """
        prompt = (
            f"Query: {query}\n\n"
            f"Passage: {chunk.text[:400]}\n\n"
            "Is this passage relevant to the query for a movement assessment? "
            "Reply with only 'yes' or 'no'."
        )
        response = self._llm.chat.completions.create(
            model=CLAUDE_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = (response.choices[0].message.content or "").strip().lower()
        return answer.startswith("y")

    def query(
        self,
        text: str,
        collection_name: str,
        top_k: int = TOP_K,
    ) -> list[RetrievedChunk]:
        """
        在指定 collection 中检索与 text 最相似的 chunk，并做两阶段过滤：

          score < SIMILARITY_ACCEPT          → 直接保留（高置信度）
          score >= SIMILARITY_REJECT         → 直接丢弃（低置信度）
          SIMILARITY_ACCEPT <= score < REJECT → LLM 判断（边界情况）

        参数：
          text            — 用户的查询文本（自然语言）
          collection_name — 要搜索的 Chroma collection（如 "fms_sfma"）
          top_k           — 返回多少个结果（默认 5）

        返回：
          list[RetrievedChunk] — 过滤后按相似度排序
        """
        self._ensure_loaded()

        # 把查询文本变成向量（和建索引时用同一个模型，才能比较）
        query_vector = self._model.encode(text).tolist()

        try:
            collection = self._client.get_collection(collection_name)
        except Exception:
            # collection 不存在（还没有运行 build_index.py）
            print(f"[Retriever] 警告：collection '{collection_name}' 不存在，请先运行 build_index.py")
            return []

        # 多取一些候选，给过滤留余量
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k * 3, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        kept: list[RetrievedChunk] = []
        for doc, meta, dist in zip(docs, metas, distances):
            chunk = RetrievedChunk(
                text=doc,
                source_file=meta.get("source_file", "unknown"),
                folder=meta.get("folder", "unknown"),
                chunk_index=meta.get("chunk_index", -1),
                score=dist,
            )

            if dist < SIMILARITY_ACCEPT:
                # 高置信度：直接保留
                kept.append(chunk)
            elif dist < SIMILARITY_REJECT:
                # 边界情况：交给 LLM 判断
                print(f"[Retriever] LLM 判断边界 chunk (score={dist:.3f}): {chunk.source_file}")
                if self._llm_judge(text, chunk):
                    kept.append(chunk)
            # else: score >= SIMILARITY_REJECT，直接丢弃

            if len(kept) >= top_k:
                break

        return kept

    def query_multiple(
        self,
        text: str,
        collection_names: list[str],
        top_k_each: int = TOP_K,
    ) -> dict[str, list[RetrievedChunk]]:
        """
        同时在多个 collection 中检索，返回字典 {collection_name: [chunks]}。

        用途：当问题跨多个知识域时（比如既涉及动作筛查又涉及代偿模式），
        可以同时从多个 collection 取结果，交给 Claude 综合分析。
        """
        return {
            name: self.query(text, name, top_k_each)
            for name in collection_names
        }


# ── 工具函数 ────────────────────────────────────────────────────────────────

def format_context(chunks: list[RetrievedChunk], max_chars: int = 3000) -> str:
    """
    把检索到的 chunk 列表格式化成一段文本，供 Claude 的 prompt 使用。

    参数：
      chunks    — 检索结果列表
      max_chars — 总字符数上限（避免 prompt 太长、token 过多）

    返回：
      str — 格式化后的上下文字符串，包含来源标注
    """
    parts = []
    total = 0

    for i, chunk in enumerate(chunks, 1):
        # 每个 chunk 前加来源标注，让 Claude 知道这条知识的出处
        header = f"[{i}] 来源：{chunk.folder} / {chunk.source_file}\n"
        block = header + chunk.text + "\n"

        if total + len(block) > max_chars:
            break  # 超出上限就停止追加

        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts)
