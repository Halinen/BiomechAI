"""
assessment.py — 评估路由器

功能：
  - 接收用户描述的问题，判断属于哪个"层"
  - 根据分层结果决定去哪些 collection 检索
  - 把检索结果组装成结构化的上下文，交给 agent.py 生成回答

三层框架回顾：
  控制层 → 神经肌肉控制问题 → 检索 PRI
  结构层 → 关节/组织结构问题 → 检索 FMS_SFMA
  输出层 → 肌肉失衡/代偿模式 → 检索 NASM CES

红旗症状独立检索：
  无论哪层，都会额外检索 Red_Flags，
  如果发现需要转介，会在回答中明确提示。
"""

from rag_system.config import COLLECTIONS, LAYER_DESCRIPTIONS, TOP_K
from rag_system.retriever import Retriever, RetrievedChunk, format_context


# ── 层 → Collection 映射 ───────────────────────────────────────────────────

# 每个"层"对应要检索的 Chroma collection
# 如果某层对应多个 collection，列表里写多个
LAYER_TO_COLLECTIONS: dict[str, list[str]] = {
    "控制层": [COLLECTIONS["PRI"]],
    "结构层": [COLLECTIONS["FMS_SFMA"]],
    "输出层": [COLLECTIONS["NASM_CES"]],
}

# 红旗症状 collection 始终额外检索
RED_FLAGS_COLLECTION = COLLECTIONS["Red_Flags"]


# ── 结果数据结构 ────────────────────────────────────────────────────────────

class AssessmentContext:
    """
    评估路由后的上下文容器。

    属性：
      query         — 用户原始查询
      primary_layer — 判断出的主要层（控制层/结构层/输出层/未分类）
      layer_context — 对应层的检索结果文本（已格式化）
      red_flag_chunks — 红旗症状检索结果
      all_chunks    — 所有检索到的原始 chunk（调试用）
    """
    def __init__(
        self,
        query: str,
        primary_layer: str,
        layer_context: str,
        red_flag_chunks: list[RetrievedChunk],
        all_chunks: dict[str, list[RetrievedChunk]],
    ):
        self.query = query
        self.primary_layer = primary_layer
        self.layer_context = layer_context
        self.red_flag_chunks = red_flag_chunks
        self.all_chunks = all_chunks

    def has_red_flags(self, score_threshold: float = 0.5) -> bool:
        """
        判断是否有高度相关的红旗症状内容被检索到。

        score_threshold：距离阈值，低于此值认为"高度相关"
        （余弦距离 0 = 完全相同，1 = 完全不相关）
        """
        return any(c.score < score_threshold for c in self.red_flag_chunks)


# ── 路由器 ──────────────────────────────────────────────────────────────────

class AssessmentRouter:
    """
    评估路由器：给定用户描述，检索相关知识，返回结构化上下文。

    关于"分层"的说明：
      当前版本由 Claude 在生成回答时进行分层判断。
      路由器的职责是"全查"——把三层的内容都检索出来，
      让 Claude 自己决定哪层最相关，并在回答中说明。

      更进一步的版本可以先做一次分类调用（让 Claude 只输出层名），
      再针对性检索，节省 token。这里保持简单，先全查。
    """

    def __init__(self):
        self._retriever = Retriever()

    def route(self, query: str, top_k: int = TOP_K) -> AssessmentContext:
        """
        对用户查询进行多层检索，返回结构化上下文。

        流程：
          1. 并行查询三层对应的 collection
          2. 查询红旗症状 collection
          3. 合并结果，格式化成可读文本
        """
        # 收集所有层的检索结果
        all_chunks: dict[str, list[RetrievedChunk]] = {}

        layer_contexts = []
        for layer_name, collection_names in LAYER_TO_COLLECTIONS.items():
            layer_chunks = []
            for cname in collection_names:
                chunks = self._retriever.query(query, cname, top_k)
                layer_chunks.extend(chunks)
                all_chunks[cname] = chunks

            if layer_chunks:
                # 用层名作为标题，方便 Claude 识别来源
                section = f"## {layer_name}相关知识\n\n{format_context(layer_chunks)}"
                layer_contexts.append(section)

        # 检索红旗症状
        red_flag_chunks = self._retriever.query(query, RED_FLAGS_COLLECTION, top_k=3)
        all_chunks[RED_FLAGS_COLLECTION] = red_flag_chunks

        # 组合所有上下文
        combined_context = "\n\n".join(layer_contexts)

        # primary_layer 暂时留空，由 Claude 在生成时判断
        return AssessmentContext(
            query=query,
            primary_layer="待 Claude 判断",
            layer_context=combined_context,
            red_flag_chunks=red_flag_chunks,
            all_chunks=all_chunks,
        )

    def route_targeted(self, query: str, layer: str, top_k: int = TOP_K) -> AssessmentContext:
        """
        指定层进行检索（用户或 Claude 已明确知道属于哪层时使用）。

        参数：
          layer — "控制层" / "结构层" / "输出层"
        """
        collection_names = LAYER_TO_COLLECTIONS.get(layer, [])
        all_chunks: dict[str, list[RetrievedChunk]] = {}

        if not collection_names:
            layer_context = f"（未找到 '{layer}' 对应的 collection）"
        else:
            layer_chunks = []
            for cname in collection_names:
                chunks = self._retriever.query(query, cname, top_k)
                layer_chunks.extend(chunks)
                all_chunks[cname] = chunks
            layer_context = format_context(layer_chunks)

        red_flag_chunks = self._retriever.query(query, RED_FLAGS_COLLECTION, top_k=3)
        all_chunks[RED_FLAGS_COLLECTION] = red_flag_chunks

        return AssessmentContext(
            query=query,
            primary_layer=layer,
            layer_context=layer_context,
            red_flag_chunks=red_flag_chunks,
            all_chunks=all_chunks,
        )


# ── 供 agent.py 使用的工具函数 ─────────────────────────────────────────────

def build_rag_prompt_section(ctx: AssessmentContext) -> str:
    """
    把 AssessmentContext 转换成可以直接插入 Claude prompt 的文本块。

    格式：
      [检索到的参考知识]
      ...layer_context...

      [红旗症状参考]（如有相关内容）
      ...red_flag_context...
    """
    sections = []

    if ctx.layer_context.strip():
        sections.append("【检索到的参考知识】\n\n" + ctx.layer_context)

    if ctx.red_flag_chunks:
        rf_text = format_context(ctx.red_flag_chunks, max_chars=1500)
        sections.append("【红旗症状 / 转介参考】\n\n" + rf_text)

    return "\n\n" + "=" * 50 + "\n\n".join(sections) + "\n\n" + "=" * 50
