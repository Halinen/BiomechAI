"""
assessment.py — 评估路由器

功能：
  - 接收用户描述的问题，判断属于哪个"层"
  - 根据分层结果决定去哪些 collection 检索
  - 把检索结果组装成结构化的上下文，交给 agent.py 生成回答

四层框架：
  控制层   → 神经肌肉控制问题       → 检索 PRI + Breathing_Retraining
  结构层   → 关节/组织/内脏筋膜问题 → 检索 FMS_SFMA + Visceral_Fascia
  输出层   → 肌肉失衡/代偿模式      → 检索 NASM CES
  神经敏化层 → 中枢敏化/疼痛神经科学 → 检索 Pain_Neuroscience

红旗症状独立检索：
  无论哪层，都会额外检索 Red_Flags，
  如果发现需要转介，会在回答中明确提示。
"""

import os

from openai import OpenAI

from rag_system.config import COLLECTIONS, GROQ_BASE_URL, CLAUDE_MODEL, LAYER_DESCRIPTIONS, TOP_K
from rag_system.retriever import Retriever, RetrievedChunk, format_context


# ── 层 → Collection 映射 ───────────────────────────────────────────────────

# 每个"层"对应要检索的 Chroma collection
# 如果某层对应多个 collection，列表里写多个
LAYER_TO_COLLECTIONS: dict[str, list[str]] = {
    "控制层":    [COLLECTIONS["PRI"], COLLECTIONS["Breathing_Retraining"]],
    "结构层":    [COLLECTIONS["FMS_SFMA"], COLLECTIONS["Visceral_Fascia"]],
    "输出层":    [COLLECTIONS["NASM_CES"]],
    "神经敏化层": [COLLECTIONS["Pain_Neuroscience"]],
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

    分两步走：
      1. classify()：让 LLM 判断问题属于哪些层（只输出层名，token 极少）
      2. route() / route_targeted()：按分类结果定向检索，避免全查噪音
    """

    def __init__(self):
        self._retriever = Retriever()
        self._llm = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url=GROQ_BASE_URL,
        )

    def classify(self, query: str) -> list[str]:
        """
        第一步：让 LLM 判断问题属于哪些层，只返回层名列表。

        使用极短 prompt + max_tokens=30，成本极低（比全查三层省 ~60% token）。
        返回示例：["控制层"]、["结构层", "输出层"]、["控制层", "结构层", "输出层"]

        如果 LLM 返回无法解析的内容，退化为全查（返回全部三层）。
        """
        layer_names = list(LAYER_DESCRIPTIONS.keys())  # ["控制层", "结构层", "输出层", "神经敏化层"]

        prompt = f"""你是动作评估专家。根据用户描述，判断问题主要涉及哪些层。

四层定义：
- 控制层：神经肌肉控制问题，如呼吸模式异常、骨盆感知缺失、肌肉激活顺序错误、膈肌功能异常
- 结构层：关节/组织结构问题，如髋关节活动受限、胸椎僵硬、足弓塌陷、术后腹腔粘连、内脏筋膜限制
- 输出层：肌肉失衡/代偿模式，如上交叉综合征、臀肌抑制、髋屈肌过激活
- 神经敏化层：疼痛敏化/中枢敏化问题，如按压触发全身反应、疼痛与组织损伤不成比例、恐惧回避、神经张力异常

用户描述：{query}

只输出涉及的层名，用逗号分隔，例如：控制层,输出层
不要输出任何其他内容。"""

        try:
            response = self._llm.chat.completions.create(
                model=CLAUDE_MODEL,
                max_tokens=30,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (response.choices[0].message.content or "").strip()

            # 解析：从返回文本中提取合法的层名
            detected = [name for name in layer_names if name in raw]
            if detected:
                return detected
        except Exception as e:
            print(f"[Router] classify() 调用失败，退化为全查：{e}")

        # 退化：全查
        return layer_names

    def route(self, query: str, top_k: int = TOP_K) -> AssessmentContext:
        """
        对用户查询进行两步检索，返回结构化上下文。

        流程：
          1. classify()：让 LLM 判断属于哪些层（一次轻量调用）
          2. 只检索命中层对应的 collection，避免全查噪音
          3. 始终额外检索红旗症状 collection
        """
        # 第一步：分类
        detected_layers = self.classify(query)
        print(f"[Router] 分类结果：{detected_layers}")

        # 第二步：只检索命中的层
        all_chunks: dict[str, list[RetrievedChunk]] = {}
        layer_contexts = []

        for layer_name in detected_layers:
            collection_names = LAYER_TO_COLLECTIONS.get(layer_name, [])
            layer_chunks = []
            for cname in collection_names:
                chunks = self._retriever.query(query, cname, top_k)
                layer_chunks.extend(chunks)
                all_chunks[cname] = chunks

            if layer_chunks:
                section = f"## {layer_name}相关知识\n\n{format_context(layer_chunks)}"
                layer_contexts.append(section)

        # 始终检索红旗症状
        red_flag_chunks = self._retriever.query(query, RED_FLAGS_COLLECTION, top_k=3)
        all_chunks[RED_FLAGS_COLLECTION] = red_flag_chunks

        combined_context = "\n\n".join(layer_contexts)

        return AssessmentContext(
            query=query,
            primary_layer="、".join(detected_layers),
            layer_context=combined_context,
            red_flag_chunks=red_flag_chunks,
            all_chunks=all_chunks,
        )

    def route_targeted(self, query: str, layer: str, top_k: int = TOP_K) -> AssessmentContext:
        """
        指定层进行检索（用户或 Claude 已明确知道属于哪层时使用）。

        参数：
          layer — "控制层" / "结构层" / "输出层" / "神经敏化层"
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
