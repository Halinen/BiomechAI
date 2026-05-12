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
from rag_system.trace import log_llm_call


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

        关键设计：
          - few-shot 示例覆盖单层 / 双层 / 红旗各类常见模式，约束 LLM 学到「选最相关的 1-2 层」
          - 显式禁止贪心全选（之前 14/15 case 返回 3-4 层全选）
          - JSON 输出强制结构化解析，避免自由文本歧义
          - 失败时退化为「结构层 + 输出层」（最通用的两个）而不是全查
        """
        layer_names = list(LAYER_DESCRIPTIONS.keys())  # ["控制层", "结构层", "输出层", "神经敏化层"]

        prompt = f"""你是动作评估专家。根据用户描述判断问题主要涉及哪 1-2 层。

【四层定义】
- 控制层：神经肌肉控制（呼吸模式、膈肌、骨盆感知、激活顺序）
- 结构层：关节/组织结构（活动度受限、僵硬、术后粘连、内脏筋膜限制）
- 输出层：肌肉失衡/代偿（上下交叉综合征、肌肉抑制、关节力线异常）
- 神经敏化层：疼痛敏化（按压触发全身反应、影像阴性但疼痛严重、恐惧回避、广泛敏感）

【硬约束】
- 选最相关的 1-2 层。不要贪心全选。
- 描述很简单只涉及一个领域 → 只选 1 层。
- 描述明确跨多个领域才选 2 层。
- 仅当三层都有具体证据时才选 3 层（罕见）。

【示例】
描述："深蹲膝盖内扣，脚后跟离地"
→ {{"layers": ["结构层"], "reason": "都是关节活动度问题"}}

描述："长期伏案，头前伸，圆肩，做引体拉不起"
→ {{"layers": ["输出层"], "reason": "典型上交叉综合征代偿模式"}}

描述："吸气只有胸口起伏，肚子不动"
→ {{"layers": ["控制层"], "reason": "纯呼吸模式问题"}}

描述："腰痛三年影像没问题，稍动就全身紧绷，对疼痛敏感"
→ {{"layers": ["神经敏化层"], "reason": "影像阴性 + 全身敏感 = 中枢敏化"}}

描述："深蹲下背塌、胸椎转不动、平时也驼背、腹肌不会发力"
→ {{"layers": ["结构层", "输出层"], "reason": "结构限制 + 代偿模式叠加"}}

描述："颈椎 MRI 轻微突出，但一抬头就头晕、肩颈摸哪都疼"
→ {{"layers": ["结构层", "神经敏化层"], "reason": "结构小问题 + 广泛敏化"}}

描述："剖腹产后疤痕周围拉一下就疼，但其他都正常"
→ {{"layers": ["结构层"], "reason": "纯瘢痕组织限制，不涉及敏化"}}

【现在判断】
描述："{query}"

只输出 JSON，不要解释："""

        try:
            response = self._llm.chat.completions.create(
                model=CLAUDE_MODEL,
                max_tokens=120,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (response.choices[0].message.content or "").strip()

            import json
            data = json.loads(raw)
            layers = data.get("layers", [])
            # 过滤为合法层名，保持顺序
            detected = [name for name in layers if name in layer_names]
            log_llm_call(
                "router.classify", prompt, raw, CLAUDE_MODEL,
                extra={"detected_layers": detected[:3], "reason": data.get("reason", "")},
            )
            if detected:
                # 上限保护：万一 LLM 还是全选了，截到前 3 个
                return detected[:3]
        except Exception as e:
            print(f"[Router] classify() 调用失败，退化为'结构层+输出层'：{e}")
            log_llm_call(
                "router.classify", prompt, f"ERROR: {e}", CLAUDE_MODEL,
                extra={"fallback": True},
            )

        # 退化：返回最通用的两层（之前是全查，太浪费）
        return ["结构层", "输出层"]

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
