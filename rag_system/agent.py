"""
agent.py — Claude API 集成层

功能：
  - 把 RAG 检索结果 + 用户问题 + 可选图片 打包成 prompt
  - 调用 Claude API 生成回答
  - 支持流式输出（打字机效果）
  - 支持多轮对话（对话历史管理）

本文件是整个系统的"大脑"：
  用户问题 → assessment.py（检索） → agent.py（生成回答）
"""

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from rag_system.config import CLAUDE_MODEL, GROQ_BASE_URL, LAYER_DESCRIPTIONS
from rag_system.assessment import AssessmentRouter, AssessmentContext, build_rag_prompt_section


# ── 系统提示词 ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一名专业的动作评估顾问，精通以下框架：

1. FMS/SFMA（功能性动作筛查）— 用于识别动作受限模式
2. NASM CES（纠正性训练）— 用于制定抑制→延展→激活→整合的纠正流程
3. PRI（姿势恢复）— 用于处理呼吸模式异常、肋廓不对称、骨盆旋转等问题
4. 内脏筋膜手法（Visceral Manipulation）— 用于处理术后器官粘连对体壁和呼吸的牵涉影响
5. 疼痛神经科学（Pain Neuroscience Education）— 用于解释中枢敏化、恐惧回避和疼痛放大机制
6. 呼吸重训（DNS / Breathing Retraining）— 用于处理膈肌功能异常和代偿性呼吸模式

你的评估框架分为四层：
{layer_descriptions}

工作方式：
- 收到用户描述后，先判断问题主要属于哪一层（可以多层并存）
- 说明判断依据（为什么认为是这一层而不是另一层）
- 给出对应层的纠正思路，引用知识库内容时请自然融入，不要机械罗列
- 如果发现潜在红旗症状（需要医疗转介的情况），必须明确提示

语言规范：
- 回答用中文
- 专业术语第一次出现时附上英文原文，例如"深蹲 (Squat)"
- 回答要有结构，但不要过于机械，像一位有经验的教练在解释

重要限制：
- 你只能提供动作评估和纠正训练建议
- 如果发现红旗症状（急性疼痛、神经症状、关节不稳定等），必须建议寻求医疗专业人士
- 不要替代医疗诊断
""".format(
    layer_descriptions="\n".join(
        f"  - {name}：{desc}"
        for name, desc in LAYER_DESCRIPTIONS.items()
    )
)


# ── 结构化输出 ───────────────────────────────────────────────────────────────

@dataclass
class AssessmentResult:
    """
    LLM 返回的结构化评估结果。

    比自由文本的优势：
      - red_flag_detected 可直接驱动 UI 显示警告
      - follow_up_question 可直接显示为追问提示
      - primary_layer 可用于调试和日志
      - answer 是给用户看的正文
    """
    primary_layer: str        # 判断属于哪层（可多层，如"控制层、输出层"）
    confidence: float         # 判断把握程度 0.0~1.0
    red_flag_detected: bool   # 是否发现需要转介的红旗症状
    answer: str               # 给用户的正式回答
    follow_up_question: str   # 信息不足时的追问；无需追问则为空字符串


# 告诉 LLM 必须返回的 JSON 格式（用于 response_format 或 prompt 约束）
RESPONSE_JSON_SCHEMA = """{
  "primary_layer": "控制层 | 结构层 | 输出层 | 神经敏化层（可多个，逗号分隔）",
  "confidence": 0.0到1.0的小数,
  "red_flag_detected": true或false,
  "answer": "给用户的完整评估回答",
  "follow_up_question": "需要追问用户的问题，不需要则留空字符串"
}"""


def parse_structured_response(raw: str) -> AssessmentResult:
    """
    把 LLM 返回的 JSON 字符串解析为 AssessmentResult。

    容错处理：如果解析失败，把整个原始文本作为 answer 返回，
    保证系统不崩溃（降级为旧版自由文本行为）。
    """
    try:
        # 提取 JSON 块（LLM 有时会在 JSON 前后加说明文字）
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("未找到 JSON 块")

        data = json.loads(raw[start:end])
        return AssessmentResult(
            primary_layer=str(data.get("primary_layer", "未知")),
            confidence=float(data.get("confidence", 0.5)),
            red_flag_detected=bool(data.get("red_flag_detected", False)),
            answer=str(data.get("answer", raw)),
            follow_up_question=str(data.get("follow_up_question", "")),
        )
    except Exception as e:
        print(f"[Agent] 结构化输出解析失败，降级为自由文本：{e}")
        return AssessmentResult(
            primary_layer="未知",
            confidence=0.5,
            red_flag_detected=False,
            answer=raw,
            follow_up_question="",
        )


# ── 图片处理 ────────────────────────────────────────────────────────────────

def encode_image(image_path: str | Path) -> tuple[str, str]:
    """
    把本地图片文件编码为 base64，供 Claude API 使用。

    返回：
      (base64字符串, media_type)
      例如：("iVBORw0KGgo...", "image/png")
    """
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


# ── 对话消息构建 ─────────────────────────────────────────────────────────────

def build_user_message(
    query: str,
    ctx: AssessmentContext,
    image_path: str | Path | None = None,
) -> dict:
    """
    构建一条完整的用户消息，包含：
      - 检索到的参考知识（RAG 上下文）
      - 用户的原始问题
      - 可选图片（用于姿势评估）

    Claude API 的消息格式：
      content 字段可以是字符串（只有文字）
      也可以是列表（文字 + 图片混合）
    """
    rag_section = build_rag_prompt_section(ctx)

    # 组装完整的文字部分
    text_content = f"""以下是从专业知识库中检索到的相关内容，请结合这些资料回答问题：

{rag_section}

---

用户问题：{query}

---

请直接用中文给出评估和建议，像一位有经验的教练在解释。回答完成后，在最后另起一行输出一行 JSON 元数据（不要放在代码块里）：
{{"primary_layer":"层名","confidence":0.0到1.0,"red_flag_detected":true或false,"follow_up_question":"追问内容或空字符串"}}"""

    # 如果有图片，构建多模态消息
    if image_path:
        image_data, media_type = encode_image(image_path)
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                }
            },
            {
                "type": "text",
                "text": text_content + "\n\n（请结合上方图片中可见的姿势/动作信息进行分析）"
            }
        ]
    else:
        content = text_content

    return {"role": "user", "content": content}


def _strip_metadata_line(text: str) -> str:
    """剥离 LLM 在回答末尾追加的 JSON 元数据行，返回纯正文。"""
    lines = text.rstrip().splitlines()
    # 从末尾往前找第一行以 { 开头、以 } 结尾的行（即元数据行）
    for i in range(len(lines) - 1, max(len(lines) - 4, -1), -1):
        line = lines[i].strip()
        if line.startswith("{") and line.endswith("}"):
            return "\n".join(lines[:i]).rstrip()
    return text.rstrip()


# ── Agent 主类 ───────────────────────────────────────────────────────────────

class AssessmentAgent:
    """
    AI 评估顾问。

    整合 RAG 检索 + Claude API 调用，提供：
      - 单次回答（ask）
      - 流式回答（ask_stream）—— 打字机效果，适合 Gradio UI

    对话历史管理：
      Claude API 是无状态的，每次调用都需要传入完整历史。
      这里用 self.history 列表维护，每轮对话追加一条 user + 一条 assistant。
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=__import__("os").environ.get("GROQ_API_KEY"),
            base_url=GROQ_BASE_URL,
        )
        self.router = AssessmentRouter()
        self.history: list[dict] = []  # 对话历史

    def reset(self):
        """清空对话历史（开始新的评估会话）。"""
        self.history = []

    def ask(
        self,
        query: str,
        image_path: str | Path | None = None,
        layer: str | None = None,
    ) -> str:
        """
        发送一条消息，返回完整回答（非流式）。

        参数：
          query      — 用户问题
          image_path — 可选图片路径（姿势照片、动作截图等）
          layer      — 可选指定层（"控制层"/"结构层"/"输出层"）；
                       不传则三层都查

        返回：
          str — Claude 的回答文本
        """
        # 1. RAG 检索
        if layer:
            ctx = self.router.route_targeted(query, layer)
        else:
            ctx = self.router.route(query)

        # 2. 构建当前轮消息
        user_msg = build_user_message(query, ctx, image_path)

        # 3. 把历史 + 当前消息发给 Claude
        messages = self.history + [user_msg]

        response = self.client.chat.completions.create(
            model=CLAUDE_MODEL,
            max_tokens=8000,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )

        # 4. 提取文字回答
        answer = response.choices[0].message.content or "（未能生成回答）"

        # 5. 更新对话历史（保存用户问题和 Claude 回答）
        # 注意：history 里的 user 消息去掉 RAG 上下文，只保留用户原话，节省 token
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def ask_stream(
        self,
        query: str,
        image_path: str | Path | None = None,
        layer: str | None = None,
    ):
        """
        发送一条消息，以生成器方式流式返回回答。

        用法（在 app.py 里）：
          for chunk in agent.ask_stream(query, image_path):
              accumulated += chunk
              yield accumulated  # Gradio 的 gr.update 方式

        参数/返回：
          同 ask()，但改为 yield 逐段文字
        """
        # RAG 检索
        if layer:
            ctx = self.router.route_targeted(query, layer)
        else:
            ctx = self.router.route(query)

        user_msg = build_user_message(query, ctx, image_path)
        messages = self.history + [user_msg]

        # 流式调用——先收完全文再处理，避免 JSON 元数据暴露给用户
        full_answer = ""
        stream = self.client.chat.completions.create(
            model=CLAUDE_MODEL,
            max_tokens=8000,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            stream=True,
        )
        for chunk in stream:
            text_chunk = chunk.choices[0].delta.content or ""
            full_answer += text_chunk

        # 把末尾的 JSON 元数据行剥离，只向用户展示正文
        answer_text = _strip_metadata_line(full_answer)

        yield answer_text

        # 更新对话历史（存正文，不存元数据）
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer_text})
