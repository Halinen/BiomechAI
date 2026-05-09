"""
critic.py — Critic Agent（自我反思环）

整个 agentic 升级的核心：把单轮 RAG 变成「生成 → 质检 → 不合格则修正再生成」的 loop。

LLM 在这一步获得了真正的决策权：
  - 判断当前回答是否合格
  - 决定下一步采取哪种修正动作（retrieve_more / remove_hallucination / add_red_flag / ok）
  - 若 retrieve_more，自主改写 query 让 retriever 重检索

设计参考：Reflexion、Self-Refine、Constitutional AI 的 self-critique 模式。
"""

import json
from dataclasses import dataclass

from openai import OpenAI

from rag_system.config import CLAUDE_MODEL
from rag_system.retriever import RetrievedChunk


# ── 数据结构 ────────────────────────────────────────────────────────────────

VALID_ACTIONS = {"ok", "retrieve_more", "remove_hallucination", "add_red_flag"}


@dataclass
class CritiqueResult:
    """Critic 对一条 draft 回答的审查结果。"""
    pass_check: bool                  # 是否通过
    issues: list[str]                 # 具体问题描述列表
    action: str                       # ok / retrieve_more / remove_hallucination / add_red_flag
    suggested_query: str | None       # 若 retrieve_more，给出的改写 query（英文）
    raw_response: str = ""            # 原始 LLM 输出（debug 用）


# ── Critic 主类 ─────────────────────────────────────────────────────────────

class Critic:
    """
    LLM-as-critic：读 chunks + draft，判断质量，给出修正建议。

    注意 critic 模型当前与 generator 同源（都是 Llama-3.3-70b），
    存在「自家不查自家幻觉」的偏差。预算允许时建议换更强的模型。
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def review(
        self,
        user_query: str,
        retrieved_chunks: list[RetrievedChunk],
        draft_answer: str,
    ) -> CritiqueResult:
        """对一份 draft 回答做质检。"""
        # 拼参考片段（限 2500 字符避免 prompt 太长）
        chunk_parts = []
        for i, c in enumerate(retrieved_chunks[:10], 1):
            chunk_parts.append(f"[{i}] ({c.folder}) {c.text[:400]}")
        chunks_text = "\n\n".join(chunk_parts)[:2500]

        prompt = f"""你是临床动作评估的回答质检员。审查下面的 AI 草稿是否合格。

用户问题：
{user_query}

检索到的参考片段：
{chunks_text}

AI 回答草稿：
{draft_answer}

审查标准（任何一条不合格就 fail）：
1. concept 引用：草稿是否实质性引用了参考片段里的核心概念？
   只用通用建议（"保持坐姿"、"定期休息"）而不引用专业术语 = fail
2. 幻觉：草稿是否有"参考片段没说、常识也支持不了"的具体断言？
3. 红旗症状：用户描述里有没有需要立即就医的信号？
   信号包括：夜间痛醒、马尾综合征（大小便失禁/会阴麻木）、心源性（胸痛放射/喘不上气）、
   全身症状（不明发热/体重下降）、急性神经压迫。
   如果有，草稿必须明确建议立即就医；没有则不应误报红旗。

返回严格 JSON（不要任何 JSON 以外内容）：
{{
  "pass": true 或 false,
  "issues": ["<具体问题1>", "<具体问题2>"],
  "action": "ok" | "retrieve_more" | "remove_hallucination" | "add_red_flag",
  "suggested_query": "<若 action=retrieve_more，给出英文改写后的检索 query；否则空字符串>"
}}

action 选择规则：
- 草稿合格 → "ok"
- 检索片段里没有讨论核心 concept，但 query 改写后可能找到 → "retrieve_more"
- 草稿有幻觉断言需要剔除 → "remove_hallucination"
- 该触发红旗但草稿没提 → "add_red_flag"
"""
        try:
            resp = self.llm.chat.completions.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
        except Exception as e:
            print(f"[Critic] 调用失败，默认通过：{e}")
            return CritiqueResult(
                pass_check=True,
                issues=[],
                action="ok",
                suggested_query=None,
                raw_response=str(e),
            )

        action = str(data.get("action", "ok"))
        if action not in VALID_ACTIONS:
            action = "ok"

        suggested = data.get("suggested_query") or None
        if isinstance(suggested, str) and not suggested.strip():
            suggested = None

        return CritiqueResult(
            pass_check=bool(data.get("pass", True)),
            issues=[str(i) for i in data.get("issues", [])],
            action=action,
            suggested_query=suggested,
            raw_response=raw,
        )
