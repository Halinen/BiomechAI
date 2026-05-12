"""
trace.py — 轻量 LLM 调用 trace 工具

记录每次 LLM 调用的完整上下文（component / prompt / response / tokens / latency），
按 trace_id 串成一次用户 query 的完整决策链路。

设计原则：
  - 零依赖（不引入 langsmith / langfuse / opentelemetry）
  - 写 JSONL 文件，简单可查
  - 装饰器 + 上下文管理器两套 API
  - 默认关闭，通过环境变量 RAG_TRACE_DIR 启用

用途：
  - 调试 critic 反思过程：看每轮 LLM 选了什么 action
  - 审查 token 消耗：哪些 component 烧得最多
  - 简历级 observability 标配（生产级 AI 系统都有 trace）
"""

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


# 全局 trace 状态（线程不安全；当前是单进程单用户工具）
_current_trace_id: str | None = None
_trace_dir: Path | None = None
_step_counter: int = 0


def _get_trace_dir() -> Path | None:
    """惰性读环境变量；只在第一次启用时创建目录。"""
    global _trace_dir
    if _trace_dir is not None:
        return _trace_dir
    env_dir = os.environ.get("RAG_TRACE_DIR")
    if not env_dir:
        return None
    _trace_dir = Path(env_dir)
    _trace_dir.mkdir(parents=True, exist_ok=True)
    return _trace_dir


@contextmanager
def start_trace(query: str, metadata: dict | None = None):
    """
    开启一次完整 query 的 trace 上下文。

    用法：
        with start_trace(user_query, metadata={"case_id": "case_01"}):
            answer = agent.ask(user_query)
        # → 写入 RAG_TRACE_DIR/<timestamp>_<trace_id>.jsonl

    若未设置 RAG_TRACE_DIR 环境变量，此上下文为空操作。
    """
    global _current_trace_id, _step_counter

    trace_dir = _get_trace_dir()
    if trace_dir is None:
        yield None
        return

    trace_id = uuid.uuid4().hex[:8]
    _current_trace_id = trace_id
    _step_counter = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = trace_dir / f"{timestamp}_{trace_id}.jsonl"

    header = {
        "type": "trace_start",
        "trace_id": trace_id,
        "query": query,
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
    }
    trace_file.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")

    t0 = time.time()
    try:
        yield trace_id
    finally:
        footer = {
            "type": "trace_end",
            "trace_id": trace_id,
            "total_steps": _step_counter,
            "total_seconds": round(time.time() - t0, 2),
        }
        with trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(footer, ensure_ascii=False) + "\n")
        _current_trace_id = None


def log_llm_call(
    component: str,
    prompt: Any,
    response: str,
    model: str,
    extra: dict | None = None,
):
    """
    记录一次 LLM 调用。

    参数：
      component  逻辑组件名（"router.classify" / "retriever.translate" / "agent.generate" /
                 "critic.review" / "retriever.judge"）
      prompt     发给 LLM 的内容（字符串或 messages list）
      response   LLM 返回的文本
      model      模型名
      extra      额外字段（如 iteration, action, score）
    """
    global _step_counter

    trace_dir = _get_trace_dir()
    if trace_dir is None or _current_trace_id is None:
        return

    _step_counter += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = None
    # 找到当前 trace 文件（按 trace_id 后缀匹配）
    for f in trace_dir.glob(f"*_{_current_trace_id}.jsonl"):
        trace_file = f
        break
    if trace_file is None:
        return

    # prompt 可能是 messages list 或字符串，统一序列化
    if isinstance(prompt, str):
        prompt_text = prompt
    else:
        prompt_text = json.dumps(prompt, ensure_ascii=False)

    entry = {
        "type": "llm_call",
        "step": _step_counter,
        "trace_id": _current_trace_id,
        "component": component,
        "model": model,
        "prompt_len": len(prompt_text),
        "response_len": len(response),
        "prompt_preview": prompt_text[:300],
        "response_preview": response[:300],
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        entry["extra"] = extra

    with trace_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def is_enabled() -> bool:
    """便于调用方判断是否需要构造 trace 数据（避免无用 string 操作）。"""
    return _get_trace_dir() is not None


# ── Trace 分析工具 ───────────────────────────────────────────────────────────

def summarize_trace(trace_file: Path) -> dict:
    """读一个 trace 文件，返回汇总信息。给 dashboard / 报告用。"""
    steps = []
    header = None
    footer = None
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["type"] == "trace_start":
            header = entry
        elif entry["type"] == "trace_end":
            footer = entry
        elif entry["type"] == "llm_call":
            steps.append(entry)

    by_component: dict[str, int] = {}
    for s in steps:
        by_component[s["component"]] = by_component.get(s["component"], 0) + 1

    return {
        "trace_id": header["trace_id"] if header else "?",
        "query": header["query"] if header else "",
        "total_llm_calls": len(steps),
        "total_seconds": footer.get("total_seconds") if footer else None,
        "calls_by_component": by_component,
        "steps": steps,
    }
