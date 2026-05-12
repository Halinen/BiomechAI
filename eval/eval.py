"""
eval.py — RAG 三段式评估脚本

跑法：
  py -m eval.eval                       # 跑全部用例
  py -m eval.eval --sample 3            # 只跑前 3 条（调试用）
  py -m eval.eval --out custom.md       # 指定输出路径

输出：
  控制台 — rich 表格汇总
  markdown — 逐条详细日志（默认 eval/eval_results.md）

评估三段独立：
  1. 路由：classify() 命中 gold_layers（程序计算 IoU + strict）
  2. 检索：每个 gold_concept 是否被 top-k chunk 覆盖（LLM judge）
  3. 生成：回答的 concept 覆盖、幻觉、红旗处置（LLM judge 一次）
"""

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Windows GBK 控制台不支持 ▶ ✓ 等符号，强制 stdout 用 UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 加载 .env（GROQ_API_KEY）
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

import yaml
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from rag_system.assessment import LAYER_TO_COLLECTIONS
from rag_system.config import CLAUDE_MODEL, GROQ_BASE_URL, COLLECTIONS, TOP_K
from rag_system.retriever import RetrievedChunk
from rag_system.trace import start_trace


# ── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class Case:
    id: str
    query: str
    gold_layers: list[str]
    gold_concepts: list[str]
    gold_red_flag: bool
    notes: str = ""


@dataclass
class Trace:
    """一次完整流水线的产物。"""
    case: Case
    predicted_layers: list[str]
    chunks: list[RetrievedChunk]                 # 所有命中层的 chunk 合并
    chunks_by_collection: dict[str, list[RetrievedChunk]]
    answer: str
    # Critic agent trace（从 AssessmentAgent 拿）
    critic_iterations: int = 1
    critic_actions: list[str] = None              # type: ignore
    critic_issues: list[list[str]] = None         # type: ignore


@dataclass
class CaseScore:
    case_id: str
    # 路由
    layer_strict: bool                           # 集合严格相等
    layer_jaccard: float                         # IoU
    # 检索
    concept_recall: float                        # gold_concepts 中被覆盖的比例
    concept_hits: dict[str, bool]                # 每个 concept 是否命中
    # 生成
    gen_concept_coverage: float                  # 0/0.5/1 加权平均
    gen_concept_breakdown: dict[str, float]      # 每个 concept 的得分
    gen_hallucination_count: int
    gen_hallucination_spans: list[str]
    gen_red_flag_handling: str                   # correct / missed / false_alarm
    gen_overall: float                           # 0-5
    gen_reason: str


# ── 加载测试集 ──────────────────────────────────────────────────────────────

def load_testset(path: Path) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Case(**item) for item in raw]


# ── 流水线执行 ──────────────────────────────────────────────────────────────

def run_pipeline(case: Case, agent) -> Trace:
    """
    跑完整流水线：路由 → 多 collection 检索 → AssessmentAgent.ask（含 critic loop）。

    关键点：复用 AssessmentAgent.ask 直接拿到 critic trace（last_iterations / last_critic_actions 等），
    这样 eval 评估的就是真实生产路径，不再单独拼 prompt。

    Trace 包裹：设置环境变量 RAG_TRACE_DIR=eval/traces/ 即启用 JSONL trace，
    每条 case 一个文件，含所有 LLM 调用的 prompt/response/component。
    """
    with start_trace(case.query, metadata={"case_id": case.id}):
        return _run_pipeline_inner(case, agent)


def _run_pipeline_inner(case: Case, agent) -> Trace:
    router = agent.router

    # 重置 agent history（每条 case 独立评估）
    agent.reset()

    # 1. 路由（独立调一次，记录 predicted_layers；agent 内部还会再调一次，
    #    可以接受这点冗余以保持 trace 完整）
    predicted_layers = router.classify(case.query)

    # 2. 收集 chunks（用于 score_retrieval；agent 内部也会查，但我们要拿到 chunks 列表）
    all_chunks: list[RetrievedChunk] = []
    chunks_by_collection: dict[str, list[RetrievedChunk]] = {}
    for layer in predicted_layers:
        for cname in LAYER_TO_COLLECTIONS.get(layer, []):
            chunks = router._retriever.query(case.query, cname, TOP_K)
            chunks_by_collection[cname] = chunks
            all_chunks.extend(chunks)
    rf_chunks = router._retriever.query(case.query, COLLECTIONS["Red_Flags"], 3)
    chunks_by_collection[COLLECTIONS["Red_Flags"]] = rf_chunks
    all_chunks.extend(rf_chunks)

    # 3. 通过 AssessmentAgent.ask 走完整 critic loop
    answer = agent.ask(case.query)

    return Trace(
        case=case,
        predicted_layers=predicted_layers,
        chunks=all_chunks,
        chunks_by_collection=chunks_by_collection,
        answer=answer,
        critic_iterations=agent.last_iterations,
        critic_actions=list(agent.last_critic_actions),
        critic_issues=list(agent.last_critic_issues),
    )


# ── LLM judge ───────────────────────────────────────────────────────────────

def llm_judge_json(llm: OpenAI, prompt: str, max_tokens: int = 400) -> dict:
    """调 Llama judge，强制 JSON 输出。失败返回空 dict。"""
    try:
        resp = llm.chat.completions.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or "{}"
        return json.loads(raw)
    except Exception as e:
        print(f"  [judge error] {e}")
        return {}


# ── 评分 ────────────────────────────────────────────────────────────────────

def score_routing(trace: Trace) -> tuple[bool, float]:
    gold = set(trace.case.gold_layers)
    pred = set(trace.predicted_layers)
    strict = (gold == pred)
    union = gold | pred
    jaccard = len(gold & pred) / len(union) if union else 0.0
    return strict, jaccard


def score_retrieval(trace: Trace, llm: OpenAI) -> tuple[float, dict[str, bool]]:
    """
    批量版：一次 LLM 调用同时判断「每个 gold_concept 是否被任一 chunk 实质性承载」。

    旧 N×M 版（嵌套循环每 chunk × 每 concept 各调一次 judge）总开销 150-300 次 LLM 调用 / eval，
    免费档配额直接爆。批量版砍到 15 次（每 case 1 次），快 10 倍，省 80% token。

    Trade-off：LLM 在长 context 多目标判断会有轻微准确率损失（漏看 chunk 或编号混淆）。
    eval 是观察 trend 的工具，不需要 100% 精确。
    """
    chunks = trace.chunks[:15]
    if not chunks or not trace.case.gold_concepts:
        return 0.0, {c: False for c in trace.case.gold_concepts}

    chunks_text = "\n\n".join(
        f"[chunk_{i}] ({c.folder}) {c.text[:400]}"
        for i, c in enumerate(chunks)
    )
    concepts_list = "\n".join(f"- {c}" for c in trace.case.gold_concepts)

    prompt = f"""判断下面每个知识点是否被任一片段实质性承载。

知识点列表：
{concepts_list}

文献片段（编号 chunk_0 到 chunk_{len(chunks)-1}）：
{chunks_text}

对每个知识点，返回一个布尔值（true=任一片段实质性承载该知识点，false=没有任何片段实质讨论它）。
仅出现关键词但语义无关算 false。

返回严格 JSON（key 必须和知识点列表一一对应）：
{{
  "coverage": {{"<知识点1>": true/false, "<知识点2>": true/false, ...}}
}}"""

    result = llm_judge_json(llm, prompt, max_tokens=400)
    cov = result.get("coverage", {})
    hits = {c: bool(cov.get(c, False)) for c in trace.case.gold_concepts}
    recall = sum(hits.values()) / len(hits) if hits else 0.0
    return recall, hits


def score_generation(trace: Trace, llm: OpenAI) -> dict:
    case = trace.case
    # 拼参考片段（生成 judge 看的是 trace 实际看到的资料）
    ref_parts = [c.text for c in trace.chunks[:10]]
    ref_text = "\n---\n".join(ref_parts)[:3000]

    prompt = f"""你是临床动作评估的回答质检员。

用户问题：{case.query}

参考片段（合并）：
{ref_text}

AI 回答：
{trace.answer}

应覆盖知识点：{case.gold_concepts}
应触发红旗：{case.gold_red_flag}

返回严格 JSON：
{{
  "concept_coverage": {{"<concept>": 0|0.5|1, ...}},
  "hallucination_spans": ["<回答中无法被参考片段或常识支持的具体断言>", ...],
  "red_flag_handling": "correct|missed|false_alarm|n/a",
  "overall_score": 0到5的整数,
  "reason": "<80字内总评>"
}}

评分标准：
- concept_coverage: 0=完全没提 / 0.5=只提到名词 / 1=有充分讨论
- hallucination_spans: 列出所有"参考片段没说、常识也支持不了"的具体断言
- red_flag_handling: 应触发红旗时"correct"=明确建议就医 / "missed"=没建议 / "false_alarm"=不该触发但触发了；不需要红旗时回 "correct" 或 "false_alarm"
"""
    return llm_judge_json(llm, prompt, max_tokens=600)


def evaluate_case(case: Case, agent, llm: OpenAI, console: Console) -> tuple[CaseScore, Trace]:
    console.print(f"\n[bold cyan]▶ {case.id}[/bold cyan] {case.query[:50]}...")

    trace = run_pipeline(case, agent)
    console.print(f"  路由命中：{trace.predicted_layers}（gold: {case.gold_layers}）")
    console.print(f"  检索 chunk 数：{len(trace.chunks)}")
    if trace.critic_actions and trace.critic_actions[0] != "disabled":
        console.print(f"  Critic: {trace.critic_iterations} 轮，actions={trace.critic_actions}")

    # 路由
    strict, jaccard = score_routing(trace)

    # 检索
    recall, hits = score_retrieval(trace, llm)

    # 生成
    gen = score_generation(trace, llm)
    cov = gen.get("concept_coverage", {})
    cov_avg = sum(float(v) for v in cov.values()) / len(cov) if cov else 0.0

    score = CaseScore(
        case_id=case.id,
        layer_strict=strict,
        layer_jaccard=jaccard,
        concept_recall=recall,
        concept_hits=hits,
        gen_concept_coverage=cov_avg,
        gen_concept_breakdown=cov,
        gen_hallucination_count=len(gen.get("hallucination_spans", [])),
        gen_hallucination_spans=gen.get("hallucination_spans", []),
        gen_red_flag_handling=gen.get("red_flag_handling", "n/a"),
        gen_overall=float(gen.get("overall_score", 0)),
        gen_reason=gen.get("reason", ""),
    )
    return score, trace


# ── 报告渲染 ────────────────────────────────────────────────────────────────

def render_summary_table(scores: list[CaseScore], console: Console):
    table = Table(title="RAG 评估总览", show_lines=True)
    table.add_column("用例", style="cyan", no_wrap=True)
    table.add_column("路由", justify="center")
    table.add_column("Jaccard", justify="right")
    table.add_column("检索 recall", justify="right")
    table.add_column("生成覆盖", justify="right")
    table.add_column("幻觉", justify="right")
    table.add_column("红旗", justify="center")
    table.add_column("总分", justify="right")

    for s in scores:
        rf_emoji = {"correct": "✓", "missed": "✗ missed", "false_alarm": "✗ false", "n/a": "-"}.get(
            s.gen_red_flag_handling, "?"
        )
        table.add_row(
            s.case_id,
            "✓" if s.layer_strict else "✗",
            f"{s.layer_jaccard:.2f}",
            f"{s.concept_recall:.2f}",
            f"{s.gen_concept_coverage:.2f}",
            str(s.gen_hallucination_count),
            rf_emoji,
            f"{s.gen_overall:.1f}/5",
        )

    console.print()
    console.print(table)

    # 总体均值
    n = len(scores)
    if n == 0:
        return
    avg_strict = sum(1 for s in scores if s.layer_strict) / n
    avg_jaccard = sum(s.layer_jaccard for s in scores) / n
    avg_recall = sum(s.concept_recall for s in scores) / n
    avg_cov = sum(s.gen_concept_coverage for s in scores) / n
    halluc_rate = sum(1 for s in scores if s.gen_hallucination_count > 0) / n
    avg_overall = sum(s.gen_overall for s in scores) / n

    console.print()
    console.print("[bold yellow]═══ 总体均值 ═══[/bold yellow]")
    console.print(f"  路由严格命中率: {avg_strict:.1%}")
    console.print(f"  路由 Jaccard:    {avg_jaccard:.2f}")
    console.print(f"  [bold]检索 concept recall@5: {avg_recall:.2f}[/bold]  ← 关键指标")
    console.print(f"  生成 concept 覆盖: {avg_cov:.2f}")
    console.print(f"  幻觉发生率:        {halluc_rate:.1%}")
    console.print(f"  生成总分:          {avg_overall:.1f}/5")


def write_markdown_report(scores: list[CaseScore], traces: list[Trace], path: Path):
    n = len(scores)
    if n == 0:
        path.write_text("# RAG 评估报告\n\n无用例。\n", encoding="utf-8")
        return

    avg_strict = sum(1 for s in scores if s.layer_strict) / n
    avg_jaccard = sum(s.layer_jaccard for s in scores) / n
    avg_recall = sum(s.concept_recall for s in scores) / n
    avg_cov = sum(s.gen_concept_coverage for s in scores) / n
    halluc_rate = sum(1 for s in scores if s.gen_hallucination_count > 0) / n
    avg_overall = sum(s.gen_overall for s in scores) / n
    rf_cases = [s for s in scores if s.gen_red_flag_handling != "n/a"]
    rf_correct = sum(1 for s in rf_cases if s.gen_red_flag_handling == "correct")

    lines = [
        f"# RAG 评估报告  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**模型：** `{CLAUDE_MODEL}`（生成 + judge 同源——见末尾局限说明）",
        "",
        "## 总览",
        "",
        f"用例数：**{n}**",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 路由严格命中 | {avg_strict:.1%} |",
        f"| 路由 Jaccard | {avg_jaccard:.2f} |",
        f"| **检索 concept recall@5** | **{avg_recall:.2f}** |",
        f"| 生成 concept 覆盖 | {avg_cov:.2f} |",
        f"| 幻觉发生率 | {halluc_rate:.1%} |",
        f"| 红旗处置正确率 | {rf_correct}/{len(rf_cases)} |",
        f"| 生成总分 | {avg_overall:.1f}/5 |",
        "",
    ]

    # Critic 反思统计（只在启用时有意义）
    critic_traces = [t for t in traces if t.critic_actions and t.critic_actions[0] != "disabled"]
    if critic_traces:
        avg_iter = sum(t.critic_iterations for t in critic_traces) / len(critic_traces)
        triggered = [t for t in critic_traces if t.critic_iterations > 1]
        from collections import Counter
        action_counts: Counter = Counter()
        for t in critic_traces:
            for a in t.critic_actions:
                if a not in ("ok", "disabled", "max_iter_fallback"):
                    action_counts[a] += 1
        lines.extend([
            "## Critic 反思统计",
            "",
            f"- 启用 critic 的 case 数: **{len(critic_traces)}**",
            f"- 平均迭代轮数: **{avg_iter:.2f}**（1 = 一次过，>1 = 反思过）",
            f"- 触发反思的 case: **{len(triggered)}/{len(critic_traces)}**",
            f"- 修正动作分布: {dict(action_counts)}",
            "",
        ])

    lines.extend([
        "## 失败模式分类",
        "",
    ])

    # 区分两类失败
    routing_ok_retrieval_bad = [s for s in scores if s.layer_strict and s.concept_recall < 0.5]
    routing_bad_retrieval_ok = [s for s in scores if not s.layer_strict and s.concept_recall >= 0.5]

    if routing_ok_retrieval_bad:
        lines.append("**路由对了但检索没召回（疑似 embedding 瓶颈）：**")
        for s in routing_ok_retrieval_bad:
            missed = [c for c, hit in s.concept_hits.items() if not hit]
            lines.append(f"- `{s.case_id}` — 漏召回: {missed}")
        lines.append("")

    if routing_bad_retrieval_ok:
        lines.append("**路由错了但检索误打误撞拿到 concept（疑似 classify() prompt 问题）：**")
        for s in routing_bad_retrieval_ok:
            lines.append(f"- `{s.case_id}` — Jaccard {s.layer_jaccard:.2f}")
        lines.append("")

    # 逐条
    lines.extend(["## 逐条详情", ""])
    for score, trace in zip(scores, traces):
        case = trace.case
        lines.extend([
            f"### {score.case_id}",
            "",
            f"**Query:** {case.query}",
            "",
            f"- 期望层：`{case.gold_layers}` ｜ 实际：`{trace.predicted_layers}` "
            f"{'✓' if score.layer_strict else f'✗ (Jaccard {score.layer_jaccard:.2f})'}",
            f"- 检索 recall: **{score.concept_recall:.2f}**",
        ])
        for concept, hit in score.concept_hits.items():
            lines.append(f"  - {concept}: {'✓' if hit else '✗'}")
        lines.extend([
            f"- 生成 concept 覆盖: {score.gen_concept_coverage:.2f}",
        ])
        for concept, v in score.gen_concept_breakdown.items():
            lines.append(f"  - {concept}: {v}")

        if score.gen_hallucination_spans:
            lines.append(f"- **幻觉 ({score.gen_hallucination_count})：**")
            for sp in score.gen_hallucination_spans:
                lines.append(f"  - {sp}")
        else:
            lines.append("- 幻觉：无")

        rf_label = {"correct": "✓ 正确", "missed": "✗ 漏报", "false_alarm": "✗ 误报", "n/a": "—"}.get(
            score.gen_red_flag_handling, "?"
        )
        lines.extend([
            f"- 红旗处置：{rf_label} (期望 {case.gold_red_flag})",
            f"- 生成总分：{score.gen_overall:.1f}/5",
            f"- Judge 评语：{score.gen_reason}",
        ])
        if trace.critic_actions and trace.critic_actions[0] != "disabled":
            lines.append(f"- Critic: {trace.critic_iterations} 轮，actions={trace.critic_actions}")
            if trace.critic_issues:
                for i, issues in enumerate(trace.critic_issues, 1):
                    if issues:
                        lines.append(f"  - 第 {i} 轮发现: {issues}")
        lines.extend([
            "",
            "<details><summary>AI 回答全文</summary>",
            "",
            "```",
            trace.answer.strip(),
            "```",
            "",
            "</details>",
            "",
        ])

    lines.extend([
        "## 局限说明",
        "",
        "- Judge 模型与生成模型同源（都是 Llama-3.3-70b），对幻觉的检测会系统性偏低估。",
        "  关键决策版本前建议人工抽检 3-5 条幻觉相关 case，或用 Claude 复核。",
        "- 检索 recall 用的是「任一 chunk 命中即算覆盖」，不区分 precision。",
        "  若需要 precision 信息，需要扩展评分逻辑。",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG 三段式评估")
    parser.add_argument("--testset", default="eval/testset.yaml")
    parser.add_argument("--out", default="eval/eval_results.md")
    parser.add_argument("--sample", type=int, default=None,
                        help="只跑前 N 条用例（调试用）")
    parser.add_argument("--no-critic", action="store_true",
                        help="禁用 Critic agent 反思环（用于和有 critic 的版本做对比）")
    args = parser.parse_args()

    console = Console()

    # 1. 加载测试集
    testset_path = Path(args.testset)
    if not testset_path.exists():
        console.print(f"[red]找不到测试集：{testset_path}[/red]")
        sys.exit(1)
    cases = load_testset(testset_path)
    if args.sample:
        cases = cases[:args.sample]
    console.print(f"[green]加载 {len(cases)} 条用例[/green]")

    # 2. 初始化
    from rag_system.agent import AssessmentAgent
    enable_critic = not args.no_critic
    console.print(f"[dim]Critic agent: {'enabled' if enable_critic else 'DISABLED'}[/dim]")
    agent = AssessmentAgent(enable_critic=enable_critic)
    llm = OpenAI(
        api_key=__import__("os").environ.get("GROQ_API_KEY"),
        base_url=GROQ_BASE_URL,
    )

    # 3. 逐条评估
    scores: list[CaseScore] = []
    traces: list[Trace] = []
    t0 = time.time()
    for case in cases:
        try:
            score, trace = evaluate_case(case, agent, llm, console)
            scores.append(score)
            traces.append(trace)
        except Exception as e:
            console.print(f"[red]✗ {case.id} 失败：{e}[/red]")
            import traceback; traceback.print_exc()
    elapsed = time.time() - t0
    console.print(f"\n[dim]总耗时 {elapsed:.1f}s[/dim]")

    # 4. 渲染
    render_summary_table(scores, console)

    # 5. 写 markdown
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(scores, traces, out_path)
    console.print(f"\n[green]详细报告已写入：[/green]{out_path}")


if __name__ == "__main__":
    main()
