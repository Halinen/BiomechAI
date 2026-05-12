# Fitting — Agentic RAG for Postural Assessment

A Chinese-language posture assessment system built on **RAG + self-critique agent loop**, designed to diagnose movement compensations across a four-layer framework (neuromuscular control / structural / output / pain sensitization).

## What makes this interesting

Most "RAG demos" stop at retrieve→generate. This project shows the full diagnostic engineering arc of building one for production:

1. **Built an evaluation pipeline first** — three independent metrics (routing accuracy / retrieval recall / generation quality) before optimizing anything
2. **Used eval data to locate the root cause** — Chinese queries had **0.00 concept recall** against an English-embedding index because `all-MiniLM-L6-v2` doesn't speak Chinese
3. **Solved it with a query translation layer** — not by rebuilding the entire vector store with a multilingual model; translated query → embed → match against English chunks, recall jumped to **0.67**
4. **Upgraded from single-shot RAG to agentic** — added a Critic agent that reviews each draft answer for concept grounding / hallucinations / red flag handling, and decides one of four actions (`ok` / `retrieve_more` / `remove_hallucination` / `add_red_flag`)

## Architecture

```
User query (Chinese)
       │
       ▼
[Translator]            ──→  English query
       │
       ▼
[Router / Classify]     ──→  one of {控制层, 结构层, 输出层, 神经敏化层}
       │
       ▼
[Retriever]             ──→  top-K chunks across mapped collections
       │   (cosine threshold gating; LLM judge for borderline)
       ▼
[Generator]             ──→  draft answer
       │
       ▼
[Critic agent]  ◀────── reads chunks + draft, returns action
       │
       ├─ pass → output
       │
       └─ fail → modify retrieval / inject grounding constraint /
                 add red flag → regenerate (up to MAX_CRITIC_ITERATIONS)
```

### Four-layer assessment framework

| Layer | Concept | Knowledge bases |
|---|---|---|
| 控制层 (Control) | Neuromuscular control, breathing patterns | PRI, Breathing Retraining |
| 结构层 (Structure) | Joint mobility, tissue restrictions, post-surgical adhesions | FMS/SFMA, Visceral Fascia |
| 输出层 (Output) | Compensatory movement patterns, muscle imbalances | NASM CES |
| 神经敏化层 (Pain neuroscience) | Central sensitization, fear-avoidance | Pain Neuroscience |
| Red flags (cross-cutting) | Triage criteria — always checked | Red Flags |

## Tech stack

- **LLM**: Llama-3.3-70b via Groq (default; designed to be model-agnostic)
- **Vector DB**: Chroma (local persistent)
- **Embedding**: `all-MiniLM-L6-v2` (sentence-transformers)
- **UI**: Gradio
- **Evaluation**: custom three-stage scorer with LLM-as-judge

## Repository layout

```
rag_system/
  config.py        Paths, models, similarity thresholds, layer mappings
  indexer.py       Chunk .txt → embed → store in Chroma
  retriever.py     Vector search + LLM judge for borderline + query translation
  assessment.py    Router: classify query → multi-collection retrieval
  agent.py         AssessmentAgent + Critic loop
  critic.py        Critic agent (self-critique LLM)
knowledge_base/    Seven knowledge bases (PMC papers, NASM blog, PRI articles)
eval/
  testset.yaml     15 mock cases covering all layers + red flags + edge traps
  eval.py          Three-stage scorer (routing / retrieval / generation)
  debrief_*.md     Investigation logs (read these for the full story)
build_index.py     One-time Chroma index builder
app.py             Gradio web UI
```

## Quick start

```bash
# 1. Setup
pip install -r requirements.txt
cp .env.example .env  # add GROQ_API_KEY

# 2. Build vector index (one-time, ~2 min)
py build_index.py

# 3. Run the UI
py app.py            # http://localhost:7863
```

## Running the eval

```bash
# Full pipeline with Critic agent
py -m eval.eval

# Baseline (no critic) for A/B comparison
py -m eval.eval --no-critic --out eval/eval_no_critic.md

# Debug: first 3 cases only
py -m eval.eval --sample 3
```

The eval produces both a terminal table (via `rich`) and a markdown report with per-case breakdown, hallucination spans, red-flag handling, and critic iteration traces.

## Observability

Set `RAG_TRACE_DIR` to enable structured LLM-call tracing:

```bash
export RAG_TRACE_DIR=eval/traces       # bash
$env:RAG_TRACE_DIR = "eval/traces"     # PowerShell
py -m eval.eval --sample 3
```

Each user query produces one JSONL file with every LLM call in the pipeline:

```
{"type": "trace_start", "trace_id": "a3f2", "query": "深蹲膝盖内扣", ...}
{"step": 1, "component": "retriever.translate", "model": "llama-3.3-70b", "response_preview": "My knees..."}
{"step": 2, "component": "router.classify", "extra": {"detected_layers": ["结构层"], "reason": "都是关节活动度问题"}}
{"step": 3, "component": "retriever.judge", "extra": {"chunk_score": 0.61, "kept": true}}
{"step": 4, "component": "agent.generate", "extra": {"streaming": false, "has_constraints": false}}
{"step": 5, "component": "critic.review", "extra": {"pass": false, "action": "retrieve_more"}}
{"step": 6, "component": "agent.generate", "extra": {"has_constraints": true}}
{"type": "trace_end", "total_steps": 6, "total_seconds": 14.2}
```

Use `rag_system.trace.summarize_trace(path)` to get aggregate metrics per query (LLM calls by component, total latency).

## Measured impact

| Stage | Metric | Before | After |
|---|---|---|---|
| Translation layer | Retrieval concept recall | 0.00 | 0.67 |
| Threshold calibration | Single-query LLM judge calls | 30-45 | ~5 |
| Threshold calibration | Single-query latency | 262s | 122s |
| Critic agent | Hallucination spans per answer | observable in eval | observable + agent self-corrects |

See [eval/debrief_2026_05_06.md](eval/debrief_2026_05_06.md) and [eval/debrief_2026_05_06_part2.md](eval/debrief_2026_05_06_part2.md) for the full reasoning trail behind each change.

## Known limitations

- **Router over-triggers**: classifies into 3-4 layers when 1-2 would suffice. Wastes retrieval and inflates token cost. Fix planned: few-shot exemplars + max-layers constraint.
- **Critic / Generator same model**: both are Llama-3.3-70b, so the critic underestimates its own hallucinations. Production deployment should use different model classes (e.g., Haiku critic + Sonnet generator).
- **Free-tier rate limits**: 100k tokens/day cap on Groq is a frequent blocker for full eval runs. Solved by either upgrading tier or implementing batched judge calls (already done for retrieval scoring).
- **Embedding language mismatch**: query translation layer is a pragmatic workaround. Long-term: switch to a multilingual embedding model and rebuild the index.

## Roadmap

| Priority | Item |
|---|---|
| Next | Anthropic SDK integration (Claude Haiku as critic, Sonnet as generator) |
| Next | Routing prompt with few-shot exemplars |
| Soon | Observability: structured trace of every LLM call per query |
| Soon | Deploy to Hugging Face Spaces |
| Later | LangGraph-style tool-calling agent (multi-step planning, not just self-critique) |

---

**Background**: Started as a way to learn RAG by building something real (a chatbot for postural assessment). Evolved into a deeper study of why "build a RAG" demos go wrong in production — and how systematic evaluation makes the problems debuggable rather than mysterious.
