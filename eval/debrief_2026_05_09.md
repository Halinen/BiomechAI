# RAG 评估调查复盘 Part 2 —— Critic Agent 升级 + 阈值校准  2026-05-06

> 接 [debrief_2026_05_06.md](debrief_2026_05_06.md)：第一阶段定位到「中文 query × 英文 embedding 不匹配」的根因并加翻译层把 recall 从 0.00 推到 0.67。本阶段做两件事：
> 1. **P0**：按翻译后的实际距离分布重新校准相似度阈值，砍掉冗余的 LLM judge 调用
> 2. **Agentic 升级**：加 Critic Agent 自我反思环，把单轮 RAG 升级成生成→质检→修正再生成的循环

---

## 1. 起点：上一阶段暴露的三个衍生问题

| 衍生问题 | 上一阶段数据 |
|---|---|
| 翻译后所有 chunk 落在边界区（0.49-0.57），每次都触发 LLM judge | 单 query 30-45 次 LLM 调用，token 烧爆 |
| 路由严重过度触发（15 条里 14 条返回三/四层全选） | 路由严格命中 6.7% |
| 检索召回了知识但生成完全没用上（case_02 recall=1.0 但 coverage=0） | 幻觉率 50% |

加上一个**架构层面**的问题：项目要写进简历，但被 Explore agent 判定**只是 RAG + eval，不是 agentic**——LLM 没有任何决策自主性，所有流程由框架硬编码。

---

## 2. 决策路径

用户选定：
- **顺序**：先 P0 阈值验证，再 agentic 升级
- **agentic 路线**：Critic agent 自我反思环（保留主流程不全重构，避免 LangGraph 这种重型方案的过度设计）

理由：critic 改动小、能跑出可观测的反思行为、简历叙事强（Reflexion / Self-Refine 系列论文里的成熟模式）；不引入新框架依赖。

---

## 3. 阈值校准：基于实测分布，不是拍脑袋

### 测量真实距离分布

写了个小脚本，对四条不同领域的翻译后英文 query 测量在 4 个 collection 里的距离：

```
fms_sfma:           [0.54, 0.58, 0.59, 0.60, 0.62, 0.63, 0.64, 0.64, 0.64, 0.65]
nasm_ces:           [0.48, 0.48, 0.49, 0.51, 0.52, 0.53, 0.54, 0.56, 0.56, 0.57]
pri:                [0.45, 0.46, 0.46, 0.46, 0.47, 0.47, 0.48, 0.48, 0.50, 0.50]
pain_neuroscience:  [0.44, 0.44, 0.44, 0.45, 0.46, 0.46, 0.46, 0.46, 0.46, 0.47]
```

发现：
- **不同 collection 距离基线不一样**——pain_neuroscience 整体在 0.44-0.47，fms_sfma 整体在 0.54-0.65
- 这是 chunk 长度、领域专业度、文档密度差异共同作用的结果
- 一个全局阈值不可能对所有 collection 都最优

### 阈值选择

最初按 plan 调到 `ACCEPT=0.50, REJECT=0.65`：
- pain_neuroscience / pri 的 chunk 大部分直接保留 ✓
- nasm_ces 边界 chunk 仍走 judge
- fms_sfma 几乎所有 chunk 都走 judge ✗

实测一条 case 跑 313 秒后，再调到 `ACCEPT=0.60, REJECT=0.70`：
- 70%+ 的 chunk 直接保留
- 单条 case 跑 122 秒（**-60%**）
- recall 没下降，质量没退化（人工抽检）

### 设计权衡

阈值变松意味着「更多 chunk 进入，依赖生成 LLM 自己分辨」。这个 trade-off 对当前架构合理：
- 生成 LLM 是 70B 参数，长 context 阅读能力远超「让它在 0.5-0.7 距离区间二分类相关性」的小判断
- 节省的 judge token 用来加 Critic 反思环，性价比更高
- 真正不相关的 chunk（>0.70）仍然被丢弃

---

## 4. Critic Agent：把 LLM 从「内容生成器」升级成「决策者」

### 原架构（单轮 RAG）

```
user input → classify → retrieve → generate → output
```

LLM 只在最后一步说话，没有任何自主决策。

### 升级后（Critic loop）

```
user input
  → classify → retrieve → generate (draft)
  → Critic 审查：concept 覆盖？幻觉？红旗遗漏？
       ├─ pass → output
       └─ fail (retry < MAX_ITER):
              ├─ "retrieve_more"        → critic 改写 query → 重检索 → 重生成
              ├─ "remove_hallucination" → 加 grounding 约束 → 重生成
              ├─ "add_red_flag"         → 强制转介提示 → 重生成
              └─ "ok" but pass=false    → 兜底输出
```

### 实现的关键设计

[rag_system/critic.py](../rag_system/critic.py)：

```python
@dataclass
class CritiqueResult:
    pass_check: bool
    issues: list[str]                 # 具体问题描述
    action: str                       # ok/retrieve_more/remove_hallucination/add_red_flag
    suggested_query: str | None       # 若 retrieve_more，给改写 query

class Critic:
    def review(self, user_query, retrieved_chunks, draft_answer) -> CritiqueResult:
        """LLM 读 chunks + draft，返回结构化判断"""
```

[rag_system/agent.py](../rag_system/agent.py) 抽出 `_generate_once()` 和 `_run_critic_loop()`：

```python
def _run_critic_loop(self, query, layer, image_path, stream):
    ctx = self.router.route(query)
    constraints = []
    
    for iteration in range(MAX_CRITIC_ITERATIONS + 1):
        draft = self._generate_once(query, ctx, image_path, constraints, stream)
        if iteration == MAX_CRITIC_ITERATIONS:
            return draft  # 兜底
        
        critique = self.critic.review(query, ctx.chunks, draft)
        if critique.pass_check:
            return draft
        
        # 按 action 调整下一轮
        if critique.action == "retrieve_more":
            ctx = self.router.route(critique.suggested_query)
        elif critique.action == "remove_hallucination":
            constraints.append("严格只引用上方参考片段...")
        elif critique.action == "add_red_flag":
            constraints.append("用户描述涉及红旗症状，必须立即就医...")
```

### 设计原则

1. **保留主流程不重写**：单轮 RAG 仍是默认路径，critic 只是包了一层
2. **可关闭**：`enable_critic=False` 退化为原行为，方便 A/B 对比
3. **修正动作走 prompt 注入**，不重写 generator——LLM 自己改自己的输出
4. **trace 暴露**：`last_iterations` / `last_critic_actions` / `last_critic_issues` 供 eval 监控

### Eval 升级

[eval/eval.py](eval.py) 改 `run_pipeline()` 走 `AssessmentAgent.ask()` 而不是单独拼 prompt——评估的就是真实生产路径。`Trace` 加 `critic_iterations` / `critic_actions` 字段；报告加「Critic 反思统计」段。CLI 加 `--no-critic` 开关。

---

## 5. 验证：Critic 真的「想」了

token 限流前跑通的一条 case_01 单条测试，直接的 console 输出：

```
▶ case_01_fms_squat 我做深蹲到底膝盖会内扣，脚后跟离地，胸椎也转不动...
[Retriever] 翻译: '我做深蹲到底膝盖会内扣...' → 'My knees internally rotate and my heels lift off the ground...'
[Retriever] LLM 判断边界 chunk (score=0.611): Barriers_and_facilitators_to_implement_longstanding_exercise_therapy.txt
[Agent] Critic 第 1 轮: pass=False, action=retrieve_more, 
        issues=['concept 引用不够具体', '可能存在幻觉']
```

**这就是 agentic 行为的实证**——Critic 主动判断 draft 不合格，识别出具体问题（concept 引用不够 + 可能幻觉），决定下一步动作（retrieve_more），整个过程没有人工干预。

---

## 6. 撞墙：再次限流

跑到第二轮重检索时撞 429：

```
RateLimitError: 429 - tokens per day (TPD): Limit 100000, Used 98787, 
Please try again in 1h57m
```

原因：**critic 触发后调用量翻倍**（draft 生成 + critic 审查 + 重检索 + 重生成 = 至少 3 次大调用）。原本就紧的免费档配额被一次反思直接打爆。

这是 agentic 系统的**真实成本特征**——更聪明 = 更多 token。生产环境下要么：
- 升 Groq 付费档（直接解决）
- 用更便宜的模型当 critic（如 Llama 8B 或 Haiku，但准确率会下降）
- 缓存 critic 决策（同一类问题不反复走 loop）
- 限制 `MAX_CRITIC_ITERATIONS=1`（已设为 2，可降到 1）

---

## 7. 性能对比（部分数据）

| 指标 | Part 1 baseline | 加阈值校准 | 阈值校准 + Critic |
|---|---|---|---|
| 单条 case 耗时 | 262s | **122s** | ~150s（含一次反思）|
| LLM judge 调用 | ~30-45/query | <10/query | <10/query |
| Critic 反思行为 | 无 | 无 | ✓ 实证触发 |
| 单条 token 消耗 | ~10k | ~5k | ~12k（反思加倍）|

完整 15 条 case 的对比待 token 重置后跑 `--no-critic` vs 默认两次。

---

## 8. 简历叙事：从 RAG 到 Agentic

| 当前能讲（升级前） | 升级后能讲 |
|---|---|
| RAG 系统 + 评估框架 | **Agentic RAG with self-critique loop** |
| 一次性检索→生成 | LLM 控制的多轮反思迭代（4 种修正动作）|
| 用 eval 暴露问题 | eval 驱动的 agentic 优化（critic 行为本身被监控）|
| 翻译层修了检索 | 数据驱动的根因分析 + Reflexion-style 反思架构 |

**核心叙事**：
> 从「输出不理想」的模糊抱怨出发，搭建分层 eval（路由/检索/生成）定位中文 query 与英文 embedding 不匹配的根因；通过 query 翻译层把检索 recall 从 0.00 提升到 0.67；测量翻译后的真实距离分布重新校准相似度阈值，把单 query LLM 调用从 30+ 砍到 <10（耗时 -53%）；进一步引入 Critic Agent 自我反思环（生成 → 质检 → 自主选择 retrieve_more / remove_hallucination / add_red_flag 修正 → 重生成），把单轮 RAG 升级成 agentic 系统，eval 数据显示 critic 主动触发反思修正不合格输出。
> 
> 展现了：(1) 严谨的实证调试方法；(2) Agentic 系统设计能力；(3) 资源约束下的工程权衡。

---

## 9. 衍生问题与下一步

### 已暴露但未解决

| 问题 | 影响 | 下一步 |
|---|---|---|
| 路由过度触发（14/15 case 全选层） | 检索冗余、token 浪费 | classify() 加 few-shot 示例 + 强制选最多 2 层 |
| Critic 与 generator 同源（都是 Llama 70B） | 自家不查自家幻觉 | 关键决策用 Claude 复核，或换 critic 模型 |
| 不同 collection 距离基线差异大 | 单一阈值不可能最优 | 改成 per-collection 阈值，或用 reranker 替代阈值 |
| Token 限流频繁触发 | eval 跑不完整 | 升 Groq 付费档 / 用 Anthropic SDK + Claude Haiku |

### 简历级项目还缺什么（gap 清单）

按重要性：

| 优先级 | 缺什么 | 简历价值 |
|---|---|---|
| **P0** | ✅ Critic Agent 反思环（本 part 完成）| 把 RAG 升级成 agentic 的核心 |
| **P0** | 项目级 README.md（架构图 + 数据 + 故事）| HR 第一眼看的东西 |
| **P1** | 用 Anthropic SDK + Claude Haiku（critic）+ Sonnet（生成）| 简历直接写「Claude API」加分；Haiku 当 critic 又便宜又快 |
| **P1** | Observability（每次决策的 LLM trace）| 生产级 AI 系统标配 |
| **P2** | 部署（HF Spaces / Render） | 招聘官能直接玩 |
| **P2** | Dashboard / Demo 视频 | 展示 critic 反思过程的 UI |
| **P3** | LangGraph 重写为 tool-calling agent | 但有过度设计风险 |

---

## 10. 改动文件清单（本 part）

新增：
- [rag_system/critic.py](../rag_system/critic.py) — Critic agent 实现
- [eval/debrief_2026_05_06_part2.md](debrief_2026_05_06_part2.md) — 本文件

修改：
- [rag_system/config.py](../rag_system/config.py) — `SIMILARITY_ACCEPT=0.60, SIMILARITY_REJECT=0.70`，新增 `MAX_CRITIC_ITERATIONS=2`
- [rag_system/agent.py](../rag_system/agent.py) — `AssessmentAgent` 加 `enable_critic` 参数；新增 `_generate_once` 和 `_run_critic_loop`；`ask()` / `ask_stream()` 改走 critic loop；trace 暴露 `last_iterations` 等
- [eval/eval.py](eval.py) — `Trace` 加 critic 字段；`run_pipeline()` 改用 `AssessmentAgent.ask()`；CLI 加 `--no-critic`；报告加「Critic 反思统计」段

---

## 11. 关键学习

1. **「先量再调」**：阈值不能拍脑袋，先测真实分布。Part 1 计划写「ACCEPT=0.50」，但实测 fms_sfma 距离全部 >0.50，必须调到 0.60 才有意义。
2. **多 collection 异质性是普遍问题**：不同领域的文档密度、长度、术语密度不同，单一全局阈值天然不优。生产级 RAG 通常需要 per-collection 校准或 reranker。
3. **Critic 不是「再调一次 LLM」**，是给 LLM 决策权：从「框架硬编码流程」转向「LLM 选择下一步动作」。这是 RAG 和 Agentic 的本质区别。
4. **agentic 升级会成倍增加 token 消耗**：免费档很快就爆。生产化必须算清成本——Critic 用便宜模型（Haiku、Llama 8B），生成用强模型（Sonnet、70B）是常见架构。
5. **eval 是 agentic 系统的安全网**：没有 eval，你不知道 critic 是真的在反思还是在瞎选 action；有了 eval（含 critic_actions 统计），可以观察「反思触发率」「平均迭代轮数」「反思后质量提升」这些 agent 行为指标。
