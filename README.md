# GenRewrite

论文 *GenRewrite: Query Rewriting via Large Language Models* 的非官方实现。

项目使用 LLM 生成 SQL 改写，并尝试通过自然语言重写规则（NLR2）复用优化经验。目前已编写候选生成、语义纠正、规则分组和检索模块，但主流程仍有错误，尚不能完整运行论文方法。

## 安装与配置

使用 Python 3.12 和 uv 创建环境：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirement.txt
uv pip install --python .venv "httpx[socks]<0.28"
```

`psycopg2` 安装需要 `pg_config`。机器未安装 PostgreSQL 开发库时，可将依赖中的 `psycopg2==2.9.9` 替换为 `psycopg2-binary==2.9.9` 后安装。

模型未下载时，执行：

```bash
.venv/bin/python prepare/download.py
.venv/bin/python prepare/download_bert.py
```

两个脚本分别将 Longformer 和 BERT 保存到 `config_file/`。现有加载代码仍需修正路径，具体见后面的代码问题表。

在项目根目录的 `.env` 中配置 LLM：

```dotenv
GENREWRITE_API_KEY=your-api-key
GENREWRITE_API_BASE=https://api.openai.com/v1
GENREWRITE_MODEL=gpt-4o-mini
```

也可使用环境变量。读取逻辑见 [env.py](src/genrewrite/env.py)。

### 数据库配置（需自行创建，不入库）

`config_file/` 整个目录已被 [.gitignore](.gitignore) 忽略，**不会随仓库分发**，原因：

- `postgres.json`、`mysql.json` 含数据库明文密码（`dbname/user/password/host/port`），属于敏感凭据；
- `bert-base-uncased/`、`longformer/` 是模型权重（合计约 1GB），克隆后用上文脚本重新下载即可。

因此克隆仓库后必须**手动创建** `config_file/postgres.json`，否则评估模块无法连接数据库。当前评估代码只使用 PostgreSQL，MySQL 配置尚未接入。文件格式（字段读取见 [evaluate_rewrite.py](evaluate_module/evaluate_rewrite.py) 的 `connect_to_database`）：

```json
{
    "dbname": "your_database",
    "user": "your_user",
    "password": "your_password",
    "host": "127.0.0.1",
    "port": 5432
}
```

主流程位于 [pipeline.py](pipeline_module/pipeline.py)，示例入口为 [run.py](run.py)：

```bash
.venv/bin/python run.py
```

示例使用占位 SQL，且存在下文列出的流程错误，不能直接用来复现论文实验。`pyproject.toml` 中的 `genrewrite` 命令目前只输出 Hello。

若代理配置为 `ALL_PROXY=socks5h://...`，当前依赖版本可能报 `Unknown scheme for proxy URL`。可取消这两个变量后运行：

```bash
env -u ALL_PROXY -u all_proxy .venv/bin/python run.py
```

## 论文与代码对照

对照依据是仓库内的 [GenRewrite.pdf](GenRewrite.pdf)，作者为 Jie Liu 和 Barzan Mozafari，版本为 **arXiv:2403.09060v3（2025-12-02）**。下文章节和图号均指这个版本。

检查日期为 2026-09-16，覆盖主要 Python 模块和示例数据。候选返回值错误与评估参数错误已通过模拟调用确认；其余结论来自源码检查，未运行真实 LLM、数据库或基准实验。

当前主要缺少瓶颈分析、主导规则收益归因、SQLSolver / tester 验证，以及新规则用于后续查询的流程。现有等价判断来自 LLM，不能替代论文中的独立验证。`data/reportory.json` 是占位规则，`data/result.json` 是空结果模板。

| 状态 | 含义 |
| --- | --- |
| 已有实现 | 已有与论文描述相符的代码；不代表整套流程已运行通过。 |
| 部分实现 | 有相关代码，但步骤不完整、算法不同或没有接入主流程。 |
| 完全缺失 | 未找到对应实现；注释、模型文件和空模板不算实现。 |

### 功能对照

| 论文部分 / 功能 | 状态 | 代码位置 | 当前实现与差异 |
| --- | --- | --- | --- |
| §4、Algorithm 1：建议 → 纠正 → 评估的顶层流程 | 部分实现 | [pipeline.py](pipeline_module/pipeline.py)，`GenRewrite.run` | 有三阶段调用骨架，但评估调用多传一个实参；建议阶段还返回规则列表而非 SQL，流程不能按设计完成。 |
| Algorithm 1：从空仓库或预建仓库启动 | 部分实现 | `GenRewrite.__init__`；[reportory.json](data/reportory.json)；分组与选择模块的 `load_json` | 接受 JSON 路径，有预建示例结构；主流程不读取规则，空仓库会在分组模块的 `np.vstack([])` 处失败。 |
| §4.1、Figure 4：用自然语言 hints 引导重写 | 已有实现 | [suggest_candidate_rewrite.py](suggest_module/suggest_candidate_rewrite.py)，`suggest_and_explain` | prompt 包含输入 SQL、优化目标与 hints，允许 LLM 发现其他规则。 |
| §4.1：同时提取 NLR2 和解释 | 已有实现 | 同上，`total_number`、`rewrite_rule_*`、`rewrite_rule_*_explanation` | 请求并解析多条规则和解释；返回的是这两个列表，SQL 的返回问题另见下一项。 |
| §4.1：返回候选 SQL 并交给纠正模块 | 部分实现 | 同上，`Candidate rewrite` 字段与返回语句 | prompt 请求 SQL，但没有读取 `answer["Candidate rewrite"]`；名为 `candidate_rewrite` 的列表实际装的是规则。 |
| §4.1：去除表名、列名等查询特定细节 | 已有实现 | 同上，prompt 的 `must not include any specific query details` | 明确要求生成可复用规则，与论文的提示方式一致；示例仓库仍是占位内容。 |
| §4.1：记录规则、查询特征和观测收益 | 部分实现 | [suggest_group_rewrite.py](suggest_module/suggest_group_rewrite.py)，`add_rule_to_json`；`GenRewrite.update_rules` | JSON 写入规则、组和一条查询；主流程仅在内存追加 `(explanation, speedup)`，未保存特征、收益或调用 JSON 更新。 |
| §4.1：主导 NLR2 的单规则候选生成 | 完全缺失 | 建议与评估模块 | 无逐条规则作为唯一 hint 生成候选的流程。 |
| §4.1：用 EXPLAIN 成本匹配识别主导规则 | 完全缺失 | [evaluate_rewrite.py](evaluate_module/evaluate_rewrite.py)，`execute_query` | 虽有文本 `cost=` 提取，但不比较单规则候选与最终候选的计划成本，不能完成主导规则识别。 |
| §4.1：BERT 计划嵌入与 L2 距离回退 | 完全缺失 | 分组模块的 `embed`、`knn` | BERT / 欧氏距离用于规则文字检索，不是 EXPLAIN 计划相似度；没有论文的计划距离回退。 |
| §4.1：将收益记到主导 NLR2 | 完全缺失 | `GenRewrite.update_rules`；选择模块 `benefit = 1` | 不识别主导规则、不维护对应收益；把解释和数值放进列表不能替代归因。 |
| §4.1、Figure 5：LLM 判断规则是否语义同组 | 已有实现 | 分组模块 `predict_group` | 有“严格相同规则 / Unseen rule”的选择题及解释，提示与论文相符。 |
| §4.1：分组持久化、新组创建与减少重复评估 | 部分实现 | 分组模块 `knn`、`add_rule_in_group`、`add_rule_to_json` | BERT 先取不同组候选，再调用 LLM；遇到 Unseen rule 得到 `None`，后续 `int(None)` 失败；未接入主流程，也没有跳过重复单规则评估的流程。 |
| §4.2：从执行历史取得含实际运行统计的计划 | 完全缺失 | 选择模块；评估模块 | 没有执行历史存储或读取，也没有供瓶颈分析使用的 actual time / rows / loops。 |
| §4.2：裁剪计划中的无关字段 | 完全缺失 | 选择与建议模块 | 没有计划预处理、字段删除或对应 token 降低步骤。 |
| §4.2：LLM 总结查询最关键性能瓶颈 | 完全缺失 | 选择模块 `embed_query` | 直接嵌入 SQL 文本，没有“SQL + 裁剪计划 → 瓶颈总结”的 LLM 调用。 |
| §4.2：检索历史瓶颈总结的近邻 | 部分实现 | [suggest_select_rewrite.py](suggest_module/suggest_select_rewrite.py)，`get_top_k_similar_queries` | 有 Longformer 与余弦 top-k，但对象是原始 SQL，不是 v3 的瓶颈总结；每条规则只读取 `query_1`，会遗漏后续查询并可能重复索引同一查询。 |
| §4.2：LLM 二次选择同瓶颈查询 | 完全缺失 | 选择模块 `select_best_nlr2` | 无二阶段 LLM 确认；只有数值排序和组内选择。 |
| §4.2：无合适匹配时回退基本 prompt | 完全缺失 | `GenRewrite.run(hints)`；选择模块 | 主流程始终使用外部传入 hints，没有无匹配判定或自动回退；手动传空 hints 不等于该机制。 |
| §4.2：使用匹配查询的最有收益规则 | 部分实现 | 选择模块 `calculate_score`、`select_best_nlr2` | 有组评分，但 `benefit` 恒为 1，权重是余弦相似度的倒数；不是本地 v3 的“同瓶颈查询 → 最有收益规则”。 |
| §4.2：检索结果接入每轮建议 | 完全缺失 | `GenRewrite.suggest_select_rewrite` 与 `run` | 有包装方法，但 `run` 不调用它；各查询、各轮共用固定 hints。 |
| §4.3、Figure 6：分解语义并生成反例 | 已有实现 | [correct_candidate_rewrite.py](correct_module/correct_candidate_rewrite.py)，`perform_semantic_correction` | 要求逐步理解原查询和改写查询，给出表形式反例及等价判断；对应论文语义纠正提示。 |
| §4.3：基于反例与历史迭代修正 | 已有实现 | 同上；[gpt.py](pipeline_module/gpt.py)，`get_GPT_response_with_history` | 不等价时将分析作为历史交给 LLM 修正 `q2`，下一轮重新判断；本地有有限尝试次数。仅是 LLM 推理，不构成等价证明。 |
| §4.3：先语义、后语法的两阶段纠正 | 部分实现 | `Nlr2Correction.correct_query` | 阶段顺序一致，但语法阶段只有一次 LLM 自检，语义失败返回 `None` 后主流程不处理。 |
| §4.3：EXPLAIN 错误反馈驱动迭代语法纠正 | 部分实现 | `perform_syntax_correction` | 有“检查语法并修正”的 prompt；没有执行 EXPLAIN、传入数据库错误消息或循环至可执行。注释声称 EXPLAIN feedback，实际没有。 |
| §4.4：SQLSolver 正确性验证 | 完全缺失 | 评估模块 `check_if_equiv`；依赖配置 | 等价门仅调用 LLM；没有 SQLSolver 适配、调用或验证结果处理。 |
| §4.4：UNKNOWN 时回退 SlabCity tester | 完全缺失 | 评估模块；pipeline 中被注释的 `self.tester` | SlabCity 仅出现在注释，无输入生成、两查询执行比较或 tester 调用。 |
| §4.4：将语义阶段反例纳入 tester | 完全缺失 | 纠正模块与评估模块 | 反例只进入 LLM 对话，不转成测试数据或验证用例。 |
| §4.4：验证未通过则弃用；先正确性后性能 | 部分实现 | `GenRewrite.evaluate_rewrite`、`run` | 有布尔等价判断，但不是独立验证；即使 LLM 判不等价，仍然调用性能评估，没有严格先后检查顺序。 |
| §4.4：静态成本代理预筛选 | 完全缺失 | 评估模块 | 提取成本后只是输出，没有依据成本决定是否避免昂贵执行。 |
| §4.4：目标数据库上的离线性能测量 | 部分实现 | `connect_to_database`、`execute_query`、`evalutate` | 有 PostgreSQL 连接和计时代码；不自动加 EXPLAIN，普通结果会被当成计划文本，不能可靠测量候选。主流程还传错数据库配置路径。 |
| §4.4：never-worse-off 与原查询回退 | 部分实现 | `run` 中 `speedup > self.min_speedup` | 有阈值收集意图，但指标定义不同；没有明确的性能检查顺序、异常回退与已验证候选应用流程。 |
| Algorithm 1：预算耗尽或结果稳定时终止 | 部分实现 | `GenRewrite.budget`；`run` | 保存预算但从不消耗或检查；没有“Res 不变化”判定，未成功优化的查询会持续重试。 |
| Figure 1：最终验证与可选人工检查 | 完全缺失 | pipeline 与评估模块 | 无独立最终验证步骤、审核状态或人工检查接口。v3 引言描述人工确认，§4.4 又将探索场景人工复核列为可选；不将人工检查理解为所有部署必需。 |
| §5.1：论文模型设置 | 部分实现 | `GPT`；[env.py](src/genrewrite/env.py) | 模型可配置，默认 `gpt-4o-mini`；本地 v3 默认 `o3-mini`，wrapper 固定 `temperature=0.0`。未验证论文模型与当前 API 参数组合的兼容性。 |
| §5.1：TPC-DS、JOB 与 SQLStorm 工作负载 | 完全缺失 | [run.py](run.py)；数据目录 | 只有占位查询和样例规则，无基准 SQL、数据生成 / 加载或 SQLStorm 慢查询选择流程。 |
| §5.1：统一实验测量协议 | 完全缺失 | 评估模块 `evalutate` | 不含 PostgreSQL 14.17 环境约束、VACUUM ANALYZE、每条查询三次取均值或四个候选的实验控制；每对查询只计时一次。 |
| §5.2–§5.6：基线、消融、开销与阈值敏感性 | 完全缺失 | 全部可见执行代码；[result.json](data/result.json) | 无基线运行器、消融开关、覆盖率 / 几何平均加速比统计、token / 美元开销记录或实验产物；结果 JSON 只是 null 模板。 |
| Appendix A：五类重写模式的已验证案例 | 完全缺失 | 示例数据；`run.py` 固定 hints | 有类似“预计算聚合”“移除 UNION ALL”的通用提示，但无附录查询对、实测结果或验证证据。LLM 可能生成这些模式不能算仓库已经实现案例。 |
| Appendix B：多查询对提炼带前置条件的可执行规则 | 完全缺失 | 规则仓库和建议模块 | 没有 AST / plan 公共变换提取、guards 或可执行规则生成。这是附录提出的扩展方向，不是顶层管线必需模块。 |

### 代码问题

| 优先级 | 问题与位置 | 错误原因或复现方式 | 影响与修改方向 |
| --- | --- | --- | --- |
| P0 | 候选 SQL 未返回：`suggest_candidate_rewrite.suggest_and_explain` | 用包含 `Candidate rewrite: SELECT 1` 和一条 NLR2 的模拟响应调用，返回 `(['Remove redundant joins'], ['Explanation'])` | 纠正模块接到规则列表；应分别返回 SQL、规则和解释。 |
| P0 | 主流程评估签名错误：`GenRewrite.run` | 实际调用 `self.evaluate_rewrite(query, query, corrected_query)`，方法仅收两个业务参数；隔离调用复现 `TypeError: ... takes 3 positional arguments but 4 were given` | 进入评估阶段必定报错；应仅传原 SQL 和修正 SQL。 |
| P0 | 仓库路径被当作数据库配置：`GenRewrite.evaluate_rewrite` | `Evaluate_rewrite_model(self.json_path)`，而 `self.json_path` 是 `data/reportory.json`；连接函数需要 `dbname/user/password/host/port` | 即使修好签名也会缺少数据库字段；应分别维护数据库配置和规则仓库路径。 |
| P0 | Unseen rule 无法创建新组：分组模块 | `self.hash.get('Unseen rule')` 得到 `None`；`add_rule_to_json` 使用 `int(group_id)` | 新知识无法正常入库；应显式创建新组 ID，空仓库应可直接开始。 |
| P0 | 查询执行与结果解析不一致：`execute_query` | 直接执行传入 SQL，再对 `row[0]` 做 `"cost=" in line` | 普通数值结果可能触发 TypeError；字符串结果通常得不到成本；应将计划获取与运行时测量分开并解析明确格式。 |
| P1 | 加速指标和论文不同：`evalutate` | 当前 `(t_original - t_rewrite) / t_original`；10 秒变 5 秒得到 0.5；论文 2x 对应 `t_original / t_rewrite = 2` | 阈值、统计与论文不能直接比较。代码的 `min_speedup=0.2` 是缩短耗时超过 20%，对应倍数超过 1.25x，不能直接读成论文的倍数阈值。 |
| P1 | 失败路径及终止条件未完成：`run`、`correct_query`、`evalutate` | 预算未使用；Res 不变仍循环；纠正或数据库失败可返回 None；计时结果无完整失败检查 | 可能无限重试、比较 None 或进行无效计算；应拒绝失败候选，并按预算或稳定结果结束。 |
| P1 | 检索组内选择可能越界：`select_best_nlr2` | 对组内每条规则执行 `top_k_queries.index(associated_query)`；该查询不一定在 top-k 内 | 组跨多个查询时可能 ValueError；候选范围和关联数据需一致。 |
| P1 | 余弦权重方向反转：`calculate_score` | 近邻按较大 cosine 选取，权重却是 `1 / similarity` | 正相似度越小权重越大；0 会产生无效除法，负值会导致不合理权重；且这一组评分不是本地 v3 的瓶颈检索流程。 |
| P1 | 模型路径与长度：选择、分组模块 | Longformer 路径为相对当前目录的 `../config_file/longformer/`，根目录运行指向仓库外；`max_length=512`；BERT 使用模型名而非仓库内下载路径 | 下载的模型未正确加载；长 SQL 超过 512 token 会被截断。应使用明确资源路径和与特征对象匹配的编码方案。 |
| P1 | embedding 不稳定：两个 embedding 模块 | 未调用 `model.eval()`；Longformer 仅 `torch.no_grad()`，BERT 仅末尾 detach | no_grad / detach 不关闭 dropout，近邻结果可能波动；不宜当作稳定特征检索。 |
| P1 | 规则库与主流程断开 | `run` 不调用 `suggest_select_rewrite` / `suggest_group_rewrite`，`update_rules` 只更新内存列表 | 成功重写不会影响后续 hints，论文的跨查询知识迁移流程缺失。 |
| P2 | prompt 字段名不一致：候选模块 | 模板有 `rewrite_rule 2_explanation`，读取要求 `rewrite_rule_2_explanation` | LLM 照模板返回第二条解释时可能 KeyError；需要统一响应字段与解析契约。 |
| P2 | 入口与运行说明不同 | `pyproject.toml` 的 `genrewrite` 指向 `src/genrewrite/__init__.py:main`，只输出 Hello；`run.py` 使用占位 SQL | 安装命令入口不运行重写流程；运行现有示例也不代表论文可复现。 |

P0：流程无法继续；P1：结果或算法有误；P2：接口和使用问题。

## 代码中采用、但论文未描述的方法

以下对照本地论文 v3，只列方法上的差异。配置读取、模型下载、路径处理等辅助代码不在此列。这些内容大多是对论文步骤的替代实现，并非新增的独立功能；对早期论文版本是否存在相同设计，本次未作判断。

| 代码中的方法 | 文件 / 函数 | 论文 v3 的做法 | 处理建议 |
| --- | --- | --- | --- |
| 倒余弦加权的规则组评分 | [suggest_select_rewrite.py](suggest_module/suggest_select_rewrite.py)，`calculate_score` | 代码以 `1 / similarity` 加权，结合规则交集和固定 `benefit=1` 计算组分数。§4.2 没有这套评分公式，而是用 LLM 确认瓶颈匹配，再取匹配查询的收益规则 | 实现论文检索方法后，移除这套评分 |
| 按原始 SQL 相似度选多个规则组 | 同文件，`embed_query`、`get_top_k_similar_queries`、`select_best_nlr2` | 代码直接编码 SQL，选 top-k 组和组内规则。§4.2 检索的是性能瓶颈总结，再由 LLM 确认匹配查询 | 将检索对象改为瓶颈总结，补上二次确认和收益规则选择 |
| 所有查询共用四条手写提示 | [run.py](run.py) 的 `hints`；[pipeline.py](pipeline_module/pipeline.py)，`run(hints)` | §4 使用规则仓库为当前查询选择提示，无合适匹配时使用基本 prompt。自然语言提示本身属于论文功能，但示例中始终使用固定四条提示的策略不是论文流程 | 从规则仓库动态选择提示，替换固定提示策略 |
| 评估阶段额外调用 LLM 判断等价 | [evaluate_rewrite.py](evaluate_module/evaluate_rewrite.py)，`check_if_equiv` | §4.3 用 LLM 等价判断辅助语义纠正；§4.4 的最终正确性检查使用 SQLSolver，并在 UNKNOWN 时回退 tester，没有代码中的这次 LLM 最终判定 | 用独立验证替换评估阶段的 LLM 判定，保留语义纠正阶段的 LLM 推理 |
| 不使用 EXPLAIN 反馈的一次性语法自检 | [correct_candidate_rewrite.py](correct_module/correct_candidate_rewrite.py)，`perform_syntax_correction` | §4.3 根据真实 EXPLAIN 错误迭代修复。代码仅询问 LLM 是否有语法错误，未提供数据库反馈 | 改为 EXPLAIN 检查、错误反馈和重新检查的循环 |

前两项是一套不同于论文 v3 的规则选择算法，后面三项分别替代了论文的提示选择、最终验证和语法纠正步骤。它们不应作为论文已经实现的功能保留在结果说明中。

## 从哪里开始读代码

先沿着一条查询的处理过程阅读，再看规则如何学习和复用。阅读时对照论文 Algorithm 1 和 §4，重点检查每个函数收到什么、返回什么，以及返回值是否被下一步正确使用。

| 顺序 | 文件 / 入口 | 阅读重点 | 论文对应 |
| --- | --- | --- | --- |
| 1 | [pipeline.py](pipeline_module/pipeline.py)，`GenRewrite.run` | 建议、纠正、评估、规则更新的调用关系；查询如何移出待处理集合；循环何时停止 | Algorithm 1 |
| 2 | [suggest_candidate_rewrite.py](suggest_module/suggest_candidate_rewrite.py)，`suggest_and_explain` | SQL 和 hints 如何组成 prompt；候选 SQL、规则和解释如何解析并返回 | §4.1 |
| 3 | [correct_candidate_rewrite.py](correct_module/correct_candidate_rewrite.py)，`correct_query` | 反例如何指导下一轮修正；语义纠正与语法纠正分别依据什么反馈 | §4.3 |
| 4 | [evaluate_rewrite.py](evaluate_module/evaluate_rewrite.py) | 等价判断、数据库执行、计时与加速比；区分 LLM 判断和独立验证 | §4.4 |
| 5 | [suggest_group_rewrite.py](suggest_module/suggest_group_rewrite.py) | 规则归组、新组创建、规则与查询关联的保存 | §4.1 |
| 6 | [suggest_select_rewrite.py](suggest_module/suggest_select_rewrite.py) | 检索对象、相似度与规则选择；结合论文检查当前 SQL 检索和组评分的差异 | §4.2 |
| 按需 | [gpt.py](pipeline_module/gpt.py)、[env.py](src/genrewrite/env.py)、[run.py](run.py) | 请求与历史对话封装、配置读取、示例启动参数；run.py 当前使用占位查询 | 各模块的调用基础 |

## 从哪里开始改

先修改候选生成模块和 pipeline 之间的接口。候选 SQL 当前没有正确返回，评估调用也存在参数错误；这些问题不解决，后续模块就无法接到正确输入。

| 批次 | 修改范围 | 为什么先做这一步 |
| --- | --- | --- |
| 1：数据传递与基本流程 | 分别返回候选 SQL、规则和解释；修正评估实参；分开数据库与仓库配置；处理失败值；补预算及稳定终止条件 | 后续纠正和评估都依赖正确的 SQL 输入。先用模拟调用确认数据传递和失败处理 |
| 2：纠正与验证 | 接入真实 EXPLAIN 错误反馈和 SQLSolver / tester；验证通过后测性能；修正倍数加速比 | 先确认候选可执行、符合正确性检查，再讨论它是否更快 |
| 3：规则学习 | 修复空仓库和新组创建；实现主导规则识别；保存收益与查询关联 | 后续规则选择需要有来源明确的收益记录 |
| 4：规则检索与复用 | 实现历史计划瓶颈总结和 LLM 二次确认；替换现有 SQL 相似度及组评分；将学习结果用于后续建议 | 检索依赖前面得到的有效规则与收益；完成后再运行论文实验 |

`calculate_score()` 不适合作为第一个修改点：当前评分需要被论文 v3 的检索方法替换，而新方法依赖历史计划和有效收益。先把候选、验证和收益记录做好，再修改检索模块。

## 改进计划

按下面的顺序补齐论文 §4 的方法，再进行 §5 的实验。各项均未完成。计划只包含论文实际功能及其必要代码修正；附录 B 的规则提炼属于讨论性扩展，不列入待办。下文新增文件名是建议，实际实现时可按现有模块组织。

### 第一批修改

- [ ] 返回候选 SQL、规则和解释，修正 pipeline 的评估参数。
- [ ] 分开数据库配置和规则仓库路径。
- [ ] 用 EXPLAIN 的真实错误反馈完成语法纠正。
- [ ] 接入 SQLSolver 与 tester，验证通过后再测性能。
- [ ] 使用倍数加速比，补上预算和结果不变时的停止条件。

随后实现规则分组与主导规则收益归因，再补瓶颈总结、两阶段检索和规则复用。最后按论文配置运行工作负载、基线和消融实验。

### 文件和函数修改清单

| 文件 / 函数 | 论文依据 | 具体修改内容 | 检查结果 |
| --- | --- | --- | --- |
| `suggest_candidate_rewrite.py:suggest_and_explain` | §4.1、Figure 4 | 读取 `answer["Candidate rewrite"]`；分别保存规则文本与解释，不再把规则列表命名并返回为候选 SQL；修正第二条解释字段名 | 模拟响应含 SELECT 1 时，返回 SQL 确实为 SELECT 1；多规则解释不会缺字段 |
| `pipeline.py:suggest_and_explain`、`run` | Algorithm 1 第 5–7 行 | 接收候选 SQL、规则和解释；只将 SQL 传入纠正；修正 `evaluate_rewrite(query, query, corrected_query)` 为原 SQL 与修正 SQL 两个参数 | 三阶段传入值与论文一致，不再触发参数数量错误 |
| `pipeline.py:__init__`、`evaluate_rewrite` | Algorithm 1 的 D 与 R | 分别保存数据库配置路径和规则仓库路径，评估器使用数据库配置；规则读写使用规则仓库 | 不再用 reportory.json 读取 dbname 等数据库字段 |
| `correct_candidate_rewrite.py:perform_semantic_correction` | §4.3、Figure 6 | 保留分解语义、表形式反例、基于分析修正的历史对话；只有明确 Equivalent 才通过，未知或非法回答不得落入等价分支 | 不等价候选用反例修正并重新检查；无法完成纠正则返回失败 |
| 同文件 `perform_syntax_correction` | §4.3 | 执行候选的 EXPLAIN；有数据库错误时，将原 SQL、候选 SQL 和错误消息送给 LLM；对修复 SQL 再 EXPLAIN，而非一次 LLM 自检 | 错列名或别名产生真实反馈；循环至可解释或预算终止 |
| 同文件 `correct_query`；pipeline 失败处理 | §4.3、Algorithm 1 | 保持先语义后语法；纠正失败返回明确失败结果，主流程跳过该候选，不能传 None 进入评估 | 失败候选不执行性能测量、不写成功结果 |
| `evaluate_rewrite.py:check_if_equiv`；新增 `verification.py` | §4.4 正确性检查 | 用 SQLSolver 替代 LLM 最终等价判断；UNKNOWN 回退 SlabCity tester；保存工具来源和状态，验证未通过则弃用 | LLM 判断 Equivalent 但独立验证失败的候选被拒绝；工具不支持不默认为通过 |
| 新增 tester 适配；纠正模块反例输出 | §4.4 | 将能够转换的语义反例纳入测试输入，结合 tester 的输入生成和结果比较；记录测试证据 | 反例实际用于执行比较，而非只留在 LLM 对话；有限测试结果与形式证明分开描述 |
| `pipeline.py:evaluate_rewrite` | §4.4 两项检查 | 先检查独立正确性结果；通过后才进行成本预筛选和性能测量 | 不等价 / 验证未通过候选的性能执行次数为零 |
| `evaluate_rewrite.py:execute_query` | §4.1、§4.3、§4.4 | 分离估计计划获取与实际执行计时；结构化读取 EXPLAIN 计划，不对普通 SELECT 的 row[0] 搜索 cost= | 数值结果不再触发文本解析错误；普通 EXPLAIN 不被当作实际执行耗时 |
| 同文件成本预筛选 | §4.4 性能检查 | 在昂贵性能试运行前使用静态成本代理筛选；记录筛选条件，成本不能代替实测收益 | 被预筛选拒绝的候选不试运行；通过者仍需实际测量 |
| 同文件 `evalutate`；pipeline 阈值判断 | §4.4、Algorithm 1 第 10 行 | 倍数用原耗时 / 改写耗时；处理测量失败；先保证不变慢，再按用户最小加速倍数收集；未接受则保留原查询 | 10 秒变 5 秒记录 2x；10 秒变 12 秒被拒绝；使用严格大于阈值的算法边界 |
| `suggest_group_rewrite.py:load_json`、`add_rule_in_group`、`add_rule_to_json` | §4.1、Figure 5；Algorithm 1 仓库初始化 | 空仓库直接创建首组；Unseen rule 分配合法新组 ID，修复 int(None)；语义相同规则加入已有组；保留查询关联 | 空仓库可学习第一条规则；同义规则可正确归组 |
| 新增 `dominant_rule.py`；评估模块计划接口 | §4.1 主导规则识别 | 以实测更快的最终候选作为比较基准；每个独立规则单独作为 hint 生成候选；只获取 EXPLAIN 估计计划；先匹配基准计划的成本，否则用 bert-base-uncased 计划嵌入的 L2 距离选择 | 归因候选不额外实际执行；BERT 编码对象是计划，区别于现有规则文本编码 |
| 分组模块与主导规则模块 | §4.1 分组减少重复调用 | 对同一查询的语义同组规则复用归因结果，避免反复生成同组单规则候选 | 同义规则不会触发重复归因流程 |
| `pipeline.py:update_rules`；规则 JSON 写入 | §4.1 收集规则和收益 | 保存原 / 改写查询关联、规则、查询特征、实测收益及主导规则；收益归因给选中主导规则；无归因结果时不得虚构固定收益 | 每个收益可回溯到成功候选；重启后可用于检索 |
| 新增 `bottleneck_analysis.py` | §4.2 性能瓶颈分析 | 读取历史执行计划，保留 actual time / rows / loops 等实际统计；裁剪无关字段；将 SQL、裁剪计划和论文指令送给 LLM 生成瓶颈总结 | 估计计划不能代替实际执行历史；摘要有明确计划来源 |
| `suggest_select_rewrite.py:embed_query`、`get_top_k_similar_queries` | §4.2 第一阶段相似度检索 | 编码并检索历史瓶颈总结，替代原始 SQL 相似度；按查询关联读取完整历史，不局限 query_1，避免规则重复导致查询重复入选 | top-k 对象是瓶颈总结；同查询多规则不重复占据近邻名额 |
| 同文件新增 `refine_match` | §4.2 第二阶段 LLM 确认 | 将近邻瓶颈交给 LLM 判断最接近者，提供无匹配选项 | 文本相似但瓶颈不同可拒绝；无匹配返回空 hints |
| 同文件 `calculate_score`、`select_best_nlr2` | §4.2 选择有收益规则 | 替换当前倒余弦和 benefit=1 组评分；按 LLM 选中查询的已记录收益选规则；去掉对 top-k 外查询的 index 查找 | 选择逻辑与论文 v3 一致；组跨查询时不报 ValueError |
| 两个 embedding 模块的模型加载 | §4.1、§4.2 的特征编码 | 正确定位仓库内模型；使用推理模式；瓶颈总结和计划分别编码，不再沿用固定 SQL 512 token 截断作为 v3 检索策略 | 模型能加载，特征对象正确，同输入可重复检索 |
| `pipeline.py:run`、`suggest_select_rewrite`、`suggest_group_rewrite` | §4、Algorithm 1 | 每次建议从当前仓库选 hints；无合适规则用基本 prompt；有效候选更新仓库并用于后续查询，替代所有查询共用固定 hints | 第二条同瓶颈查询能够使用前一条新学规则 |
| `pipeline.py:run`、`budget` | Algorithm 1 第 12–14 行 | 实现结果集合不变与预算耗尽检查；移除已优化查询；明确 B 的本地计量方式，不新增多套预算功能 | 无收益工作负载有限结束；预算被实际检查和消耗 |

### 实验代码

完成方法实现后，按论文的实验设置检查效果。

| 代码位置 | 论文依据 | 需要补齐的内容 |
| --- | --- | --- |
| `run.py`；新增工作负载加载模块 | §5.1 | 替换占位 SQL，加载 TPC-DS 99 条、JOB，以及论文选择的 SQLStorm 慢查询集：TPC-DS 100 条、JOB 50 条；保存来源和规模 |
| 评估模块；新增实验运行模块 | §5.1 | 记录 PostgreSQL 14.17 与实验环境；执行统计准备；每查询四个候选，每条查询三次运行取均值；统一原查询与候选的计时口径 |
| `gpt.py`、`src/genrewrite/env.py` 的模型配置 | §5.1、§5.5 | 论文模式使用默认 o3-mini，并记录其他论文模型的比较配置；实际核对所选模型的请求参数；保存调用 token 和时间开销，按记录的价目估算成本 |
| 新增基线与消融运行模块 | §5.2–§5.4 | 按论文列出的基线及消融配置运行；尚未适配的基线明确标为未复现，保持与论文相同的消融设置 |
| 新增统计模块；`data/result.json` | §5 的结果表与开销分析 | 保存逐查询结果、验证状态、实测样本、加速倍数及调用开销；计算论文对应的阈值覆盖与几何平均指标，注明统计集合；null 模板不计实验结果 |
