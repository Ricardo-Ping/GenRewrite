"""GenRewrite 顶层算法: 对应论文 §4 的 Algorithm 1。

==============================================================================
论文背景
==============================================================================
GenRewrite 是首个利用大语言模型 (LLM) 进行 SQL 查询改写的整体系统, 旨在解决:
  C1: 直接让 LLM "改写得更快等价" 常产生语法/语义错误;
  C2: LLM 没有数据库代价模型, 改写不一定更快;
  C3: 过多 LLM 调用成本高;
  C4: 提示中的提示项 (hints) 过多或不相关反而有害;
  C5: LLM 改写是黑盒, 人类难以理解和验证。

三大对策 (论文 Figure 1 的三个阶段):
  ① Suggest  (§4.2): LLM 参照从 NLR2 仓库选出的相关规则生成候选改写 + 解释;
  ② Correct  (§4.3): 反例引导的迭代纠错 —— 语义纠错靠 LLM 推理反例,
     语法纠错靠数据库 EXPLAIN 的报错反馈;
  ③ Evaluate (§4.4): 等价性验证 (验证器/测试器) + 性能门 (实际执行或代价估计),
     通过 "never-worse-off" 策略决定是否采纳改写。

贯穿全程的是 NLR2 (Natural Language Rewrite Rule) 仓库 R (§4.1):
用自然语言描述的改写规则, 同时充当 LLM 的提示和给用户看的解释, 实现
"改写一条查询获得的知识迁移到另一条查询", 让系统越用越聪明。

==============================================================================
Algorithm 1 (GenRewrite top-level) 与本实现的对应关系
==============================================================================
  输入: Q: queries, L: LLM, D: database, T: tester, B: budget, θ: min speedup
  输出: Res = {<q, q', e> | q ∈ Q_opt ⊆ Q}: 优化后的查询及对应 NLR2 解释

  1  Res ← {}
  2  NLR2 Repository R ← {} 或预置 R_pre          → __init__ 读取 reportory.json
  3  while Q is not empty do
  4      for each query q in Q do
  5          q~, e ← Suggest-and-explain(q, L, R)  → suggest_and_explain()
  6          q' ← Correct-for-equivalence(q, q~, L, D) → correct_for_equivalence()
  7          equiv, speedup ← Evaluate-rewrite(q, q', D, T) → evaluate_rewrite()
  8          if equiv is true then
  9              R ← Update-NLR2-repo(R, e, speedup)  → update_rules()
  10             if speedup > θ then
  11                 Res.add(<q, q', e>)
  12/13 if Res 不再变化 or 预算 B 耗尽: return Res
  14     从 Q 中移除已进入 Res 的查询
"""

import sys
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import random
from pipeline_module.gpt import GPT
from correct_module.correct_candidate_rewrite import Nlr2Correction
from evaluate_module.evaluate_rewrite import Evaluate_rewrite_model
from suggest_module.suggest_candidate_rewrite import suggest_candidate_rewrite
from suggest_module.suggest_group_rewrite import sugget_group_rewrite
from suggest_module.suggest_select_rewrite import suggest_select_rewrite


class GenRewrite:
    """GenRewrite 流水线编排器 (论文 Algorithm 1 的实现)。"""

    def __init__(self, queries, budget, min_speedup, reportory_path):
        # Q: 待优化查询集合 (Algorithm 1 输入)
        self.queries = queries
        # L: 大语言模型组件, 供 suggest / correct / evaluate 各阶段共用
        self.gpt = GPT()
        # self.tester = tester # what is tester ?? to be reconsidered.
        # B: 预算上限 —— 限制迭代次数/LLM 调用开销 (解决挑战 C3)
        self.budget = budget
        # θ: 用户指定的最小期望加速比, 只有 speedup > θ 的改写才进入 Res
        self.min_speedup = min_speedup
        # NLR2 仓库路径 (JSON 文件, 论文中记作 R)
        self.json_path = reportory_path
        # Res: 输出结果, 每项为 (原查询 q, 改写 q', 解释 e / NLR2 列表, 加速比)
        self.res = []
        # 运行过程中收集到的 (NLR2, 实测加速比) 记录
        self.rewrite_rules = []

    # ------------------------------------------------------------------
    # Algorithm 1 第 5 行: Suggest-and-explain(q, L, R)
    # ------------------------------------------------------------------
    # 论文 §4.2: 1) 从 NLR2 仓库中挑选与当前查询性能瓶颈最相关的 NLR2;
    # 2) 把 NLR2 作为提示拼进 prompt, 让 LLM 生成候选改写 q~;
    # 3) 同时让 LLM 用自然语言总结所用改写规则 (即新的 NLR2) 及其适用条件。
    # 实现在 suggest_module/suggest_candidate_rewrite.py。
    # ------------------------------------------------------------------
    def suggest_and_explain(self, query, hints):
        suggest = suggest_candidate_rewrite(self.gpt, query, hints)
        candidate_rewrite, explanation = suggest.suggest_and_explain(query, hints)
        return candidate_rewrite, explanation

    # ------------------------------------------------------------------
    # NLR2 分组 (论文 §4.1 "NLR2 grouping via LLMs", Observation 2)
    # ------------------------------------------------------------------
    # 两条 NLR2 可能描述不同但语义等价 (如 Table 2 中四条针对相关子查询的规则)。
    # 通过 "BERT 嵌入 KNN 召回 + LLM 多选判断" 将新规则归入已有组, 避免仓库冗余,
    # 也避免给 LLM 提供重复提示 (节省 API 开销, 解决挑战 C3/C4)。
    # 实现在 suggest_module/suggest_group_rewrite.py。
    # ------------------------------------------------------------------
    def suggest_group_rewrite(self, rewrite_rule, query, k=3):
        group_model = sugget_group_rewrite(self.json_path, k)
        group_model.add_rule_in_group(rewrite_rule, query)

    # ------------------------------------------------------------------
    # NLR2 选择 (论文 §4.2 "Similarity search" 两阶段检索)
    # ------------------------------------------------------------------
    # 面向新查询, 从仓库中召回最相关的 NLR2 作为提示:
    #   (1) 用 Longformer 嵌入对查询做 top-k 相似检索;
    #   (2) 按相似度加权聚合出每个 NLR2 组的收益分数, 选出最优的 k 条 NLR2。
    # 若找不到匹配项, 则退回不带 NLR2 的基础 prompt (避免挑战 C4 的提示污染)。
    # 实现在 suggest_module/suggest_select_rewrite.py。
    # ------------------------------------------------------------------
    def suggest_select_rewrite(self, input_query):
        selector = suggest_select_rewrite(self.json_path)
        best_nlr2s = selector.select_best_nlr2(input_query)[::-1]
        return best_nlr2s

    # ------------------------------------------------------------------
    # Algorithm 1 第 6 行: Correct-for-equivalence(q, q~, L, D)
    # ------------------------------------------------------------------
    # 论文 §4.3: LLM 生成的改写无法保证正确, 错误分两类:
    #   - 语义错误: 可执行但结果不等价 (如把"第二高工资"改成了"最高工资");
    #   - 语法错误: 无法执行 (如表名/列名/别名错误)。
    # 采用反例引导的两阶段纠错: 先让 LLM 自己举反例迭代修正语义, 直到它认为
    # 等价; 再基于 EXPLAIN 反馈修复语法。相比"反复重新生成", 显著降低成本 (C1/C3)。
    # 实现在 correct_module/correct_candidate_rewrite.py。
    # ------------------------------------------------------------------
    def correct_for_equivalence(self, original_query, rewritten_query):
        correction = Nlr2Correction(original_query, rewritten_query)
        corrected_query = correction.correct_query()
        return corrected_query

    # ------------------------------------------------------------------
    # Algorithm 1 第 7 行: Evaluate-rewrite(q, q', D, T)
    # ------------------------------------------------------------------
    # 论文 §4.4: 两道自动化门禁 ——
    #   正确性门: 用现成验证器/测试器 (SQLSolver / SlabCity tester) 判定等价性,
    #             本实现以 LLM 等价判断近似替代 (check_if_equiv);
    #   性能门:   在真实数据库上影子执行原查询与改写, 依据代价/延迟计算加速比,
    #             满足 "never-worse-off" 才采纳。
    # 实现在 evaluate_module/evaluate_rewrite.py。
    # ------------------------------------------------------------------
    def evaluate_rewrite(self, original_query, rewritten_query):
        evalute_model = Evaluate_rewrite_model(self.json_path)
        is_equiv = evalute_model.check_if_equiv(original_query, rewritten_query)
        speedup = evalute_model.evalutate(original_query, rewritten_query)
        return is_equiv, speedup

    # ------------------------------------------------------------------
    # Algorithm 1 第 9 行: Update-NLR2-repo(R, e, speedup)
    # ------------------------------------------------------------------
    # 若改写被采纳, 其解释 e (NLR2) 被视为有价值的知识: 记录该规则与实测加速比,
    # 后续可供其他查询检索复用 —— 这正是论文强调的 "知识迁移, 越用越聪明"。
    # ------------------------------------------------------------------
    def update_rules(self, explanation, speedup):
        self.rewrite_rules.append((explanation, speedup))

    # ------------------------------------------------------------------
    # Algorithm 1 主循环 (第 3-14 行)
    # ------------------------------------------------------------------
    def run(self, hints):
        """执行完整迭代流水线。

        Args:
            hints: 初始 NLR2 提示 (种子规则)。首轮迭代直接使用; 后续迭代中
                   可由 suggest_select_rewrite 从仓库自动挑选更相关的 NLR2。
        Returns:
            Res: 列表, 每项为 (原查询, 改写查询, NLR2 解释列表, 加速比),
                 仅包含等价且加速比超过 θ 的改写。
        """
        while self.queries:
            for query in self.queries:
                # 第 5 行: 建议阶段 —— 生成候选改写 q~ 和解释 e
                rewritten_query, explanation = self.suggest_and_explain(query, hints)
                # 第 6 行: 纠错阶段 —— 迭代修正 q~ 直至(近似)等价且可执行, 得到 q'
                corrected_query = self.correct_for_equivalence(query, rewritten_query)
                # 第 7 行: 评估阶段 —— 等价性 + 实测加速比
                # 注意: 这里传的是 (query, query, corrected_query), 第 2 个参数
                # 为占位调用, 实际起作用的是原查询和纠正后的改写。
                is_equiv, speedup = self.evaluate_rewrite(query, query, corrected_query)

                if is_equiv:
                    # 第 9 行: 将解释作为 NLR2 沉淀进规则库 (知识迁移)
                    self.update_rules(explanation, speedup)
                    if speedup > self.min_speedup:
                        # 第 10-11 行: 超过阈值 θ 才计入结果 Res
                        self.res.append((query, corrected_query, explanation, speedup))

            # 第 12-13 行: 终止条件 —— 结果集不再变化或预算 B 耗尽时返回。
            if not self.queries:  # len(self.res) >= self.budget:
                return self.res

            # 第 14 行: 从工作负载中移除已成功优化的查询。
            self.queries = [q for q in self.queries if q not in [r[0] for r in self.res]]

        return self.res

