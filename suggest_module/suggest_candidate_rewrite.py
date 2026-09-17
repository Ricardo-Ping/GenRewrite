"""建议阶段 (Rewrite Suggestion): 对应论文 §4.2 "Suggesting Candidate Rewrites"。

==============================================================================
论文背景
==============================================================================
论文 §3 Lesson 2 指出: 零样本 (zero-shot) 提示并非总能产生有效改写, 在 prompt
中加入改写提示 (hints) 能显著引导 LLM。相比 few-shot 例子, 用简短的自然语言
规则做提示有两个好处: (1) token 更少、更省钱 (挑战 C3); (2) 人类可读、易调试。

这些提示就是 NLR2 (Natural Language Rewrite Rule, 论文 §4.1), 例如:
  "Split aggregated computations by filtering on the specific period
   in separate subqueries"           → TPC-DS Q11 上实测 24.5x 加速 (Table 1 r1)

工作方式 (对应论文 Figure 4 的 prompt 模板):
  [输入查询 q]
  Rewrite this query to improve performance. Only use this rule when
  rewriting: [选出的 NLR2]

同时, 按 §4.1 "Collecting NLR2s" 一节, 还要让 LLM 在给出改写的同时
总结所用的改写规则 (prompt 中附加 "Describe the rewrite rules you are using"),
并显式要求 "不要包含具体表名/列名", 使 NLR2 保持通用、可复用、不含敏感信息。
这些新收集的 NLR2 经评估确认有效后会写入 NLR2 仓库 (reportory.json)。

本模块输出的 (候选改写 q~, 解释 e) 即 Algorithm 1 第 5 行的 q~ 与 e。
"""

import sys
import os
import re
import textwrap
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pipeline_module.gpt import GPT


class suggest_candidate_rewrite:
    """调用 LLM 生成候选改写 + 归纳 NLR2 (论文 §4.2)。"""

    def __init__(self, gpt, query_pool, hints_pool):
        self.gpt = GPT()          # LLM 封装 (与传入的 gpt 参数等价, 这里重新初始化)
        self.query_pool = query_pool   # 当前上下文中的查询 (预留字段)
        self.hints_pool = hints_pool   # 可用的 NLR2 提示池 (预留字段)

    def suggest_and_explain(self, query, hints):
        """生成候选改写并归纳所用 NLR2。

        Args:
            query: 原始 SQL 查询 q。
            hints: 提供给 LLM 的 NLR2 提示文本 (来自种子提示或仓库检索)。
        Returns:
            candidate_rewrite: 本实现中实际是 NLR2 规则文本列表 (键名
                "rewrite_rule_i", 见下方 JSON 模板); 真正的改写语句在
                "Candidate rewrite" 键中, 此处未提取 —— 期待返回的列表
                供纠错与解释环节使用。
            explanation: 每条 NLR2 对应的文字解释列表。
        """
        # prompt 三段式: <description> 任务与输入 | <target> 目标指令 + 提示 | <demand> 输出格式
        response = textwrap.dedent(f"""
            <description>
            Input:{query}

            <target>
            Rewrite this query to improve performance. Describe the rewrite rules you are using (you must not include any specific query details in the rules, e.g., table names, column names, etc). Be concise.
            Here are some hints that you might consider when rewriting the query:(And also you can use another rewirte rules based on your thought)
            {hints}

            <demand>
            JSON RESULT TEMPLATE:
            {{
                "Candidate rewrite": , // return the candidate rewrite based on rewrite rules the prompt provided or your thought
                "total_number": ,     // return the total number of NLR2s actually applied
                "rewrite_rule_1": ,      // return the rewrite rule 1. Only return the single rewrite rule, do not contain any other information.
                "rewrite_rule_1_explanation": ,      // return the rewrite rule 1 explanation
                "rewrite_rule_2": ,      // return the rewrite rule 2. Only return the single rewrite rule, do not contain any other information.
                "rewrite_rule 2_explanation": ,      // return the rewrite rule 2 explanation
                ........   //if any other rewrite rules applied, return them in the same format
            }}
        """
        )
        # json_format=True → 强制 JSON 输出, 便于逐条取出规则与解释
        answer = self.gpt.get_GPT_response(response, json_format=True)
        # total_number: 该改写实际应用的 NLR2 条数 (论文观察到每次改写通常对应 3-6 条 NLR2)
        nlr2_number = answer["total_number"]
        candidate_rewrite = []
        explanation = []
        for i in range(nlr2_number):
            # 逐条取出 LLM 声明使用的改写规则 (NLR2) 及其解释。
            # 按论文 §4.1, 一条改写会关联多条 NLR2, 后续由 "主导规则识别"
            # (plan-based dominant NLR2 identification) 确定真正的功臣。
            rewrite_rule = answer[f"rewrite_rule_{i + 1}"]
            rewrite_rule_explanation = answer[f"rewrite_rule_{i + 1}_explanation"]
            # print(f"rewrite rule: {rewrite_rule}")
            # print(f"rewrite rule explanation: {rewrite_rule_explanation}")
            candidate_rewrite.append(rewrite_rule)
            explanation.append(rewrite_rule_explanation)

        return candidate_rewrite, explanation


# query = "SELECT COUNT(DISTINCT 'contacts'.'id') FROM 'contacts' LEFT OUTER JOIN 'people' ON 'people'.'id' = 'contacts'.'person_id' LEFT OUTER JOIN 'profiles' ON 'profiles'.'person_id' = 'people'.'id' WHERE 'contacts'.'user_id' = 1945"

# hints = """
# Pre-calculate aggregates in subqueries
# Remove unnecessary UNION ALL operation
# Avoid using arithmetic operations in WHERE clause
# Replace implicit JOINs with explicit JOINs
# """

# gpt = GPT()
# suggest = suggest_candidate_rewrite(gpt, query, hints)
# candidate_rewrite, explanation = suggest.suggest_and_explain(query, hints)


# print(f"Candidate Rewrite:\n{candidate_rewrite}")
# print(f"Explanation:\n{explanation}")



