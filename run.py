"""GenRewrite 程序入口。

对应论文《GenRewrite: Query Rewriting via Large Language Models》
(Liu & Mozafari, arXiv:2403.09060) 的整体工作流 (论文 Figure 1):

    输入工作负载 Q (一组待优化的 SQL 查询)
        │
        ▼
    ① Rewrite Suggestion  —— suggest_module/   : LLM 参照 NLR2 提示生成候选改写及其解释 (§4.2)
    ② Rewrite Correction  —— correct_module/   : 反例引导的语义纠错 + 语法纠错 (§4.3)
    ③ Rewrite Evaluation  —— evaluate_module/  : 等价性与性能(加速比)评估 (§4.4)
    ④ NLR2 Repository     —— data/reportory.json: 自然语言改写规则的存储与知识迁移 (§4.1)

    算法整体流程见论文 Algorithm 1, 实现在 pipeline_module/pipeline.py 的
    GenRewrite 类中。产出 Res = {(q, q', e)}: 原查询 q、等价且更快的改写 q',
    以及人类可读的改写解释 e。

运行前需要:
    1. 在项目根目录 .env 中配置 LLM API (GENREWRITE_API_KEY 等, 见 src/genrewrite/env.py);
    2. config_file/postgres.json 中配置 PostgreSQL 连接 (用于 EXPLAIN 性能评估);
    3. data/reportory.json 中存放历史 NLR2 规则库 (可为空库, GenRewrite 会逐渐积累)。
"""

import time
import os
import sys
from pipeline_module.gpt import GPT
from pipeline_module.pipeline import GenRewrite
import logging


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 第 0 步: 初始化并测试 LLM 连接。
    # 对应论文 Algorithm 1 中的 L (LLM 组件) —— 所有 suggest / correct /
    # evaluate 阶段都依赖它。这里用一个简单问题做连通性冒烟测试。
    # ------------------------------------------------------------------
    print(
        f"-----------------------------------------------------------------\n"
        f"Start initializing configuration...\n"
    )
    try:
        gpt = GPT()
        print("GPT connection test ...\n  Prompt: What is the capital of France?")
        prompt = "What is the capital of France?"
        response = gpt.get_GPT_response(prompt)
        print(f"  {response}\n")
        print("GPT connection successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    print(
        f"-----------------------------------------------------------------\n"
    )

    # ------------------------------------------------------------------
    # 初始提示 (hints): 一组人工给出的"种子" NLR2 (Natural Language Rewrite Rules)。
    # 对应论文 §4.1: NLR2 是用自然语言描述的改写规则, 作为 LLM 的提示 (hint)。
    # 与传统模式匹配规则不同, NLR2 不受固定模式限制, 能表达上下文相关的改写洞察。
    # 随着系统运行, LLM 自己总结出的新 NLR2 会被存入 reportory.json (知识迁移)。
    # ------------------------------------------------------------------
    hints = """
    Pre-calculate aggregates in subqueries
    Remove unnecessary UNION ALL operation
    Avoid using arithmetic operations in WHERE clause
    Replace implicit JOINs with explicit JOINs
    """
    # NLR2 仓库路径: 论文中的 NLR2 Repository R, 既可为空也可预置 (R_pre)。
    json_path = "./data/reportory.json"
    # GenRewrite(queries, budget, min_speedup, reportory_path) 对应 Algorithm 1 的输入:
    #   Q: queries          —— 待优化查询集合 Q
    #   B: budget           —— 迭代/预算上限
    #   θ: min_speedup      —— 用户指定的最小期望加速比 (Algorithm 1 第 10 行 speedup > θ)
    #   reportory_path      —— NLR2 仓库 R (空库或预置 R_pre)
    candidate_queries = ["SELECT * FROM table WHERE column = value"]
    pipline = GenRewrite(queries = candidate_queries, budget = 10, min_speedup = 0.2, reportory_path = json_path)
    # 运行完整流水线: suggest → correct → evaluate → update NLR2 repo, 直至收敛或预算耗尽。
    res = pipline.run(hints)

    # 输出 Res = {<q, q', e>}: 每条记录包含原查询、等价且更快的改写、改写解释(即 NLR2)。
    print(
        f"-----------------------------------------------------------------\n"
        f"End of the program.\n"
        f"-----------------------------------------------------------------\n"
    )
    print(f"Result: {res}")
    