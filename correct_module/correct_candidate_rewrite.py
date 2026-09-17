"""改写纠错 (Rewrite Correction): 对应论文 §4.3 "Correcting Candidate Rewrites"。

==============================================================================
论文背景
==============================================================================
LLM 生成的改写无法保证正确 (挑战 C1)。错误分两类:
  1) 语义错误 (semantic error) —— 可执行但结果不等价。论文例子: 原查询求
     "第二高工资", 改写却求 "最高工资"。
  2) 语法错误 (syntax error) —— 根本无法执行, 如表名/列名/别名错误。

若简单地把错误改写丢弃、反复重新生成, 既低效又浪费算力。GenRewrite 的
思路是 "不仅要验证等价性, 还要把改写修正到等价为止" —— 反例引导的
两阶段迭代纠错 (counterexample-guided correction):

  语义纠错 (对应论文 Figure 6 的 prompt):
      每轮迭代中, LLM 先分别逐步拆解原查询 q1 与候选改写 q2 的意图;
      再尝试构造一个反例 (小表格数据) 展示两者结果可能不同;
      若判断 "Not Equivalent", 则基于该分析给出 q2 的修正版本, 进入下一轮;
      直到 LLM 认为两者等价为止。
      (语义纠错的反馈来自 LLM 自身的逻辑推理)

  语法纠错:
      论文中以数据库 EXPLAIN 命令的报错信息为反馈, 让 LLM 修复表名/列名/
      别名等执行错误; 用 EXPLAIN (不真正执行) 可以极低开销地判断可执行性。
      本实现暂以一次 LLM 自检近似替代 EXPLAIN 反馈 (见 perform_syntax_correction)。

纠错收敛后, 候选改写才会进入 §4.4 的评估阶段 (等价性验证 + 性能门)。
"""

# #########################################
# This module is used to correct the rewritten query to ensure semantic equivalence with the original query.
# running checked successfully
# #########################################
# correct_module/__init__.py
import sys
import os
import re
import textwrap
import json
# Obtain the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Obtain the parent directory
parent_dir = os.path.dirname(current_dir)

# Make the sys path into the parent directory
sys.path.append(parent_dir)

from pipeline_module.gpt import GPT


class Nlr2Correction:
    """反例引导的两阶段纠错器: 语义纠错 (迭代) + 语法纠错 (§4.3)。"""

    def __init__(self, q1, q2):
        # q1: 原始查询 (gold standard, 论文中的 q)
        self.q1 = q1
        # q2: 候选改写 (论文中的 q~, 会被迭代修正为 q')
        self.q2 = q2
        self.gpt = GPT()

    def perform_semantic_correction(self):
        """语义纠错的一轮迭代 (论文 Figure 6)。

        Returns:
            True  —— LLM 判断 q1 与 q2 等价, 纠错完成;
            False —— 不等价, q2 已被替换为 LLM 给出的修正版本,
                     由 correct_query() 决定是否继续下一轮。
        """
        # 第一轮 prompt: 让 LLM 逐步拆解两条查询语义, 并尝试构造反例
        prompt = textwrap.dedent(
            f"""
            <description>
            q1:{self.q1} q2:{self.q2}
            q1 is the original query, q2 is the rewritten query of q1.

            <target>
            For q1, break it down step by step and then describe what it does in one sentence. Do the same for q2.
            Give an example, using tables, to show that these two queries are not equivalent if there's any such case. Otherwise, just say they are equivalent.

            <demant>
            JSON RESULT TEMPLATE:
            {{
                "Equivalence": , // answer Equivalent or Not Equivalent
                "Break down and analysis": ,      // show wheather the two queries are equivalent or not
                "Counterexample"        // if the two queries are not equivalent, show the counterexample, else show "null"
            }}
        """
        )

        analysis = self.gpt.get_GPT_response(prompt, json_format=True)
        # analysis_dict = json.loads(analysis)
        flag = analysis["Equivalence"]
        # print(flag)
        # eturn analysis
        #### print(analysis)
        # # Assume the GPT response includes both q1_analysis and q2_analysis

        if flag == "Not Equivalent":
            print("Not Equivalent")

            # 第二轮 prompt: 把上一轮的分析作为对话历史回填,
            # 让 LLM 基于自己找出的反例/分歧点, 给出 q2 的修正版本
            prompt_improve = textwrap.dedent(f"""
                <target>
                Based on your analysis, which part of q2 should be modified so that it becomes equivalent to q1? Show the modified version of q2.

                <demand>
                JSON RESULT TEMPLATE:
                {{
                "Analysis": , // step by step analysis
                "Modified version": ,      // show the modified version of q2
                }}
        """
            )
            cot_analysis = self.gpt.get_GPT_response_with_history(prompt, prompt_improve, analysis)
            # 用修正版本替换候选改写, 供下一轮迭代重新校验
            self.q2 = cot_analysis["Modified version"]
            print(
                f"CoT correction based on semantic correction: {self.q2}\n"
            )
            return False  # Not equivalent yet
        else:
            print(
                f"Equivalent"
                f"Origin query: {self.q2}\n"
            )

            return True  # Equivalent

    def perform_syntax_correction(self):
        """语法纠错: 修复导致查询无法执行的问题 (表名/列名/别名等)。

        论文原设计中反馈来自数据库 EXPLAIN 的报错信息 (开销极低, 因为不真正
        执行); 本实现简化为让 LLM 一次性自检语法错误 —— 若无错误则原样返回。
        """
        # Perform syntax correction based on feedback from EXPLAIN command using GPT
        prompt = textwrap.dedent(f"""
            <target>
            Perform syntax correction for the following SQL query:\n{self.q2}\n Is this query has any syntax error? If it hasn't, return the origin query. If it has, return the corrected query.

            <demand>
            JSON RESULT TEMPLATE:
            {{
                "Analysis": , // brief analysis
                "return query": ,      // return the non error query
            }}
        """
        )
        return self.gpt.get_GPT_response(prompt, json_format=True)

    def correct_query(self):
        """纠错主流程: 先语义纠错 (最多 4 轮), 再语法纠错。

        Returns:
            修正后的查询 q'; 若语义纠错在迭代上限内未收敛, 返回 None
            (该候选改写将被放弃, 不进入评估阶段)。
        """
        iterations = 0
        # Perform semantic correction until the queries are equivalent using LLM
        print(
            f"-----------------------------------------------------------------\n"
            f"running semantic correction............\n"
        )
        # 迭代语义纠错: 每轮 LLM 先分析/构造反例, 再基于分析修正 q2,
        # 直到判定等价。上限 4 轮, 防止模型反复修正陷入死循环 (无限消耗预算)。
        while not self.perform_semantic_correction():
            iterations += 1
            if iterations > 4:  # Avoid infinite loops
                return None

        print("running syntax correction............\n")
        correct_answer = self.perform_syntax_correction()
        self.q2 = correct_answer["return query"]
        print(
            f"CoT correction  based on syntax correction: {self.q2}"
            f"-----------------------------------------------------------------\n"
        )
        return self.q2


# Example debug:
# q1 = "SELECT COUNT(*) FROM 'contacts'INNER JOIN 'aspect_memberships' ON 'contacts'.'id' = 'aspect_memberships'.'contact_id' WHERE 'aspect_memberships'.'aspect_id' = 3;"
# q2 = "SELECT COUNT(*) FROM 'aspect_memberships' AS 'aspect_memberships' WHERE 'aspect_memberships'.'aspect_id' = 3;"
# q3 = "SELECT COUNT(*) FROM 'tag_followings' AS 'tag_followings' WHERE 'tag_followings'.'user_id' = 1"
# correction = Nlr2Correction(q1, q2)
# corrected_query = correction.correct_query()
# print("Corrected Query:", corrected_query)