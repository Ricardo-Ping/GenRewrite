"""NLR2 分组 (NLR2 grouping via LLMs): 对应论文 §4.1。

==============================================================================
论文背景 (Observation 2)
==============================================================================
两条 NLR2 可能文字不同但语义等价。论文 Table 2 给出的例子 —— 下面四条规则
针对的都是"相关子查询聚合"这一性能问题:
    r_a: Replace the correlated subquery with a precomputed aggregate in a CTE
    r_b: Use common table expressions (CTEs) to precompute aggregates
    r_c: Replace a correlated subquery ... with an inline view ...
    r_d: Replace a correlated subquery with a CTE for aggregated values
若不能识别这种等价性, 仓库会充满冗余规则, 导致:
  (1) 对同一规则重复调用 LLM / 重复评估, 浪费 API 开销 (挑战 C3);
  (2) 给 LLM 的提示重复, 增加混淆 (挑战 C4)。

因此每当发现新 NLR2 r_new 时, 判断它是否与已有规则语义等价:
  - 等价 → 归入已有组, 不再对它单独跑 LLM 改写流水线;
  - 不等价 → 新建组, 并进入后续 "主导规则识别" 流程 (plan-based
    dominant NLR2 identification, 论文用 bert-base-uncased 对 EXPLAIN
    计划做嵌入、以 ℓ2 距离衡量计划相似度 —— 对应 config_file/bert-base-uncased/)。

实现思路: BERT 嵌入 + KNN 召回最相似的候选规则, 再让 LLM 做多选题判断
(prompt 模板对应论文 Figure 5), 最后写回 reportory.json。
"""

import sys
import os

# 将 pipeline_module 路径添加到系统路径中
module_path = os.path.abspath(os.path.join('..'))  # 根据实际路径调整
if module_path not in sys.path:
    sys.path.append(module_path)
import textwrap
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from collections import defaultdict
from pipeline_module.gpt import GPT
import json

# BERT Embedding Model
class sugget_group_rewrite:
    """用 BERT 嵌入 + LLM 多选判断, 将新 NLR2 归入语义等价组 (论文 §4.1)。"""

    def __init__(self, path, k=3):
        self.gpt = GPT()
        # 本地加载 bert-base-uncased (config_file/bert-base-uncased/, 由
        # prepare/download_bert.py 下载), 避免运行时联网拉模型。
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.load_rewrite_rules = []  # 从仓库读出的所有 NLR2 文本
        self.hash = {}  # NLR2 文本 -> group_id 的映射 (同组即语义等价)
        self.path = path  # NLR2 仓库 (reportory.json) 路径
        self.embeddings = None  # 所有规则的 BERT 嵌入矩阵 (load 后填充)
        self.k = k  # KNN 召回的候选数 (喂给 LLM 多选题的选项个数上限)

    def embed(self, text):
        """将 NLR2 文本编码为 BERT 句向量 (所有 token 末层隐状态取平均)。"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def load_json(self):
        """读取仓库: 收集所有规则文本与 (规则 -> 组) 映射, 并预计算嵌入。"""
        with open(self.path, 'r') as file:
            json_data = json.load(file)

        for key, value in json_data.items():
            if key.startswith("rules_"):
                for rule in value:
                    rewrite_rule = rule.get("rewrite_rule")
                    group_id = rule.get("group_id")
                    if rewrite_rule:
                        self.load_rewrite_rules.append(rewrite_rule)
                        if group_id:
                            self.hash[rewrite_rule] = group_id  # 将rewrite_rule与group_id关联

        # 在加载 rewrite rules 后预计算它们的嵌入向量 (KNN 检索时直接复用)
        self.embeddings = np.vstack([self.embed(rule) for rule in self.load_rewrite_rules])
        # return self.embeddings

    def knn(self, input_sentence):
        """KNN 召回与输入规则最相似的 k 条候选, 且保证来自不同 group。

        先按欧氏距离升序排序, 再沿排序去重 group —— 这样喂给 LLM 多选题的
        选项互不等价, 提高判断的信息量 (论文 Figure 5 的 Options 列表)。
        """
        # 对输入句子进行编码
        input_embedding = self.embed(input_sentence)

        # 计算欧氏距离
        distances = euclidean_distances(input_embedding, self.embeddings)[0]

        # 获取距离最近的句子的索引，按照距离从小到大排序
        sorted_indices = np.argsort(distances)

        top_k_sentences = []
        seen_groups = set()

        # 遍历排序后的索引，选择属于不同group_id的句子
        for index in sorted_indices:
            rule = self.load_rewrite_rules[index]
            group_id = self.hash.get(rule)

            if group_id not in seen_groups:
                top_k_sentences.append(rule)
                seen_groups.add(group_id)

            # 当找到的句子数量达到k时停止
            if len(top_k_sentences) >= self.k:
                break

        return top_k_sentences

    def predict_group(self, input_query, candidates):
        """让 LLM 在候选中挑选与输入规则"严格相同"的一条 (论文 Figure 5)。

        Args:
            input_query: 这里实际传入的是新发现的 NLR2 文本 (the incoming NLR2)。
            candidates: knn() 召回的候选规则列表。
        Returns:
            LLM 的 JSON 回答, 含 "option" (选中的规则或 "Unseen rule") 与解释。
            选 "Unseen rule" 表示新规则与已有规则都不同 → 需要新建组。
        """
        # 选项 1 固定为 "Unseen rule"; 候选规则从选项 2 开始编号
        options = "1. Unseen rule\n"
        for i, candidate in enumerate(candidates, start=2):
            options += f"{i}. {candidate}\n"

        prompt = textwrap.dedent(f"""
            <description>
            {input_query}
            Please select the rewrite rule that is strictly the same as the above rule and give your explanation (just give one answer).
            If not, please select the first item “Unseen rule”.
            <Options>
            {options}

            <demand>
            JSON RESULT TEMPLATE:
            {{
                "option": , // the selected option(use the content of option, do not use the index
                "Explanation": ,      // give the Explanation of the selected option
            }}
            """

        )
        response = self.gpt.get_GPT_response(prompt, json_format=True)
        return response

    # 用于添加新的规则到JSON文件中
    def add_rule_to_json(self, group_id, rewrite_rule, query):
        """将 (组ID, NLR2, 来源查询) 持久化到 reportory.json。

        仓库结构: group_info 记录规则总数 rule_number 与组总数 group_number;
        每条规则存为 rules_{n}, 含 rule_id / group_id / rewrite_rule / query_list。
        query_list 记录该规则由哪条查询发现 —— 这是 §4.2 中 "按查询相似度
        选择 NLR2" (suggest_select_rewrite) 的数据基础。
        """
        # 读取 JSON 文件
        with open(self.path, 'r') as file:
            json_data = json.load(file)

        # 检查并初始化 rule_number
        rule_number = json_data['group_info'].get('rule_number')
        if rule_number is None:
            rule_number = 0

        # 检查并初始化 group_number
        group_number = json_data['group_info'].get('group_number')
        if group_number is None:
            group_number = 0

        # 更新group_info规则信息
        rule_number += 1
        json_data['group_info']['rule_number'] = rule_number

        json_data[f'rules_{rule_number}'] = []

        # 创建新的规则条目
        new_rule = {
            "rule_id": rule_number,
            "group_id": group_id,
            "rewrite_rule": rewrite_rule,
            "query_list": {
                "query_number": 1,
                "query_1": {
                    "id": 1,
                    "query": query
                }
            }
        }
        # 如果需要添加新组 (新规则的 group_id 超过当前最大组号)
        if int(group_id) > int(group_number):
            # 更新group_info的信息
            group_number += 1
            json_data['group_info']['group_number'] = group_number

        # 将新规则添加到 'rules' 列表中
        json_data[f'rules_{rule_number}'].append(new_rule)

        # 写回 JSON 文件
        with open(self.path, 'w') as file:
            json.dump(json_data, file, indent=3)

        print(f"Added rule with group_id: {group_id}: {rewrite_rule} to {self.path}")

    def add_rule_in_group(self, rewrite_rule, query):
        """新 NLR2 入库主流程: 召回 → LLM 判组 → 持久化。

        对应论文 §4.1 "NLR2 grouping via LLMs": 每当发现新规则 r_new,
        先用 KNN 找最相似的候选, 再让 LLM 判断它是否与某候选严格等价;
        等价则复用该候选的 group_id (避免冗余), 否则归入新组。
        """
        self.load_json()
        # 第 1 步: BERT 嵌入 KNN, 召回不同组的最相似候选规则
        top_k_sentences = self.knn(rewrite_rule)
        # 第 2 步: LLM 多选判断 —— 选出严格等价的规则, 或返回 "Unseen rule"
        response = self.predict_group(rewrite_rule, top_k_sentences)['option']
        # 第 3 步: 沿用选中规则的组号 (find_or_create group)
        group_id = self.hash.get(response)
        self.add_rule_to_json(group_id, rewrite_rule, query)


# 示例使用
# reportory_path = "../data/reportory.json"
# embedding_model = sugget_group_rewrite(reportory_path,k = 3)
# embedding_model.add_rule_in_group("SELECT * FROM table1 WHERE condition1.","This is a test query.")


# reportory_path = "../data/reportory.json"
# embedding_model = sugget_group_rewrite(reportory_path,k = 3)
# embedding_model.load_json()

# input_sentence = "SELECT * FROM table1 WHERE condition1."
# top_k_sentences = embedding_model.knn(input_sentence)

# print("Top-K most similar sentences:")
# for sentence in top_k_sentences:
#     print(sentence)


# response = embedding_model.predict_group(input_sentence, top_k_sentences)['option']
# # print(response)
# # 打印hash字典
# # print("Rewrite Rule to Group ID mapping:")
# # for rule, group_id in embedding_model.hash.items():
# #     print(f"Rule: {rule}, Group ID: {group_id}")
    
# group_id = embedding_model.hash.get(response)
# print(group_id)

# embedding_model.add_rule_to_json(group_id,input_sentence,"This is a test query.")

