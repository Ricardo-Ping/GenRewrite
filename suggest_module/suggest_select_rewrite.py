"""NLR2 选择 (Similarity Search): 对应论文 §4.2 的两阶段相似检索。

==============================================================================
论文背景
==============================================================================
NLR2 对某些查询能带来大幅加速 (如 Table 1), 但对不相关的查询可能反而降低
性能。关键在于: 找出与当前查询"性能瓶颈相似"的历史查询, 只把对它们有效的
NLR2 放进 prompt (避免挑战 C4 —— 过多/不相关的提示导致 LLM 出错)。

论文 §4.2 的流程:
  1. Query performance bottleneck analysis ——
     用 LLM 分析查询执行计划 (先剪掉无关字段, PostgreSQL 计划可因此减 43%
     token), 总结最关键的性能瓶颈。
  2. Similarity search (两阶段) ——
     (1) 将瓶颈摘要编码为特征向量, 从历史查询中检索 top-k 最相似者;
     (2) 用 LLM 多选题从候选中精选最匹配的查询 (思路同 Figure 5);
         若都不匹配, 则退回不带 NLR2 的基础 prompt。
  3. 依据选中的历史查询, 取出其收益最大的关联 NLR2 作为本次改写的提示。

本实现用 Longformer 嵌入 (config_file/longformer/, 由 prepare/download.py
下载) 直接对"查询文本"而非瓶颈摘要做相似检索, 并用相似度加权聚合出每个
NLR2 组的收益分数, 选出 top-k 规则 —— 是论文思路的一个工程近似。
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LongformerTokenizer, LongformerModel
from collections import defaultdict
import torch


class suggest_select_rewrite:
    """从 NLR2 仓库中为新查询挑选最相关的 k 条规则 (论文 §4.2 相似检索的近似实现)。"""

    def __init__(self, json_file, k=3):
        # 读取JSON文件并解析所有规则
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.rules_by_group = defaultdict(list)  # key -> group_id, value -> list of rules
        self.rules_by_query = defaultdict(list)  # key -> query, value -> list of rules
        self.k = k                               # 召回的相似查询数 / 最终选出的规则数
        self.rules = []                          # 所有 NLR2 文本
        self.path = json_file
        self.hash = {}                           # NLR2 文本 -> group_id 映射
        self.query = []                          # 所有来源查询文本
        # 本地加载 Longformer 模型和分词器 (config_file/longformer/)
        self.tokenizer = LongformerTokenizer.from_pretrained('../config_file/longformer/')
        self.model = LongformerModel.from_pretrained('../config_file/longformer/')
        self.load_json()
        # 初始化二维哈希列表

    def load_json(self):
        """解析仓库: 建立 group→rules、query→rules 两套索引及规则→组映射。"""
        with open(self.path, 'r') as file:
            json_data = json.load(file)

        for key, value in json_data.items():
            if key.startswith("rules_"):
                for rule in value:
                    rewrite_rule = rule.get("rewrite_rule")
                    query = rule.get("query_list", {}).get("query_1", {}).get("query")
                    group_id = rule.get("group_id")
                    if query:
                        self.query.append(query)
                        self.rules_by_query[query].append(rule)
                        if rewrite_rule:
                            self.rules.append(rewrite_rule)
                            if group_id:
                                self.hash[rewrite_rule] = group_id  # 将rewrite_rule与group_id关联
                                self.rules_by_group[group_id].append(rule)  # 关联group_id与rule

    def embed_query(self, query):
        """将 SQL 查询文本编码为 Longformer 句向量 (token 隐状态取平均)。"""
        # 将查询转为Longformer模型的输入
        inputs = self.tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def get_top_k_similar_queries(self, input_query):
        """第一阶段检索: 余弦相似度找出与输入查询最相似的 top-k 历史查询。

        对应论文 §4.2 相似检索的 (1): 用嵌入向量衡量查询之间的相似性,
        返回 (候选查询列表, 对应相似度分数)。
        """
        input_embedding = self.embed_query(input_query)

        # 计算所有查询的嵌入并找出top-k相似查询
        # queries = [rule['query_list']['query_1']['query'] for rule in self.rules]
        queries = self.query
        embeddings = np.array([self.embed_query(query) for query in queries])

        # 将三维数组降为二维
        input_embedding = input_embedding.reshape(1, -1)
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

        similarities = cosine_similarity(input_embedding, embeddings)[0]
        top_k_indices = np.argsort(similarities)[-self.k:]

        return [self.query[i] for i in top_k_indices], similarities[top_k_indices]

    def calculate_score(self, top_k_queries, similarities):
        """给每个 NLR2 组打分: 相似度倒数加权 × (该组规则是否出现在候选查询中)。

        直觉: 与新查询越相似的历史查询, 其验证有效的规则越可能适用;
        指示函数 (查询规则集 ∩ 组规则集) 保证只给"确实被该查询用过"的组记分。
        注意: 代码把余弦相似度当作 distance 用 (1/similarity 加权), 因此
        相似度越高 → 权重越大 (pipeline 中对结果 [::-1] 倒序即按分数升序)。
        """
        # 计算每个group的分数
        group_scores = {}
        total_weight = sum([1 / distance for distance in similarities])

        for i, query in enumerate(top_k_queries):
            weight = (1 / similarities[i]) / total_weight
            rules_for_query = self.rules_by_query[query]
            # 搞出query对应的所有rewrite_rule
            query_rule_set = set([rule['rewrite_rule'] for rule in rules_for_query])

            for group_id, group_rules in self.rules_by_group.items():
                # 搞出group对应的所有rewrite_rule
                group_rule_set = set([rule['rewrite_rule'] for rule in group_rules])
                indicator = 1 if query_rule_set & group_rule_set else 0  # 使用集合交集判断

                benefit = 1  # 先设置benefit默认值，值为1
                # 鲁棒性判断，其实没必要
                if group_id not in group_scores:
                    group_scores[group_id] = 0
                group_scores[group_id] += weight * indicator * benefit

        return group_scores

    def select_best_nlr2(self, input_query):
        """主入口: 检索相似查询 → 组打分 → 每组选一条与候选查询最相似的规则。

        Returns:
            选出的 k 条 NLR2 文本列表 (调用方用 [::-1] 调整排序方向)。
        """
        # 获取最相似的top-k查询
        top_k_queries, similarities = self.get_top_k_similar_queries(input_query)

        # 计算每个group的分数
        group_scores = self.calculate_score(top_k_queries, similarities)

        # 按分数排序并选出top-k个group中的最佳NLR2
        sorted_groups = sorted(group_scores.items(), key=lambda item: item[1], reverse=True)[:self.k]
        selected_nlr2s = []
        for group_id, score in sorted_groups:
            # 组内规则可能有多条, 取其来源查询与输入查询最相似的那条作为代表
            best_nlr2 = max(
                self.rules_by_group[group_id],
                key=lambda x: similarities[top_k_queries.index(x['query_list']['query_1']['query'])]
            )
            selected_nlr2s.append(best_nlr2['rewrite_rule'])

        return selected_nlr2s
    
# 使用示例：
# selector = NLR2Selector('../data/reportory.json')
# 注意列表倒序
# best_nlr2s = selector.select_best_nlr2("IMPORVE SELECT * FROM table3 WHERE condition3")[::-1]
# print(best_nlr2s)



# selector = NLR2Selector('../data/reportory.json')

# print(selector.rules)
# print(selector.query)
# print(selector.hash)

# # 获取最相似的top-k查询
# best_queries, similarities = selector.get_top_k_similar_queries("IMPORVE SELECT * FROM table3 WHERE condition3")

# # 打印结果
# print("Best Queries:", best_queries)
# print("Similarities:", similarities)

# group_scores = selector.calculate_score(best_queries, similarities)
# print(group_scores)

# best_nlr2s = selector.select_best_nlr2("example input query")
# print(best_nlr2s)