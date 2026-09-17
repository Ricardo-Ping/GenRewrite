"""LLM 访问封装层: GenRewrite 流水线中所有大语言模型调用的唯一入口。

论文背景 (§2.2):
  - GenRewrite 通过 prompt engineering (提示工程) 驱动 LLM 完成改写任务;
    输入记作 p||q, 即提示 p 与查询 q 拼接后交给模型。
  - LLM 调用按输入/输出 token 计费, 因此论文强调控制提示长度、减少调用
    次数 (挑战 C3)。本类统一设置 temperature=0.0 以获得确定性输出,
    并通过 response_format={"type": "json_object"} 强制 JSON 返回,
    便于流水线稳定解析 NLR2、候选改写等结构化字段。
  - LLM 在流水线中承担 4 种角色:
      (1) §4.1 NLR2 分组 —— 判断新规则与已有规则是否语义等价;
      (2) §4.2 建议阶段  —— 生成候选改写并总结所用的 NLR2;
      (3) §4.3 语义纠错  —— 分析两条查询语义、构造反例、给出修正版本;
      (4) §4.4 等价检查  —— 近似代替验证器给出等价性结论。

API 配置 (base_url / api_key / model) 从项目根目录的 .env 读取,
详见 src/genrewrite/env.py。
"""

from openai import OpenAI
import re
import json
import tiktoken
import json

import configparser

from genrewrite.env import get_config

# # 创建 ConfigParser 对象
# config = configparser.ConfigParser()

# # 读取配置文件
# config.read('config.ini')

# # 读取 [parameters] 部分的变量
# budget = int(config['parameters']['budget'])
# min_speedup = float(config['parameters']['min_speedup'])

# # 读取 [gpt] 部分的变量
# api_base = config['gpt']['api_base']
# api_key = config['gpt']['api_key']


class GPT:
    """OpenAI 兼容 API 的薄封装, 提供普通文本与带历史两种调用方式。"""

    # init function to initialize the GPT class using gpt-4o, ignoring the token and money cost termerarily
    def __init__(self):
        # 从 .env / 环境变量读取 base_url、api_key、model
        config = get_config()
        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        self.model = config["model"]

    def get_GPT_response(self, prompt, json_format=False):
        """单轮调用: 发送 prompt, 返回文本或解析后的 JSON dict。

        Args:
            prompt: 拼接好的提示 (论文中的 p||q)。
            json_format: True 时强制模型输出 JSON 并解析为 dict;
                         流水线各阶段都依赖结构化字段 (如 "Candidate rewrite")。
        """
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        if (json_format == True):
            completion = client.chat.completions.create(
                temperature=0.0,  # 温度 0 → 贪心解码, 输出可复现, 利于调试与评估
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You should output JSON."},
                    {"role": "user", "content": prompt},
                ]
            )
            answer = json.loads(completion.choices[0].message.content)

        else:
            completion = client.chat.completions.create(
                temperature=0.0,
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            answer = completion.choices[0].message.content
        return answer

    def get_GPT_response_with_history(self, prompt1, prompt2, history):
        """多轮调用: 把上一轮回答作为 assistant 消息回填, 追问 prompt2。

        论文 §4.3 的语义纠错正是这种模式 (对应 Figure 6 的两个连续 prompt):
          第一轮 (prompt1): 让 LLM 拆解 q1/q2 语义并尝试构造反例, 得到分析 history;
          第二轮 (prompt2): 基于该分析, 要求 LLM 给出 q2 的修正版本。
        让模型先"看见"自己的分析再修改, 比从零重试更省 token (挑战 C3)。
        """
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        # history 既可能是 dict/list (来自上一轮 JSON 输出), 也可能是纯字符串
        # 如果 history 是字典或列表，直接转换为字符串
        # print(type(history))
        if isinstance(history, (dict, list)):
            history_content = json.dumps(history)
        else:
            try:
                history_content = history  #
            except json.JSONDecodeError:
                history_content = history

        # 将 history 字符串格式化为所需的字典 (回填为 assistant 角色)
        history_formatted = {"role": "assistant", "content": history_content}

        # print(history_formatted)
        # print(type(history_formatted))
        content = [
            {"role": "system", "content": "You should output JSON."},
            {"role": "user", "content": prompt1},
            history_formatted,          # 上一轮模型自己的回答, 提供推理上下文
            {"role": "user", "content": prompt2}
        ]
        completion = client.chat.completions.create(
            response_format={"type": "json_object"},
            temperature=0.0,
            model=self.model,
            messages=content
        )
        answer = json.loads(completion.choices[0].message.content)
        return answer
#Example usage

# gpt = GPT()
# prompt = "What is the capital of France?"

# response = gpt.get_GPT_response(prompt,json_format=False)
# # # response = gpt.get_chat_messages(prompt)
# print(f"{response}")


# prompt1 = "What is the capital of France?"
# prompt2 = "重复一遍我的上一个问题？"
# reanswer = gpt.get_GPT_response(prompt1,json_format=True)
# reanswer = gpt.get_GPT_response(prompt1)
# print(f"{reanswer}\n")
# answer = gpt.get_GPT_response_with_history(prompt1, prompt2, reanswer)
# print(f"{answer}")
# print(f"{response2}")