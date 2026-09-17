"""改写评估 (Rewrite Evaluation): 对应论文 §4.4 "Evaluating Candidate Rewrites"。

==============================================================================
论文背景
==============================================================================
目标: 找到"等价且更快"的改写, 且绝不引入性能回退。论文设计了两道自动化门禁:

  正确性门 (correctness gate):
    用现成的 SQL 等价性验证器/测试器判定改写是否与原查询语义等价 ——
    首选 SQLSolver (验证器), 返回 UNKNOWN 时回退到 SlabCity 的 tester
    (从原查询提取执行提示、生成输入做结果比对); 语义纠错阶段发现的反例
    也可喂给 tester 提升覆盖。两者都无法证实时则放弃该改写 (abstain)。
    本实现用 LLM 的等价性判断 (check_if_equiv) 近似替代该验证器。

  性能门 (performance gate):
    只有通过正确性门的改写才做性能评估: 对目标数据库做离线影子执行,
    满足 "never-worse-off" 策略才采纳, 否则自动回退到原查询; 并可用
    静态代价估计预筛, 避免昂贵的实跑。
    本实现连接 PostgreSQL (config_file/postgres.json), 比较两条查询的
    执行延迟与 EXPLAIN 代价, 计算加速比 speedup。

==============================================================================
注意 (与论文的差距): execute_query() 执行的是查询原文而非 EXPLAIN 命令,
即实际跑完查询并计时 (顺便从 EXPLAIN 风格的输出行里解析 cost)。对大查询
这意味着真实的执行开销 —— 与论文 "用 EXPLAIN/代价估计避免昂贵实跑" 的
预筛思路不同, 使用时需留意 (尤其对耗时数十分钟的重查询)。
"""

# evaluate the optimized SQL queries and provide feedback to the user.
# quivalence -> slabcity
# perfermence -> EXPLAIN
import sys
import os
import re
import textwrap
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import json
import time
import textwrap
import psycopg2
from tabulate import tabulate
import csv
from pipeline_module.gpt import GPT


class Evaluate_rewrite_model:
    """评估器: 等价性检查 (正确性门) + 实测加速比 (性能门)。"""

    def __init__(self, db_file):
        self.gpt = GPT()
        # 数据库连接配置文件路径 (postgres.json: dbname/user/password/host/port)。
        # 注意: pipeline 中传入的是 NLR2 仓库路径 reportory.json —— 与本模块
        # 期待的数据库配置不一致, 属于现有代码的接缝问题。
        self.db_file = db_file

    # 连接数据库并返回连接信息
    def connect_to_database(self):
        """读取 JSON 配置并建立 PostgreSQL 连接 (psycopg2)。"""
        with open(self.db_file, 'r') as file:
            data = json.load(file)
        # 设置连接参数
        conn_params = {
            'dbname': data['dbname'],
            'user': data['user'],
            'password': data['password'],
            'host': data['host'],
            'port': data['port']
        }
        # 在使用 psycopg2.connect(**conn_params) 时，
        # **conn_params 是一种 参数解包 的方式。它将字典中的键值对解包为函数参数，因此相当于如下调用：
        # conn = psycopg2.connect(dbname=data['dbname'], user=data['user'], password=data['password'], host=data['host'], port=data['port'])
        # print(conn_params)
        try:
            conn = psycopg2.connect(**conn_params)
            print("Database connection successful")
            return conn
        except psycopg2.Error as e:
            print(f"Database connection failed: {e}")
            return None

    # 执行 SQL 查询并返回执行时间
    def execute_query(self, conn, cursor, query):
        """执行查询, 返回 (EXPLAIN 代价, 墙钟延迟)。

        - query_latency: 真实执行耗时 (秒) —— 性能门的主要依据;
        - execution_cost: 若输出行含 "cost=..." 则解析其上限值 (来自
          EXPLAIN 风格输出), 否则为 None。
        """
        try:
            start_time = time.time()  # 记录开始时间
            explain_query = f"{query}"
            cursor.execute(explain_query)
            result = cursor.fetchall()
            end_time = time.time()  # 记录结束时间
            execution_cost = None
            query_latency = end_time - start_time

            # 逐行扫描结果, 解析 PostgreSQL 输出中的 "cost=下限..上限" 字段
            for row in result:
                line = row[0]
                if "cost=" in line:
                    cost_part = line.split("cost=")[1].split()[0]
                    execution_cost = float(cost_part.split('..')[1].strip(')'))

            return execution_cost, query_latency

        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            return None, None

    def evalutate(self, input_query, rewrite_query):
        """性能门: 影子执行原查询与改写, 计算相对加速比。

        speedup = (原查询延迟 - 改写延迟) / 原查询延迟
                ∈ [-∞, 1]   (>θ 才采纳; 为负即性能回退, 违反 never-worse-off)

        论文中此步骤为 "离线影子执行 + never-worse-off 策略", 对应
        Algorithm 1 第 7 行返回的 speedup, 供第 10 行与阈值 θ 比较。
        """
        conn = self.connect_to_database()
        if conn is None:
            return

        # 创建游标对象
        cursor = conn.cursor()

        original_execution_cost, original_query_latency = self.execute_query(conn, cursor, input_query)
        if original_execution_cost is not None and original_query_latency is not None:
            print(f"Original Query Execution Cost: {original_execution_cost:.2f}")
            print(f"Original Query Latency: {original_query_latency:.2f} seconds\n")
        else:
            print("Failed to retrieve execution original cost or query latency.")

        rewrite_excution_cost, rewrite_query_latency = self.execute_query(conn, cursor, rewrite_query)
        if rewrite_excution_cost is not None and rewrite_query_latency is not None:
            print(f"Rewrite Query Execution Cost: {rewrite_excution_cost:.2f}")
            print(f"Rewrite Query Latency: {rewrite_query_latency:.2f} seconds\n")
        else:
            print("Failed to retrieve execution rewrite cost or query latency.")

        # 加速比: 延迟降低的相对比例 (而非论文中的倍数 speedup, 二者可换算)
        speed_up = (original_query_latency - rewrite_query_latency) / original_query_latency
        cursor.close()
        conn.close()

        return speed_up

    def check_if_equiv(self, origin_query, rewrite_query):
        """正确性门 (LLM 近似): 让 LLM 判断 q1 与 q2 是否语义等价。

        论文中该门由验证器/测试器 (SQLSolver / SlabCity tester) 把关;
        本实现以 LLM 推理近似替代, 结论的可靠性依赖模型能力, 必要时仍需
        人工复核 (论文也保留了最终的人工验证环节)。
        """
        prompt = textwrap.dedent(f"""
            <description>
            q1:{origin_query} q2:{rewrite_query}
            q1 is the original query, q2 is the rewritten query of q1.

            <target>
            To give a conclusion that whether q1 and q2 are equivalent or not.
            If equivalent, please return "Equivalent" and give a brief reason.
            If not equivalent, please return "Not Equivalent" and give a brief reason.
            <demant>
            JSON RESULT TEMPLATE:
            {{
                "Equivalence": , // answer 'Equivalent' or 'Not Equivalent'
                "Briefly analysis": ,      //briefly show wheather the two queries are equivalent or not
            }}

        """)
        response = self.gpt.get_GPT_response(prompt, json_format=True)
        if response["Equivalence"] == "Equivalent":
            return True
        else:
            return False


# example usage
# input_query = "EXPLAIN (ANALYZE, COSTS) select 100.00 * sum(case when p_type like 'PROMO%' then l_extendedprice * (1 - l_discount) else 0 end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue from lineitem, part where l_partkey = p_partkey and l_shipdate >= date '1995-09-01' and l_shipdate < date '1995-09-01' + interval '1month';"
# rewrite_query_1 = "EXPLAIN (ANALYZE, COSTS) SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem JOIN part ON l_partkey = p_partkey WHERE l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-09-01' + INTERVAL '1' MONTH;"
# rewrite_query_2 = "EXPLAIN (ANALYZE, COSTS) WITH promo_revenue AS ( SELECT SUM(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) AS promo_sum, SUM(l_extendedprice * (1 - l_discount)) AS total_sum FROM lineitem JOIN part ON l_partkey = p_partkey WHERE l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-09-01' + INTERVAL '1' MONTH) SELECT 100.00 * promo_sum / total_sum AS promo_revenue FROM promo_revenue;"

# test = Evaluate_rewrite_model("./postgres.json")
# test.evalutate(input_query, rewrite_query_1)
# speed_up = test.evalutate(input_query, rewrite_query_1)
# print(f"Speedup: {speed_up:.2f}")