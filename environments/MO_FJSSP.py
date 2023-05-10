"""
多目标静态柔性作业车间调度环境:
目标：最大完工时间+总延期时间
"""
import random, math
import numpy as np
from SO_DFJSP import SO_DFJSP_Environment
# 环境类
class MO_FJSSP_Environment(SO_DFJSP_Environment):
    """单目标柔性作业车间调度环境"""
    environment_name = "single object FJSP"
    def __init__(self, DDT, M, S):
        SO_DFJSP_Environment.__init__(self, DDT, M, S)
        # 封装基本属性
        self.step_count = 0  # 决策步
        self.step_time = 0  # 时间点
        self.order_arrive_time = 0  # 顶顶那到达时间点
        self.last_observation_state = None  # 上一步观察到的状态 v(t-1)
        self.observation_state = None  # 当前时间步的状态 v(t)
        self.state_gap = None  # v(t) - v(t-1)
        self.state = None  # s(t)
        self.next_state = None  # 下一步状态  s(t+1)
        self.reward = None  # 即时奖励
        self.done = False  # 是否为终止状态
        # 动作和观察的状态空间维度
        self.action_space = [0, 1]  # 一维连续动作空间
        self.observation_space = 12  # 观察的状态向量空间
        self.reward_sum = 0  # 累计回报
        # 工序和机器选择规则相关属性
        self.kind_task_delay_e_list = []  # 估计延期工序类型列表
        self.kind_task_delay_a_list = []  # 实际延期工序类型列表
        self.kind_task_delay_time_a = {}  # 工序类型实际延期时间
        self.kind_task_delay_time_e = {}  # 工序类型估计延期时间
        self.kind_task_due_date = {}  # 工序类型的最小交期
        # 回报计算相关属性
        self.delay_time_sum_last = None  # 上一决策步的估计总延期时间
        self.delay_time_sum = None  # 剩余工件总的估计延期时间
        self.delay_time_sum_a = None  # 工件总的延迟时间
        # 输出订单到达时间和交期时间
        print("成功定义环境类")