"""
多目标静态柔性作业车间调度环境
目标：最大完工时间+总延期时间+机器平均利用率
动态因素：新订单到达+机器故障
"""
import random, math
import numpy as np
from MO_FJSSP import MO_FJSSP_Environment
# 环境类
class MO_DFJSP_Environment(MO_FJSSP_Environment):
    """单目标柔性作业车间调度环境"""
    environment_name = "single object FJSP"
    def __init__(self, DDT, M, S):
        MO_FJSSP_Environment.__init__(self, DDT, M, S)
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

