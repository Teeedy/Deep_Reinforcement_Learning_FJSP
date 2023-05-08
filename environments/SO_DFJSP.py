import copy
import random, math
import time
import numpy as np
from class_FJSP import FJSP

# 环境类
class SO_DFJSP_Environment(FJSP):
    """单目标柔性作业车间调度环境"""
    environment_name = "single object fjsp"
    def __init__(self, DDT, M, S):
        FJSP.__init__(self, DDT, M, S)
        # 封装基本属性
        self.step_count = 0  # 决策步
        self.step_time = 0  # 时间点
        self.last_observation_state = None  # 上一步观察到的状态 v(t-1)
        self.observation_state = None  # 当前时间步的状态 v(t)
        self.state_gap = None  # v(t) - v(t-1)
        self.state = None  # s(t)
        self.next_state = None  # 下一步状态  s(t+1)
        self.reward = None  # 即时奖励
        self.done = False  # 是否为终止状态
        # 动作和观察的状态空间维度
        self.action_space = [6, 6]  # 二维离散动作空间
        self.observation_space = 12  # 观察的状态向量空间
        self.reward_sum = 0  # 累计回报
        # 空闲机器列表和可选工序类型列表
        self.machine_idle_list = []  # 空闲机器编号列表
        self.kind_task_available_list = []  # 可选工序类型编号列表
        # 重置环境状态
        self.reset()
        print("成功定义环境类")

    def reset(self):
        """重置环境状态"""
        # 初始化FJSP类
        self.reset_parameter()  # 初始化参数对象中的列表和字典
        self.reset_object_add(self.order_dict[0])  # 新订单到达后更新各字典对象
        self.machine_idle_list = self.idle_machine()  # 空闲机器编号列表
        self.kind_task_available_list = self.kind_task_available()  # 可选工序类型编号列表
        # 初始化当前时间和时间步
        self.step_count = 0
        self.step_time = 0
        # 初始化状态向量
        self.last_observation_state = self.state_extract()  # 上一步观察到的状态 v(t-1)
        self.observation_state = self.state_extract()  # 当前时间步的状态 v(t)
        self.state_gap = np.array(self.observation_state) - np.array(self.last_observation_state)  # v(t) - v(t-1)
        self.state = np.concatenate((np.array(self.observation_state), self.state_gap), axis=1)  # 状态向量 [v(t), v(t) - v(t-1)]
        self.next_state = None  # 下一步状态
        self.reward = None  # 即时奖励
        self.done = False  # 是否为终止状态
        return self.state

    def state_extract(self):
        """
        提取状态向量
        :return: 观察到的状态向量
        """
        # 计算向量元素
        M = self.machine_count  # 1机器数
        ct_m_ave = sum(machine_object.time_end for m, machine_object in self.machine_dict.items())/self.machine_count
        ct_m_std = math.sqrt(sum(math.pow(machine_object.time_end - ct_m_ave, 2) for m, machine_object in self.machine_dict.items())/self.machine_count)   # 2机器完工时间标准差
        cro_ave = sum(kind_task_object.finish_rate for (r, j), kind_task_object in self.kind_task_dict.items())/len(self.kind_task_tuple)  # 3工序类型完工率均值
        cro_std = math.sqrt(sum(math.pow(kind_task_object.finish_rate - cro_ave, 2) for (r, j), kind_task_object in self.kind_task_dict.items())/len(self.kind_task_tuple))  # 4工序类型完工率标准差
        gap_ave = sum(kind_task_object.gap for (r, j), kind_task_object in self.kind_task_dict.items())/len(self.kind_task_tuple)  # 5工序类型gap_rj均值
        gap_std = math.sqrt(sum(math.pow(kind_task_object.gap - gap_ave, 2) for (r, j), kind_task_object in self.kind_task_dict.items())/len(self.kind_task_tuple))  # 6工序类型gap_rj标准差
        gap_m_ave = sum(machine_object.gap_ave for m, machine_object in self.machine_dict.items())/self.machine_count  # 机器gap_m均值
        gap_m_std = math.sqrt(sum(math.pow(machine_object.gap_ave - gap_m_ave, 2) for m, machine_object in self.machine_dict.items())/self.machine_count)  # 机器gap_m标准差
        dro_a = []

        return [M, ct_m_std, cro_ave, cro_std, gap_ave, gap_std, gap_m_ave, gap_m_std]


    def step(self, action):
        """根据动作选择工序选择规则+机器分配规则"""

        self.state = self.next_state
        return None

    def task_machine(self):
        """返回选择的工序和机器"""
        return None

    def task_select(self, action):
        """根据动作选择工序"""
        return None

    def machine_select(self, action):
        """根据动作选择机器"""
        return None

    def compute_reward(self, achieved_goal, desired_goal, info):
        """根据剩余工件估计延迟时间计算奖励"""
        return None

    def idle_machine(self):
        """返回空闲机器列表"""
        return [m for m in self.machine_tuple if self.machine_dict[m].state == 0]

    def kind_task_available(self):
        """返回可选加工工序列表"""
        return [(r, j) for (r, j) in self.kind_task_tuple if len(self.kind_task_dict[(r, j)].job_now_list) > 0 and
                set(self.kind_task_dict[(r, j)].fluid_machine_list) & set(self.machine_idle_list)]

# 测试环境
if __name__ == '__main__':
    DDT = 1.0
    M = 15
    S = 4
    env_object = SO_DFJSP_Environment(DDT, M, S)