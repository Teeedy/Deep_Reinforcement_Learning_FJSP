"""
柔性作业车间调度基础类定义
"""
import copy
import random, math
import time
import numpy as np
from SO_DFJSP_instance import Instance
from docplex.mp.model import Model
from SO_DFJSP_instance_read import Data

class Order():
    """订单对象"""
    def __init__(self, s, arrive, delivery, count_kind):
        # 基本属性
        self.order_node = s
        self.time_arrive = arrive  # 到达时间
        self.time_delivery = delivery  # 交期时间
        self.count_kind = count_kind  # 包含的各种工件的数量

class Kind():
    """工件类型类"""
    def __init__(self, r):
        self.kind = r
        self.job_arrive_list = []  # 已经到达的工件对象列表
        self.job_unprocessed_list = []  # 未加工完成的工件对象列表
    @property
    def number_start(self):
        """该类型工件已到达工件数:下一阶段的工件n其实编号"""
        return len(self.job_arrive_list)

class Tasks(Kind):
    """定义工序o_rj类"""
    def __init__(self, r, j):
        Kind.__init__(self, r)  # 调用父类的构函
        # 基本属性
        self.task = j  # 所属工序
        self.machine_tuple = None  # 可选加工机器编号
        # 附加属性
        self.job_now_list = []  # 处于该工序段的工件对象列表
        self.job_unprocessed_list = []  # 该工序段未被加工的工件对象列表
        self.task_unprocessed_list = []  # 该工序段还未加工的工序对象列表
        self.task_processed_list = []  # 该工序段已加工的工序对象列表
        # 流体相关属性
        self.fluid_process_rate_m_dict = {}  # 流体中被各机器加工的速率
        self.fluid_machine_list = []  # 流体模型解可选加工机器
        self.fluid_time_sum = None  # 流体模型中该工序的加工时间
        self.fluid_rate_sum = None  # 流体模型中加工该工序的速率
        self.fluid_number = None  # 处于该工序段的流体数量
        self.fluid_unprocessed_number = None  # 未被加工的流体数
        self.fluid_unprocessed_number_start = None  # 订单到达时刻未被加工的流体数

    # 计算属性
    @property
    def gap(self):
        """流体gap_rj值"""
        return (len(self.task_unprocessed_list) - self.fluid_unprocessed_number)/self.fluid_unprocessed_number_start
    @property
    def finish_rate(self):
        """o_rj完成率"""
        return len(self.task_processed_list)/(len(self.task_unprocessed_list) + len(self.task_processed_list))
    @property
    def due_date_min(self):
        """工序rj阶段工件的交期最小值"""
        return self.job_now_list[0].due_date

class Job(Kind):
    """工件类"""
    def __init__(self, r, n):
        Kind.__init__(self, r)  # 调用父类的构函
        # 基本属性
        self.number = n  # 该工件类型的第几个工件
        # 附加属性
        self.due_date = None  # 该工件的交期
        self.task_list = []  # 分配机器的工序对象列表
        self.task_unprocessed_list = []  # 未分配机器的工序对象列表

class Task(Tasks, Job):
    """工序类"""
    def __init__(self, r, n, j):
        Tasks.__init__(self, r, j)  # 调用父类的构函
        Job.__init__(self, r, n)  # 调用父类构函
        # 附加属性
        self.machine = None  # 选择的机器
        self.time_end = None  # 加工结束时间
        self.time_begin = None  # 加工开始时间
    @property
    def time_cost(self):
        """加工耗时"""
        return self.time_end - self.time_begin

class Machine():
    """机器类"""
    def __init__(self, m):
        # 基本属性
        self.machine_node = m  # 机器编号
        self.kind_task_tuple = None  # 可选加工工序类型元组
        # 附加属性
        self.state = 0  # 机器状态
        self.time_end = 0  # 机器完工时间
        self.task_list = []  # 机器已加工工序对象列表
        self.job_object = None  # 机器正在处理的工件对象
        self.unprocessed_rj_dict = {}  # 未被m加工的各工序类型的工序总数/随着加工过程动态更新
        # 流体附加属性
        self.fluid_kind_task_list = []  # 可选加工工序类型
        self.time_ratio_rj_dict = {}  # 流体解中分配给各工序类型的时间比例
        self.fluid_process_rate_rj_dict = {}  # 流体解中加工各工序类型的速率
        self.fluid_unprocessed_rj_dict = {}  # 未被机器m加工的各工序类型流体总数
        self.fluid_unprocessed_rj_arrival_dict = {}  # 订单到达时刻未被m加工的各工序类型流体数

    def utilize_rate(self, step_time):
        """利用率"""
        return sum([task.time_cost for task in self.task_list])/max(step_time, self.time_end)
    @property
    def gap_rj_dict(self):
        """计算gap_mrj值"""
        gap_rj_dict = {}
        for (r, j) in self.fluid_kind_task_list:
            gap_rj_dict[(r, j)] = (self.unprocessed_rj_dict[(r, j)] -
                                   self.fluid_unprocessed_rj_dict[(r, j)])/self.fluid_unprocessed_rj_arrival_dict[(r, j)]
        return gap_rj_dict
    @property
    def gap_ave(self):
        """计算gap_m_rj的均值"""
        if len(self.fluid_kind_task_list) > 0:
            return sum(self.gap_rj_dict[(r, j)] for (r, j) in self.fluid_kind_task_list)/len(self.fluid_kind_task_list)
        else:
            return 0

# 问题实例类
class FJSP(Instance):
    """柔性作业车间调度类"""
    def __init__(self, DDT, M, S):
        Instance.__init__(self, DDT, M, S)
        # 实例化工件类型、工件、工序类型、工序和机器对象字典
        self.kind_task_dict = {(r, j): Tasks(r, j) for r in self.kind_tuple for j in self.task_r_dict[r]}  # 工序类型对象字典
        self.order_dict = {s: Order(s, self.time_arrive_s_dict[s], self.time_delivery_s_dict[s], self.count_sr_dict[s])
                           for s in self.order_tuple}  # 对象订单字典
        self.order_object_list = [self.order_dict[s] for s in self.order_tuple]  # 未到达订单对象列表
        self.kind_dict = {r: Kind(r) for r in self.kind_tuple}  # 工件类型对象字典
        self.machine_dict = {m: Machine(m) for m in self.machine_tuple}  # 机器对象字典
        self.task_dict = {}  # (r,n,j) 工序对象字典 订单到达更新
        self.job_dict = {}  # (r,n)  # 工件对象字典
        self.process_rate_m_rj_dict = {m: {(r, j): 1 / self.time_mrj_dict[m][(r, j)]
                                           for (r, j) in self.kind_task_m_dict[m]} for m in self.machine_tuple}  # 机器加工流体速率

        # 初始化参数对象中的列表和字典
        # self.reset_parameter()
        # 初始化各对象属性# 新订单到达后更新各字典对象
        # self.reset_object_add(self.order_dict[0])
        # print("成功定义FJSP类")

    def reset_parameter(self):
        """初始化各字典和参数"""
        for r, kind in self.kind_dict.items():
            kind.job_arrive_list = []  # 已经到达的工件对象列表
            kind.job_unprocessed_list = []  # 未分配机器的工件对象列表
        for (r, j), kind_task_object in self.kind_task_dict.items():
            kind_task_object.machine_tuple = self.machine_rj_dict[(r, j)]  # 可选加工机器编号元组
            kind_task_object.job_now_list = []  # 处于该工序段的工件对象列表
            kind_task_object.job_unprocessed_list = []  # 该工序段未被加工的工件对象列表
            kind_task_object.task_unprocessed_list = []  # 该工序段还未加工的工序对象列表
            kind_task_object.task_processed_list = []  # 该工序段已加工的工序对象列表
        for m, machine_object in self.machine_dict.items():
            machine_object.kind_task_tuple = self.kind_task_m_dict[m]  # 可选加工工序类型元组
            machine_object.machine_state = 0  # 机器状态
            machine_object.time_end = 0  # 机器完工时间
            machine_object.task_list = []  # 机器已加工工序对象列表
            machine_object.job_object = None

    def reset_fluid_parameter(self):
        """初始化流体属性参数"""
        for (r, j), kind_task_object in self.kind_task_dict.items():
            kind_task_object.fluid_machine_list = []  # 流体模型中可选加工机器
            kind_task_object.fluid_process_rate_m_dict = {}  # 被各机器加工的速率
        for m, machine_object in self.machine_dict.items():
            machine_object.fluid_kind_task_list = []  # 流体解中可选加工工序类型列表
            machine_object.time_ratio_rj_dict = {}  # 流体解中分配给各工序类型的时间比例
            machine_object.fluid_process_rate_rj_dict = {}  # 流体解中加工各工序类型的速率
            machine_object.unprocessed_rj_dict = {}  # 未被m加工的工序o_rj的总数 (r,j)
            machine_object.fluid_unprocessed_rj_dict = {}  # 流体解中未被机器m加工的各工序类型总数
            machine_object.fluid_unprocessed_rj_arrival_dict = {}  # 订单到达时刻未被m加工的各工序类型数量

    def reset_object_add(self, order_object):
        """
        :param order_object: 新到达的订单对象
        :return: 添加工序对象和机器对象+更新流体模型和属性
        """
        # 更新工件类型字典、工序类型对象字典、工序对象字典、工件对象字典、
        for r in self.kind_tuple:
            n_start = self.kind_dict[r].number_start
            n_end = n_start + order_object.count_kind[r]
            for n in range(n_start, n_end):
                job_object = Job(r, n)  # 工件对象
                job_object.due_date = order_object.time_delivery  # 工件交期
                job_object.task_list = []
                job_object.task_unprocessed_list = []
                self.kind_dict[r].job_arrive_list.append(job_object)
                self.kind_dict[r].job_unprocessed_list.append(job_object)
                self.job_dict[(r, n)] = job_object  # 加入工件字典
                self.kind_task_dict[(r, 0)].job_now_list.append(job_object)
                for j in self.task_r_dict[r]:
                    task_object = Task(r, n, j)  # 工序对象
                    job_object.task_unprocessed_list.append(task_object)  # 加入工序未处理工序对象字典
                    task_object.due_date = self.job_dict[(r, n)].due_date  # 工序交期
                    self.kind_task_dict[(r, j)].job_unprocessed_list.append(job_object)
                    self.kind_task_dict[(r, j)].task_unprocessed_list.append(task_object)
                    self.task_dict[(r, n, j)] = task_object  # 加入工序字典
        # 初始化流体属性
        for (r, j), task_kind_object in self.kind_task_dict.items():
            task_kind_object.fluid_number = len(task_kind_object.job_now_list)  # 处于该工序段的流体数量
            task_kind_object.fluid_unprocessed_number = len(task_kind_object.task_unprocessed_list)  # 未被加工的流体数
            task_kind_object.fluid_unprocessed_number_start = len(task_kind_object.task_unprocessed_list)  # 订单到达时刻未被加工的流体数量
        # 求解流体模型更新流体模型属性
        x = self.fluid_model()
        # 初始化流体属性
        self.reset_fluid_parameter()
        # 基于流体解更新流体属性
        self.update_fluid_parameter(x)


    def fluid_model(self):
        """
        最小化最大完工时间目标流体模型求解
        """
        # 初始化模型对象
        model = Model('LP')
        # 定义决策变量
        var_list = {(m, (r, j)) for m in self.machine_tuple for (r, j) in self.kind_task_m_dict[m]}
        # 定义决策变量上下界
        X = model.continuous_var_dict(var_list, lb=0, ub=1, name='X')
        # 各流体初始未加工数量
        fluid_number = {(r, j): self.kind_task_dict[(r, j)].fluid_unprocessed_number_start
                        for (r, j) in self.kind_task_tuple}
        # 各流体初始瞬态数量
        fluid_number_time = {(r, j): self.kind_task_dict[(r, j)].fluid_number for (r, j) in self.kind_task_tuple}
        process_rate_rj_sum = {(r, j): sum(X[m, (r, j)] * self.process_rate_m_rj_dict[m][(r, j)]
                                           for m in self.machine_rj_dict[(r, j)]) for (r, j) in self.kind_task_tuple}
        # 定义目标函数
        model.maximize(model.min(process_rate_rj_sum[(r, j)]/fluid_number[(r, j)] for (r, j) in self.kind_task_tuple))
        # 添加约束条件
        # 机器利用率约束
        model.add_constraints(model.sum(X[m, (r, j)] for (r, j) in self.kind_task_m_dict[m]) <= 1
                              for m in self.machine_tuple)
        # 解的可行性约束
        model.add_constraints(process_rate_rj_sum[(r, j)] >= process_rate_rj_sum[(r, j+1)] for r in self.kind_tuple
                              for j in self.task_r_dict[r][:-2] if fluid_number_time[(r, j+1)] == 0)
        # 求解模型
        solution = model.solve()
        x = solution.get_value_dict(X)
        # 输出流体完工时间
        process_rate_rj_sum = {(r, j): sum(x[m, (r, j)] * self.process_rate_m_rj_dict[m][(r, j)]
                                           for m in self.machine_rj_dict[(r, j)]) for (r, j) in self.kind_task_tuple}
        fluid_makespan = max(fluid_number[(r, j)]/process_rate_rj_sum[(r, j)] for (r, j) in self.kind_task_tuple)
        print("流体完工时间：", fluid_makespan)
        return x

    def update_fluid_parameter(self, x):
        """基于流体解更新流体参数"""
        for (m, (r, j)), rate in x.items():
            if rate != 0:
                machine_object = self.machine_dict[m]
                kind_task_object = self.kind_task_dict[(r, j)]
                machine_object.fluid_kind_task_list.append((r, j))
                kind_task_object.fluid_machine_list.append(m)
                machine_object.time_ratio_rj_dict[(r, j)] = rate  # 流体解中分配给各工序类型的时间比例
                kind_task_object.fluid_process_rate_m_dict[m] = rate*self.process_rate_m_rj_dict[m][(r, j)]
                machine_object.fluid_process_rate_rj_dict[(r, j)] = rate*self.process_rate_m_rj_dict[m][(r, j)]
        for (r, j), kind_task_object in self.kind_task_dict.items():
            kind_task_object.fluid_rate_sum = sum(kind_task_object.fluid_process_rate_m_dict.values())
            kind_task_object.fluid_time_sum = 1/kind_task_object.fluid_rate_sum
        for m, machine_object in self.machine_dict.items():
            for (r, j) in machine_object.fluid_kind_task_list:
                kind_task_object = self.kind_task_dict[(r, j)]
                # 订单到达时刻未被m加工的各工序类型数量
                machine_object.fluid_unprocessed_rj_arrival_dict[(r, j)] = \
                    kind_task_object.fluid_unprocessed_number_start*\
                    machine_object.fluid_process_rate_rj_dict[(r, j)]/kind_task_object.fluid_rate_sum
                # 未被m加工的工序o_rj的总数 (r,j)
                machine_object.unprocessed_rj_dict[(r, j)] = machine_object.fluid_unprocessed_rj_arrival_dict[(r, j)]
                # 流体解中未被机器m加工的各工序类型总数
                machine_object.fluid_unprocessed_rj_dict[(r, j)] = machine_object.fluid_unprocessed_rj_arrival_dict[(r, j)]
# 测试环境
if __name__ == '__main__':
    DDT = 1.0
    M = 15
    S = 4
    fjsp_object = FJSP(DDT, M, S)
