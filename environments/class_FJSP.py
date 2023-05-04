"""
柔性作业车间调度基础类定义
"""
import copy
import random, math
import time
import numpy as np
from SO_DFJSP_instance import Instance
from data.data_process import Data
import matlab
import matlab.engine
from scipy.optimize import minimize, NonlinearConstraint

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
        self.job_unfinished_list = []  # 未加工完成的工件对象列表
    @property
    def job_number(self):
        """该类型工件已到达工件数"""
        return len(self.job_arrive_list)

class Tasks(Kind):
    """定义工序o_rj类"""
    def __init__(self, r, j):
        Kind.__init__(self, r)  # 调用父类的构函
        # 基本属性
        self.task = j  # 所属工序
        self.machine_tuple = None  # 可选加工机器编号元组
        # 附加属性
        self.job_now_list = []  # 处于该工序段的工件对象列表
        self.job_unprocessed_list = []  # 该工序段未被加工的工件对象列表
        self.task_unprocessed_list = []  # 该工序段还未加工的工序对象列表
        self.task_processed_list = []  # 该工序段已加工的工序对象列表
        # 流体相关属性
        self.fluid_time = None  # 流体模型中该工序的加工时间
        self.fluid_rate = None  # 流体模型中加工该工序的速率
        self.fluid_number = None  # 处于该工序段的流体数量
        self.fluid_unprocessed_number = None  # 未被加工的流体数
        self.fluid_unprocessed_number_start = None  # 订单到达时刻未被加工的流体数
        # 流体机器相关属性
        self.fluid_machine_list = []  # 流体模型中可选加工机器编号列表

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
        return self.job_now_list[0].due_data
    @property
    def due_date_ave(self):
        return sum([job.due_data for job in self.job_unprocessed_list])/len(self.job_unprocessed_list)

class Job(Kind):
    """工件类"""
    def __init__(self, r, n):
        Kind.__init__(self, r)  # 调用父类的构函
        # 基本属性
        self.number = n  # 该工件类型的第几个工件
        # 附加属性
        self.due_date = None  # 该工件的交期
        self.task_list = []  # 已处理工序对象列表

class Task(Tasks, Job):
    """工序类"""
    def __init__(self, r, n, j):
        Tasks.__init__(self, r, j)  # 调用父类的构函
        Job.__init__(self, r, n)  # 调用父类构函
        # 附加属性
        self.machine = None  # 选择的机器
        self.time_end = None  # 加工结束时间
        self.time_begin = None  # 加工开始时间
        self.time_cost = None  # 加工时间

class Machine():
    """机器类"""
    def __init__(self, m):
        # 基本属性
        self.machine_node = m  # 机器编号
        self.kind_task_tuple = None  # 可选加工工序类型元组
        self.process_rate_rj_dict = {}  # 加工各工序类型的速率
        # 附加属性
        self.machine_state = 0  # 机器状态
        self.time_end = 0  # 机器完工时间
        self.task_list = []  # 机器已加工工序对象列表
        self.unprocessed_rj_dict = {}  # 未被m加工的各工序类型的工序总数
        # 流体附加属性
        self.fluid_kind_task_list = []  # 流体解中可选加工工序类型列表
        self.time_rate_rj_dict = {}  # 流体解中分配给各工序类型的时间比例
        self.fluid_process_rate_rj_dict = {}  # 流体解中加工各工序类型的速率
        self.gap_rj_dict = {}  # 流体gap_mrj值 rj
        self.fluid_unprocessed_rj_dict = {}  # 未被机器m加工的各工序类型总数
        self.fluid_unprocessed_rj_arrival_dict = {}  # 订单到达时刻未被m加工的各工序类型数量

    def utilize_rate(self, step_time):
        """利用率"""
        return sum([task.time_cost for task in self.task_list])/max(step_time, self.time_end)

# 问题实例类
class FJSP(Instance):
    """柔性作业车间调度类"""
    def __init__(self, DDT, M, S):
        Instance.__init__(self, DDT, M, S)
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

        # 实例化工件类型、工件、工序类型、工序和机器对象字典
        self.task_kind_dict = {(r, j): Tasks(r, j) for r in self.kind_tuple for j in self.task_r_dict[r]}  # 工序类型对象字典
        self.order_dict = {s: Order(s, self.time_arrive_s_dict[s], self.time_delivery_s_dict[s], self.count_sr_dict[s])
                           for s in self.order_tuple}  # 对象订单字典
        self.kind_dict = {r: Kind(r) for r in self.kind_tuple}  # 工件类型对象字典
        self.machine_dict = {m: Machine(m) for m in self.machine_tuple}  # 机器对象字典
        self.task_kind_number_dict = {}  # (r,n,j) 工序对象字典 订单到达更新
        self.job_dict = {}  # (r,n)  # 工件对象字典
        self.reset_parameter()  # 初始化参数对象中的列表和字典
        # 流体号和工序类型号互相索引
        self.fluid_tuple = tuple(fluid for fluid in range(len(self.kind_task_tuple)))  # 流体编号
        self.kind_task_fluid_dict = {fluid: self.kind_task_tuple[fluid] for fluid in self.fluid_tuple}  # 流体对应的工序类型
        self.fluid_kind_task_dict = {self.kind_task_tuple[fluid]: fluid for fluid in self.fluid_tuple}  # 工序类型对应的流体
        self.process_rate_m_rj_list = [[self.machine_dict[m].process_rate_rj_dict[(r, j)] if (r, j) in self.kind_task_m_dict[m] else 0
                                        for (r, j) in self.kind_task_tuple] for m in self.machine_tuple]  # 机器加工流体速率
        self.task_number_r_list = [len(self.task_r_dict[r]) for r in self.kind_tuple]  # 各工序类型的工序数
        self.fluid_end_tuple = tuple(self.fluid_kind_task_dict[(r, self.task_r_dict[r][-1])] for r in self.kind_tuple)
        # 初始化各对象属性# 新订单到达后更新各字典对象
        self.reset_object_add()
        # 初始化空闲机器列表和可选工序类型列表
        self.machine_idle = []  # 空闲机器编号列表
        self.kind_task_list = []  # 可选工序类型编号列表


    def reset_parameter(self):
        """初始化各字典和参数"""
        for r, kind in self.kind_dict.items():
            kind.job_arrive_list = []  # 已经到达的工件对象列表
            kind.job_unfinished_list = []  # 未加工完成的工件对象列表
        for (r, j), task_kind_object in self.task_kind_dict.items():
            task_kind_object.machine_tuple = self.machine_rj_dict[(r, j)]  # 可选加工机器编号元组
            task_kind_object.job_now_list = []  # 处于该工序段的工件对象列表
            task_kind_object.job_unprocessed_list = []  # 该工序段未被加工的工件对象列表
            task_kind_object.task_unprocessed_list = []  # 该工序段还未加工的工序对象列表
            task_kind_object.task_processed_list = []  # 该工序段已加工的工序对象列表
            task_kind_object.fluid_machine_list = []  # 流体模型中可选加工机器
        for m, machine_object in self.machine_dict.items():
            machine_object.kind_task_tuple = self.kind_task_m_dict[m]  # 可选加工工序类型元组
            machine_object.machine_state = 0  # 机器状态
            machine_object.time_end = 0  # 机器完工时间
            machine_object.task_list = []  # 机器已加工工序对象列表
            machine_object.process_rate_rj_dict = {(r, j): 1/self.time_mrj_dict[m][(r, j)]
                                                   for (r, j) in self.kind_task_m_dict[m]}  # 加工各工序类型的速率
            machine_object.unprocessed_rj_dict = {}  # 未被m加工的工序o_rj的总数 (r,j)
            # 流体附加属性
            machine_object.fluid_kind_task_list = []  # 流体解中可选加工工序类型列表
            machine_object.time_rate_rj_dict = {}  # 流体解中分配给各工序类型的时间比例
            machine_object.fluid_process_rate_rj_dict = {}  # 流体解中加工各工序类型的速率
            machine_object.gap_rj_dict = {}  # 流体gap_mrj值 rj
            machine_object.fluid_unprocessed_rj_dict = {}  # 流体解中未被机器m加工的各工序类型总数
            machine_object.fluid_unprocessed_rj_arrival_dict = {}  # 订单到达时刻未被m加工的各工序类型数量

        return None

    def reset_object_add(self):
        """添加字典对象"""
        order_object = self.order_dict[0]  # 到达的新订单
        # 更新工件类型字典、工序类型对象字典、工序对象字典、工件对象字典、
        for r in self.kind_tuple:
            n_start = len(self.kind_dict[r].job_arrive_list)
            n_end = n_start + order_object.count_kind[r]
            for n in range(n_start, n_end):
                job_object = Job(r, n)  # 工件对象
                job_object.due_date = order_object.time_delivery  # 工件交期
                job_object.task_list = []
                self.kind_dict[r].job_arrive_list.append(job_object)
                self.kind_dict[r].job_unfinished_list.append(job_object)
                self.job_dict[(r, n)] = job_object  # 加入工件字典
                self.task_kind_dict[(r, 0)].job_now_list.append(job_object)
                for j in self.task_r_dict[r]:
                    task_object = Task(r, n, j)  # 工序对象
                    task_object.due_date = self.job_dict[(r, n)].due_date  # 工序交期
                    self.task_kind_dict[(r, j)].job_unprocessed_list.append(job_object)
                    self.task_kind_dict[(r, j)].task_unprocessed_list.append(task_object)
                    self.task_kind_number_dict[(r, n, j)] = task_object  # 加入工序字典
        # 初始化流体属性
        for (r, j), task_kind_object in self.task_kind_dict.items():
            task_kind_object.fluid_number = len(task_kind_object.job_now_list)  # 处于该工序段的流体数量
            task_kind_object.fluid_unprocessed_number = len(task_kind_object.task_unprocessed_list)  # 未被加工的流体数
            task_kind_object.fluid_unprocessed_number_start = len(task_kind_object.task_unprocessed_list)  # 订单到达时刻未被加工的流体数量
        # 求解流体模型更新流体模型属性
        self.fluid_model_delivery()

    def fluid_model_delivery(self):
        """
        运行流体模型——————基于matlab引擎
        输入：各工序类型未加工工序总数、处于该工序类型阶段的工件数
        输出：各机器分配给各工序类型的时间比例、流体最大延迟时间、各工序类型可选加工机器、各机器可选工序类型+加工速率、各工序类型加工速率+时间
        """
        # 生成各流体相关数据
        fluid_number_list = [self.task_kind_dict[self.kind_task_fluid_dict[fluid]].fluid_unprocessed_number_start for fluid in self.fluid_tuple]
        fluid_number_now_list = [self.task_kind_dict[self.kind_task_fluid_dict[fluid]].fluid_number for fluid in self.fluid_tuple]
        # 计算每种工件的末尾流体每个流体的交期时间
        fluid_end_number_list = [fluid_number_list[fluid] for fluid in self.fluid_end_tuple]  # 各工序类型最后一道工序的待加工数量
        fluid_end_number_max = max(fluid_end_number_list)  # 最大值
        # 初始化各工件类型最后一道工序类型的各工件的交期时间
        fluid_end_number_delivery = np.zeros([len(fluid_end_number_list), fluid_end_number_max])
        # 更新每个工件的交期
        for f, fluid in enumerate(self.fluid_end_tuple):
            rj_fluid = self.kind_task_fluid_dict[fluid]
            for n, task_object in enumerate(self.task_kind_dict[rj_fluid].task_unprocessed_list):
                fluid_end_number_delivery[f, n] = task_object.due_date - self.step_time
        # 启动matlab引擎求解模型
        time_start = time.clock()  # matlab引擎求解开始时间
        engine = matlab.engine.start_matlab()  # 启动引擎
        fluid_solve = engine.line_cplex(matlab.double(self.machine_tuple), matlab.double(self.fluid_tuple),
                                        matlab.double(self.process_rate_m_rj_list), matlab.double(fluid_number_list),
                                        matlab.double(fluid_number_now_list), matlab.double(self.task_number_r_list),
                                        matlab.double(fluid_end_number_delivery), nargout=2)
        engine.exit()  # 关闭引擎
        print("matlab求解耗时：", time.clock() - time_start)

        return fluid_solve[0], fluid_solve[1]

# 测试环境
if __name__ == '__main__':
    DDT = 1.0
    M = 15
    S = 4
    fjsp_object = FJSP(DDT, M, S)
