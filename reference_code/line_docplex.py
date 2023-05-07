"""
docplex 求解器
"""
from docplex.mp.model import Model
import numpy as np

class FluidModel():
    """
    求得流体解
    """
    def __init__(self, data, indexed, class_number, class_number_time):
        self.class_number = class_number    # 各类未加工总数列表
        self.class_number_time = class_number_time  # 各类当前重调度点瞬态数量列表
        self.machine_list = indexed.machine_list    # 机器列表
        self.class_list = indexed.class_list    # 类列表
        self.rate_machine_class_index = tuple(indexed.rate_machine_class_index.values())    # 各机器加工各类速率二维列表
        self.number_task = data.number_task  # 各类工件工序数列表
        self.machine_class = data.machine_class  # 各类可选加工机器
        self.number_job_kind = data.number_job_kind  # 工件种类数
        self.class_kind = indexed.class_kind  # 类索引
        self.class_machine = data.class_machine
        # 求解流体模型
        self.machine_class_allocated_time, self.fluid_finish_time = self.line_solve()   # 流体解中各机器分配给各类的时间比例二维列表、流体完工时间
        # 数据处理
        self.class_machine, self.machine_class = self.machine_class_selected()  # 流体解中各类可选加工机器、各机器可选加工类二维列表
        self.machine_class_process_rate, self.class_process_rate = self.process_rate()  # 流体解中各机器处理各类的速率二维列表、各类处理速率列表

    def process_rate(self):
        machine_class_process_rate = np.multiply(np.asarray(self.machine_class_allocated_time),
                                                 np.asarray(self.rate_machine_class_index))
        class_process_rate = np.sum(machine_class_process_rate, axis=0)
        machine_class_process_rate = tuple([tuple(item) for item in machine_class_process_rate])

        return machine_class_process_rate, tuple(class_process_rate)

    def line_solve(self):
        model = Model('LP')
        var_list = {(m, k) for m in self.machine_list for k in self.class_list if m in self.machine_class[k]}
        X = model.continuous_var_dict(var_list, lb=0, ub=1, name='X')
        # 目标函数
        # 1 最大完工时间
        model.maximize(model.min(model.sum(X[m, k] * self.rate_machine_class_index[m][k] for m in self.machine_list if m in self.machine_class[k])/self.class_number[k] for k in self.class_list))
        # 2 机器平均利用率
        # model.maximize(model.sum(model.sum(X[m, k] for k in self.class_list if m in self.machine_class[k]) for m in self.machine_list))

        # 添加约束条件
        model.add_constraints(model.sum(X[m, k] for k in self.class_list if m in self.machine_class[k]) <= 1 for m in self.machine_list)
        model.add_constraints(model.sum(X[m, k] * self.rate_machine_class_index[m][k] for m in self.machine_list if m in self.machine_class[k])/self.class_number[k]
                              >= model.sum(X[m, k+1] * self.rate_machine_class_index[m][k+1] for m in self.machine_list if m in self.machine_class[k+1])/self.class_number[k+1]
                              for r in range(self.number_job_kind) for k in self.class_kind[r][:self.number_task[r] - 1] if self.class_number_time[k+1] == 0)
        # model.add_constraints(model.sum(X[m, k] for m in self.machine_list if k in self.class_machine[m]) >= 0.0001 for k in self.class_list)
        # 求解模型
        solution = model.solve()
        x = solution.get_value_dict(X)
        machine_class_allocated_time = [[x[m, k] if m in self.machine_class[k] else 0 for k in self.class_list] for m in self.machine_list]
        class_process_rate = [sum(machine_class_allocated_time[m][k] * self.rate_machine_class_index[m][k] for m in self.machine_list) for k in self.class_list]
        fluid_finish_time = max([self.class_number[k]/class_process_rate[k] for k in self.class_list])
        print("流体完工时间：", fluid_finish_time)
        return machine_class_allocated_time, fluid_finish_time


    def machine_class_selected(self):
        """
        更新流体解中各机器可选加工类和各类可选加工机器
        :return:
        """
        class_machine = {}
        machine_class = {}
        for m in self.machine_list:
            class_set = []
            for k in self.class_list:
                if self.machine_class_allocated_time[m][k] > 0:
                    class_set.append(k)
            class_machine[m] = tuple(class_set)
        for k in self.class_list:
            machine_set = []
            for m in self.machine_list:
                if self.machine_class_allocated_time[m][k] > 0:
                    machine_set.append(m)
            machine_class[k] = tuple(machine_set)
        return class_machine, machine_class