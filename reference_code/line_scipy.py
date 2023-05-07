"""
scipy 线性规划求解器
"""
from scipy.optimize import minimize

def fluid_model(self):
    """
    最大完工时间
    """
    fluid_number = {(r, j): self.task_kind_dict[(r, j)].fluid_unprocessed_number_start for (r, j) in
                    self.kind_task_tuple}
    fluid_number_time = {(r, j): self.task_kind_dict[(r, j)].fluid_number for (r, j) in self.kind_task_tuple}
    task_end_r_dict = {r: self.task_r_dict[r][-1] for r in self.kind_tuple}
    mrj_tuple = tuple((m, (r, j)) for m in self.machine_tuple for (r, j) in self.kind_task_m_dict[m])
    x_mrj_dict = {(m, (r, j)): mrj_tuple.index((m, (r, j))) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    mrj_x_dict = {mrj_tuple.index((m, (r, j))): (m, (r, j)) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    # 定义决策变量边界
    bound = [(0, 1) for (m, (r, j)) in mrj_tuple]
    # 添加约束
    cons_temp = []
    for m in self.machine_tuple:
        cons_temp.append({'type': 'ineq',
                          'fun': lambda x: 1 - sum(x[x_mrj_dict[(m, (r, j))]] for (r, j) in self.kind_task_m_dict[m])})
    cons = tuple(cons_temp)
    # 求解模型
    solution = minimize(self.objective, np.ones(len(mrj_tuple)), bounds=bound, constraints=cons)
    print(solution)
    print(solution.x)
    x_mrj_result_dict = {(m, (r, j)): solution.x[x_mrj_dict[(m, (r, j))]] for m in self.machine_tuple for (r, j) in
                         self.kind_task_m_dict[m]}
    print(x_mrj_result_dict)


def objective(self, x):
    """计算目标值"""
    mrj_tuple = tuple((m, (r, j)) for m in self.machine_tuple for (r, j) in self.kind_task_m_dict[m])
    x_mrj_dict = {(m, (r, j)): mrj_tuple.index((m, (r, j))) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    mrj_x_dict = {mrj_tuple.index((m, (r, j))): (m, (r, j)) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    # 初始化参数
    fluid_number = {(r, j): self.task_kind_dict[(r, j)].fluid_unprocessed_number_start for (r, j) in
                    self.kind_task_tuple}
    fluid_number_time = {(r, j): self.task_kind_dict[(r, j)].fluid_number for (r, j) in self.kind_task_tuple}
    task_end_rj_dict = {(r, self.task_r_dict[r][-1]) for r in self.kind_tuple}
    fluid_due_date_dict = {
        (r, j): [task_object.due_date for task_object in self.task_kind_dict[(r, j)].task_unprocessed_list] for (r, j)
        in task_end_rj_dict}
    # 计算各类的完工时间
    process_rate_sum = {(r, j): sum(
        x[x_mrj_dict[(m, (r, j))]] * self.machine_dict[m].process_rate_rj_dict[(r, j)] for m in
        self.machine_rj_dict[(r, j)]) for (r, j) in task_end_rj_dict}
    time_finish = {(r, j): fluid_number[(r, j)] / process_rate_sum[(r, j)] for (r, j) in task_end_rj_dict}
    # 计算目标函数
    fluid_finish_dict = {(r, j): [time_finish[(r, j)] / len(fluid_due_date_dict[(r, j)]) * (number + 1) for number in
                                  range(len(fluid_due_date_dict[(r, j)]))] for (r, j) in task_end_rj_dict}
    due_time_dict = {(r, j): [max(0, c - d) for c, d in zip(fluid_finish_dict[(r, j)], fluid_due_date_dict[(r, j)])] for
                     (r, j) in task_end_rj_dict}
    function_objection = sum(sum(due_time_dict[(r, j)]) for (r, j) in task_end_rj_dict)
    return function_objection