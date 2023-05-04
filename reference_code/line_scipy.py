"""
scipy 求解线性规划代码
"""
def run_fluid_model(self):
    """
    运行流体模型
    输入：各工序类型未加工工序总数、处于该工序类型阶段的工件数
    输出：各机器分配给各工序类型的时间比例、流体最大延迟时间、各工序类型可选加工机器、各机器可选工序类型+加工速率、各工序类型加工速率+时间
    """

    # 生成各流体相关数据
    fluid_number = {(r, j): self.task_kind_dict[(r, j)].fluid_unprocessed_number_start for (r, j) in
                    self.kind_task_tuple}
    fluid_number_time = {(r, j): self.task_kind_dict[(r, j)].fluid_number for (r, j) in self.kind_task_tuple}
    task_end_r_dict = {r: self.task_r_dict[r][-1] for r in self.kind_tuple}
    # 决策变量
    mrj_tuple = tuple((m, (r, j)) for m in self.machine_tuple for (r, j) in self.kind_task_m_dict[m])
    x_mrj_dict = {(m, (r, j)): mrj_tuple.index((m, (r, j))) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    mrj_x_dict = {mrj_tuple.index((m, (r, j))): (m, (r, j)) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    # 各变量的范围
    bound = [(0, 1) for (m, (r, j)) in mrj_tuple]
    # 添加约束条件
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
    """目标函数"""
    # 决策变量
    mrj_tuple = tuple((m, (r, j)) for m in self.machine_tuple for (r, j) in self.kind_task_m_dict[m])
    x_mrj_dict = {(m, (r, j)): mrj_tuple.index((m, (r, j))) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    mrj_x_dict = {mrj_tuple.index((m, (r, j))): (m, (r, j)) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    # 计算相关参数
    fluid_number = {(r, j): self.task_kind_dict[(r, j)].fluid_unprocessed_number_start for (r, j) in
                    self.kind_task_tuple}
    fluid_number_time = {(r, j): self.task_kind_dict[(r, j)].fluid_number for (r, j) in self.kind_task_tuple}
    task_end_rj_dict = {(r, self.task_r_dict[r][-1]) for r in self.kind_tuple}
    fluid_due_date_dict = {
        (r, j): [task_object.due_date for task_object in self.task_kind_dict[(r, j)].task_unprocessed_list] for (r, j)
        in task_end_rj_dict}
    # 目标中的相关参数
    process_rate_sum = {(r, j): sum(
        x[x_mrj_dict[(m, (r, j))]] * self.machine_dict[m].process_rate_rj_dict[(r, j)] for m in
        self.machine_rj_dict[(r, j)]) for (r, j) in task_end_rj_dict}
    time_finish = {(r, j): fluid_number[(r, j)] / process_rate_sum[(r, j)] for (r, j) in task_end_rj_dict}
    # 计算总的延期时间
    fluid_finish_dict = {(r, j): [time_finish[(r, j)] / len(fluid_due_date_dict[(r, j)]) * (number + 1) for number in
                                  range(len(fluid_due_date_dict[(r, j)]))] for (r, j) in task_end_rj_dict}
    due_time_dict = {(r, j): [max(0, c - d) for c, d in zip(fluid_finish_dict[(r, j)], fluid_due_date_dict[(r, j)])] for
                     (r, j) in task_end_rj_dict}
    function_objection = sum(sum(due_time_dict[(r, j)]) for (r, j) in task_end_rj_dict)
    return function_objection