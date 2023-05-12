"""
单目标动态柔性作业车间调度环境
目标：总延期时间
动态因素：新订单到达
"""
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
        self.order_arrive_time = 0  # 顶顶那到达时间点
        self.last_observation_state = None  # 上一步观察到的状态 v(t-1)
        self.observation_state = None  # 当前时间步的状态 v(t)
        self.state_gap = None  # v(t) - v(t-1)
        self.state = None  # s(t)
        self.next_state = None  # 下一步状态  s(t+1)
        self.reward = None  # 即时奖励
        self.done = False  # 是否为终止状态
        # 动作和观察的状态空间维度
        self.actions_size = [6, 4]  # 二维离散动作空间
        self.state_size = 24  # 状态空间
        self.action_types = "DISCRETE"  # 动作类型
        self.observation_space = 12  # 观察的状态向量空间
        self.reward_sum = 0  # 累计回报
        # 工序和机器选择规则相关属性
        self.kind_task_delay_e_list = []  # 估计延期工序类型列表
        self.kind_task_delay_a_list = []  # 实际延期工序类型列表
        self.kind_task_delay_time_a = {}  # 工序类型实际延期时间
        self.kind_task_delay_time_e = {}  # 工序类型估计延期时间
        self.kind_task_due_date = {}  # 工序类型的最小交期
        # 回报计算相关属性
        self.delay_time_sum_last = None  # 上一决策步的剩余工件估计总延期时间
        self.delay_time_sum = None  # 剩余工件总的估计延期时间
        self.delay_time_sum_a = None  # 工件实际总的延期时间
        # 输出订单到达时间和交期时间
        print("成功定义环境类")

    def reset(self):
        """重置环境状态"""
        # 初始化FJSP类
        print("剩余订单数", len(self.order_object_list))
        self.reset_parameter()  # 初始化参数对象中的列表和字典
        self.reset_object_add(self.order_object_list[0])  # 新订单到达后更新各字典对象
        self.order_object_list.remove(self.order_object_list[0])  # 更新未到达订单对象列表
        # 初始化当前时间和时间步
        self.delay_time_sum_a = 0  # 工件实际总延期时间
        self.step_count = 0
        self.step_time = 0
        self.reward_sum = 0  # 累计回报
        # 初始化last时间步
        self.last_observation_state = self.state_extract()  # 上一步观察到的状态 v(t-1)
        self.delay_time_sum_last = self.delay_time_sum  # 上一时间步的估计总延期时间
        # 初始化初始时间步
        self.observation_state = self.state_extract()  # 当前时间步的状态 v(t)
        self.state_gap = np.array(self.observation_state) - np.array(self.last_observation_state)  # v(t) - v(t-1)
        self.state = np.concatenate((np.array(self.observation_state), self.state_gap))  # 状态向量 [v(t), v(t) - v(t-1)]
        self.next_state = None  # 下一步状态
        self.reward = None  # 即时奖励
        self.done = False  # 是否为终止状态
        return self.state

    def state_extract(self):
        """
        提取状态向量
        更新相关参数
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
        dro_a, dro_e, drj_a, drj_e = self.update_parameter()  # 返回工序实际和估计延迟率、工件实际和估计延迟率+更新相关参数
        return [M, ct_m_std, cro_ave, cro_std, gap_ave, gap_std, gap_m_ave, gap_m_std, dro_a, dro_e, drj_a, drj_e]

    def update_parameter(self):
        """
        计算当前时间步：剩余工序实际延迟率和估计延迟率+剩余工件实际延迟率+估计延迟率
        更新参数：工序类型的估计/实际延期时间最大值+工序类型的最小交期值+实际/估计延迟工序类型集合
        """
        delay_task_number_a = 0  # 实际延迟工序总数
        delay_task_number_e = 0  # 估计延迟工序总数
        task_number = 0  # 剩余工序总数
        delay_job_number_a = 0  # 实际延迟工件总数
        delay_job_number_e = 0  # 估计延迟工件总数
        job_number = 0  # 剩余工件总数
        self.delay_time_sum = 0  # 初始化估计延期时间
        self.kind_task_delay_e_list = []  # 估计延期工序类型列表
        self.kind_task_delay_a_list = []  # 实际延期工序类型列表
        # 计算各状态向量的元素
        for (r, j), kind_task_object in self.kind_task_dict.items():
            time_to_end = sum(self.kind_task_dict[(r, jj)].fluid_time_sum for jj in self.task_r_dict[r][j:])
            # 更新工序类型的估计延期时间+实际延期时间+最小交期时间
            if (r, j) in self.kind_task_available_list:
                self.kind_task_delay_time_a[(r, j)] = self.step_time - kind_task_object.due_date_min
                if self.kind_task_delay_time_a[(r, j)] > 0:
                    self.kind_task_delay_a_list.append((r, j))  # 实际延期工序类型列表
                self.kind_task_delay_time_e[(r, j)] = self.step_time + time_to_end - kind_task_object.due_date_min
                if self.kind_task_delay_time_e[(r, j)] > 0:
                    self.kind_task_delay_e_list.append((r, j))  # 估计延期工序类型列表
                self.kind_task_due_date[(r, j)] = kind_task_object.due_date_min
            else:
                self.kind_task_delay_time_a[(r, j)] = None
                self.kind_task_delay_time_e[(r, j)] = None
                self.kind_task_due_date[(r, j)] = None
            # 计算各延迟率
            for job_index, job_object in enumerate(kind_task_object.job_now_list):
                job_number += 1  # 剩余工件总数
                job_delay_time = 0  # 初始化该工件的估计延迟时间
                if job_object.due_date < self.step_time:
                    delay_job_number_a += 1  # 实际延迟工件数
                task_time_start = self.step_time + job_index * kind_task_object.fluid_time_sum  # 初始化工件在该工序段开工时间
                for task_object in job_object.task_unprocessed_list:
                    task_number += 1  # 剩余工序总数
                    if task_object.due_date < self.step_time:
                        delay_task_number_a += 1  # 实际延迟工序数
                    task_time_end = task_time_start + self.kind_task_dict[(task_object.kind, task_object.task)].fluid_time_sum  # 该工序的估计完工时间
                    task_time_start = task_time_end  # 下一工序的开工时间
                    if task_time_end > task_object.due_date:
                        delay_task_number_e += 1
                        job_delay_time = task_time_end - task_object.due_date  # 该工序的估计延期时间
                if job_delay_time > 0:
                    delay_job_number_e += 1  # 更新估计延迟工件数
                self.delay_time_sum += max(job_delay_time, 0)  # 更新剩余工件总的估计延期时间
        # 根据各机器上正在加工的工件对象再次更新对应参数
        for m, machine_object in self.machine_dict.items():
            job_object = machine_object.job_object
            if job_object is not None and len(job_object.task_unprocessed_list) > 0:
                job_number += 1  # 更新未完工工件总数
                job_delay_time = 0  # 初始化该工件的估计延迟时间
                if job_object.due_date < self.step_time:
                    delay_job_number_a += 1  # 更新实际延迟工件数
                task_time_start = job_object.task_list[-1].time_end  # 初始化未分配机器工序开工时间
                for task_object in job_object.task_unprocessed_list:
                    task_number += 1  # 更新未完工工序总数
                    if task_object.due_date < self.step_time:
                        delay_task_number_a += 1  # 剩余工序实际延迟工序数
                    task_time_end = task_time_start + self.kind_task_dict[(task_object.kind, task_object.task)].fluid_time_sum # 该工序的估计完工时间
                    task_time_start = task_time_end  # 下一工序的开工时间
                    if task_time_end > task_object.due_date:
                        delay_task_number_e += 1  # 剩余工序估计延迟工序数
                        job_delay_time = task_time_end - task_object.due_date
                if job_delay_time > 0:
                    delay_job_number_e += 1  # 更新估计延迟工件数
                self.delay_time_sum += max(job_delay_time, 0)  # 更新剩余工件总的估计延期时间
        # 输出延迟率
        if self.done:
            dro_a = 0  # 实际剩余工序延迟率
            dro_e = 0  # 估计剩余工序延迟率
            drj_a = 0  # 实际剩余工件延迟率
            drj_e = 0  # 估计剩余工件延迟率
        else:
            dro_a = delay_task_number_a/task_number  # 实际剩余工序延迟率
            dro_e = delay_task_number_e/task_number  # 估计剩余工序延迟率
            drj_a = delay_job_number_a/job_number   # 实际剩余工件延迟率
            drj_e = delay_job_number_e/job_number  # 估计剩余工件延迟率
        return dro_a, dro_e, drj_a, drj_e

    def step(self, action):
        """根据动作选择工序选择规则+机器分配规则"""
        task_rule = action[0]  # 工序类型选择规则
        machine_rule = action[1]  # 机器选择规则
        rj_selected = self.task_select(task_rule)  # 选择的工序类型
        m_selected = self.machine_select(machine_rule, rj_selected)  # 选择的机器
        # 定义相关对象
        job_object_selected = self.kind_task_dict[rj_selected].job_now_list[0]  # 工件对象
        task_object_selected = job_object_selected.task_unprocessed_list[0]  # 工序对象
        kind_task_object_selected = self.kind_task_dict[rj_selected]  # 工序类型对象
        kind_object_selected = self.kind_dict[rj_selected[0]]  # 工件类型对象
        machine_object_selected = self.machine_dict[m_selected]  # 机器对象
        # 更新工序对象属性
        task_object_selected.time_begin = self.step_time  # 工序开工时间
        task_object_selected.machine = m_selected  # 选择的机器
        task_object_selected.time_end = self.step_time + self.time_mrj_dict[m_selected][rj_selected]  # 完工时间
        # 更新工件对象属性
        job_object_selected.task_list.append(task_object_selected)  # 更新已分配机器工序对象列表
        job_object_selected.task_unprocessed_list.remove(task_object_selected)  # 更新未分配机器工序对象列表
        # 更新工序类型对象
        kind_task_object_selected.job_now_list.remove(job_object_selected)  # 更新处于该工序段的工件对象列表
        kind_task_object_selected.job_unprocessed_list.remove(job_object_selected)  # 更新该工序段未加工的工件对象列表
        kind_task_object_selected.task_unprocessed_list.remove(task_object_selected)  # 更新该工序段未加工的工序对象列表
        kind_task_object_selected.task_processed_list.append(task_object_selected)  # 更新该工序段已分配机器的工序对象列表
        # 更新机器对象
        machine_object_selected.state = 1  # 更新机器状态
        machine_object_selected.time_end = task_object_selected.time_end  # 更新机器完工时间
        machine_object_selected.task_list.append(task_object_selected)  # 更新机器已加工工序对象列表
        machine_object_selected.job_object = job_object_selected  # 更新机器正在加工的工件对象
        machine_object_selected.unprocessed_rj_dict[rj_selected] -= 1  # 更新该机器未加工工序rj的数量
        # 更新工件类型对象
        if len(job_object_selected.task_unprocessed_list) == 0:
            kind_object_selected.job_unprocessed_list.remove(job_object_selected)
            self.delay_time_sum_a += max(task_object_selected.time_end - task_object_selected.due_date, 0)
        # 判断是否移动时钟
        while len(self.kind_task_available_list) == 0:
            # 更新当前时间点
            if len(self.order_object_list) > 0:
                print("下个订单的到达时间", self.order_object_list[0].time_arrive)
            time_point_list = [self.machine_dict[m].time_end for m in self.machine_tuple
                               if self.machine_dict[m].time_end > self.step_time]
            self.step_time = min(time_point_list)
            # 更新对象相关属性: 工序类型阶段的工件对象列表
            for m, machine_object in self.machine_dict.items():
                if machine_object.time_end == self.step_time:
                    job_object = machine_object.job_object  # 刚加工完的工件对象
                    if len(job_object.task_unprocessed_list) > 0:
                        task_object = job_object.task_unprocessed_list[0]  # 新到达的工序对象
                        kind_task_object = self.kind_task_dict[(task_object.kind, task_object.task)]  # 对应的工序类型
                        kind_task_object.job_now_list.append(job_object)
                        sorted(kind_task_object.job_now_list, key=lambda x: x.due_date)
            # 判断新订单是否到达
            if len(self.order_object_list) > 0 and self.order_object_list[0].time_arrive <= self.step_time:
                print("____________未加工完时新订单到达____________")
                order_object = self.order_object_list[0]
                self.order_object_list.remove(order_object)
                self.reset_object_add(order_object)
                self.order_arrive_time = order_object.time_arrive
            elif len(self.order_object_list) > 0 and sum(len(kind_object.job_unprocessed_list) for r, kind_object in self.kind_dict.items()) == 0:
                print("*************直接移动到下一个时间点**************")
                order_object = self.order_object_list[0]
                self.order_object_list.remove(order_object)
                self.reset_object_add(order_object)
                self.order_arrive_time = order_object.time_arrive
                self.step_time = self.order_arrive_time  # 时钟移动到新订单到达时间点
            # 更新机器状态和正在加工的工件对象
            for m, machine_object in self.machine_dict.items():
                if machine_object.time_end <= self.step_time:
                    machine_object.state = 0  # 更新机器状态
            # 更新流体相关属性：工序类型流体量、机器-工序类型流体量
            gap_time = self.step_time - self.order_arrive_time  # 流动时间
            for (r, j), kind_task_object in self.kind_task_dict.items():
                kind_task_object.fluid_unprocessed_number = kind_task_object.fluid_unprocessed_number_start - \
                                                            kind_task_object.fluid_rate_sum*gap_time
            for m, machine_object in self.machine_dict.items():
                for (r, j) in machine_object.fluid_kind_task_list:
                    machine_object.fluid_unprocessed_rj_dict[(r, j)] = \
                        machine_object.fluid_unprocessed_rj_arrival_dict[(r, j)] - \
                        gap_time*machine_object.fluid_process_rate_rj_dict[(r, j)]
            # 判断是否终止
            if len(self.order_object_list) == 0 and sum(len(kind_object.job_unprocessed_list) for r, kind_object in self.kind_dict.items()) == 0:
                self.done = True
                print("#############周期结束####################")
                print(len(self.order_object_list))
                break
        # 提取新的状态、计算回报值、判断周期循环是否结束
        self.step_count += 1
        self.last_observation_state = self.observation_state  # 上一步观察到的状态 v(t-1)
        self.delay_time_sum_last = self.delay_time_sum  # 上一时间步的估计总延期时间
        # 初始化初始时间步
        self.observation_state = self.state_extract()  # 当前时间步的状态 v(t)
        self.state_gap = np.array(self.observation_state) - np.array(self.last_observation_state)  # v(t) - v(t-1)
        self.next_state = np.concatenate((np.array(self.observation_state), self.state_gap))  # 状态向量 [v(t), v(t) - v(t-1)]
        self.reward = self.compute_reward()  # 即时奖励
        self.reward_sum += self.reward  # 更新累计回报
        print("reward：", self.reward)
        print("state：", self.observation_state[-4:])
        print("剩余工件估计延期时间：", self.delay_time_sum)
        print("当前时间：", self.step_time)
        self.state = self.next_state
        return self.state, self.reward, self.done

    def task_select(self, task_rule):
        """6个工序选择规则"""
        # 工序选择规则1
        if task_rule == 1:
            if len(self.kind_task_delay_e_list) == 0:
                rj = max(self.kind_task_available_list, key=lambda x: self.kind_task_dict[x].gap)
            else:
                rj = max(self.kind_task_available_list, key=lambda x: self.kind_task_delay_time_e[x])
                rj_delay_time_e_max_list = [(r, j) for (r, j) in self.kind_task_available_list
                                            if self.kind_task_delay_time_e[(r, j)] == self.kind_task_delay_time_e[rj]]
                rj = max(rj_delay_time_e_max_list, key=lambda x: self.kind_task_dict[x].gap)
        # 工序选择规则2
        elif task_rule == 2:
            if len(self.kind_task_delay_a_list) == 0:
                rj = max(self.kind_task_available_list, key=lambda x: self.kind_task_dict[x].gap)
            else:
                rj = max(self.kind_task_available_list, key=lambda x: self.kind_task_delay_time_a[x])
                rj_delay_time_a_max_list = [(r, j) for (r, j) in self.kind_task_available_list
                                            if self.kind_task_delay_time_a[(r, j)] == self.kind_task_delay_time_a[rj]]
                rj = max(rj_delay_time_a_max_list, key=lambda x: self.kind_task_dict[x].gap)
        # 工序选择规则3
        elif task_rule == 3:
            rj = max(self.kind_task_available_list, key=lambda x: self.kind_task_dict[x].gap)
            rj_gap_max_list = [(r, j) for (r, j) in self.kind_task_available_list
                                         if self.kind_task_dict[(r, j)].gap == self.kind_task_dict[rj].gap]
            rj = min(rj_gap_max_list, key=lambda x: self.kind_task_due_date[x])
        # 工序选择规则4
        elif task_rule == 4:
            rj = max(self.kind_task_available_list, key=lambda x: self.kind_task_delay_time_e[x])
            rj_delay_time_e_max_list = [(r, j) for (r, j) in self.kind_task_available_list
                                        if self.kind_task_delay_time_e[(r, j)] == self.kind_task_delay_time_e[rj]]
            rj = max(rj_delay_time_e_max_list, key=lambda x: self.kind_task_dict[x].gap)
        # 工序选择规则5
        elif task_rule == 5:
            rj = max(self.kind_task_available_list, key=lambda x: self.kind_task_delay_time_a[x])
            rj_delay_time_a_max_list = [(r, j) for (r, j) in self.kind_task_available_list
                                        if self.kind_task_delay_time_a[(r, j)] == self.kind_task_delay_time_a[rj]]
            rj = max(rj_delay_time_a_max_list, key=lambda x: self.kind_task_dict[x].gap)
        # 工序选择规则6
        elif task_rule == 6:
            rj = random.choice(self.kind_task_available_list)
        else:
            rj = None
            print("报错：未定义该工序动作规则。")
        return rj

    def machine_select(self, machine_rule, rj_selected):
        """4个机器分配规则"""
        machine_selectable_list = list(set(self.machine_idle_list)&set(self.kind_task_dict[rj_selected].fluid_machine_list))  # 可选机器列表
        # 机器分配规则1
        if machine_rule == 1:
            m = max(machine_selectable_list, key=lambda x: self.machine_dict[x].gap_rj_dict[rj_selected])
        # 机器分配规则2
        elif machine_rule == 2:
            m = min(machine_selectable_list, key=lambda x: self.time_mrj_dict[x][rj_selected])
        # 机器分配规则3
        elif machine_rule == 3:
            m = max(machine_selectable_list, key=lambda x: self.machine_dict[x].gap_ave)
        # 机器分配规则4
        elif machine_rule == 4:
            m = random.choice(machine_selectable_list)
        else:
            m = None
            print("报错：未定义该机器分配规则。")
        return m

    def compute_reward(self):
        """根据剩余工件估计延迟时间计算奖励"""
        delay_time_gap = self.delay_time_sum - self.delay_time_sum_last
        if delay_time_gap < 0:
            self.reward = 1
        elif delay_time_gap == 0:
            self.reward = 0
        else:
            self.reward = -1
        return self.reward

    @property
    def machine_idle_list(self):
        """返回空闲机器列表"""
        return [m for m in self.machine_tuple if self.machine_dict[m].state == 0]
    @property
    def kind_task_available_list(self):
        """返回可选加工工序列表"""
        return [(r, j) for (r, j) in self.kind_task_tuple if len(self.kind_task_dict[(r, j)].job_now_list) > 0 and
                set(self.kind_task_dict[(r, j)].fluid_machine_list) & set(self.machine_idle_list)]

# 测试环境
if __name__ == '__main__':
    DDT = 1.0
    M = 20
    S = 1
    time_start = time.time()
    env_object = SO_DFJSP_Environment(DDT, M, S)  # 定义环境对象
    state = env_object.reset()  # 初始化状态
    replay_list = []
    # 随机选择动作测试环境
    while not env_object.done:
        action = (random.choice([1, 2, 3, 4, 5, 6]), random.choice([1, 2, 3, 4]))  # action = policy_network(state)
        next_state, reward, done = env_object.step(action)
        replay_list.append([state, action, next_state, reward, done])
        state = next_state
    print("累计回报:", env_object.reward_sum)
    print("总步数", env_object.step_count)
    print("订单到达时间", [order_object.time_arrive for s, order_object in env_object.order_dict.items()])
    print("订单交期时间", [order_object.time_delivery for s, order_object in env_object.order_dict.items()])
    print("机器完工时间", [machine_object.time_end for m, machine_object in env_object.machine_dict.items()])
    print("单周期耗时：", time.time() - time_start)
    print("实际延期时间：", env_object.delay_time_sum_a)
