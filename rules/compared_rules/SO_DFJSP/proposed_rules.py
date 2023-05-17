"""
调度规则对比算法
"""
from environments.SO_DFJSP import SO_DFJSP_Environment
from environments.SO_DFJSP import FigGan
import random
import time

class RulesEnv(SO_DFJSP_Environment):
    def __init__(self, use_instance=True, **kwargs):
        super().__init__(use_instance=use_instance, **kwargs)
        self.rule_dict = {'rule1': [2, 0], 'rule2': [2, 1], 'rule3': [2, 2], 'rule4': [2, 3], 'rule5': [3, 0]} # 各规则对应的动作
        self.task_rules = [0, 1, 2, 3, 4, 5]  # 工序可选规则列表
        self.machine_rules = [0, 1, 2, 3, 4]  # 机器可选规则列表

    @property
    def rule_random(self):
        return [random.choice(self.task_rules), random.choice(self.machine_rules)]


# 测试环境
if __name__ == '__main__':
    file_name_list = []  # 文件列表
    file_name = 'DDT1.0_M15_S1'
    path = 'D:\Python project\Deep_Reinforcement_Learning_FJSP\data\generated'
    # 特定算例下循环固定次数
    objective_list = []  # 存储每次循环的目标值列表
    epoch_number = 10  # 循环次数
    # for file_name in file_name_list:  # 文件循环
    env_object = RulesEnv(use_instance=False, path=path, file_name=file_name)  # 定义环境对象
    # for rule, actions in env_object.rule_dict.items():  # 规则循环
    # for n in range(epoch_number):  # 次数循环---输出最优值+平均值+标准差
    time_start = time.time()
    state = env_object.reset()  # 初始化状态
    while not env_object.done:
        action = [1, 0]
        next_state, reward, done = env_object.step(action)
        state = next_state
        # print(env_object.machine_idle_list)
    print("累计回报:", env_object.reward_sum)
    print("总步数", env_object.step_count)
    print("订单到达时间", [order_object.time_arrive for s, order_object in env_object.order_dict.items()])
    print("订单交期时间", [order_object.time_delivery for s, order_object in env_object.order_dict.items()])
    print("机器完工时间", [machine_object.time_end for m, machine_object in env_object.machine_dict.items()])
    print("最大完工时间", max([machine_object.time_end for m, machine_object in env_object.machine_dict.items()]))
    print("总延期时间：", env_object.delay_time_sum)
    print("单周期耗时：", time.time() - time_start)

    # 画甘特图
    figure_object = FigGan(env_object)
    # figure_object.figure()