"""
双重异步优势演员评论家算法
"""
import copy
import pickle
import random
import time
import numpy as np
import torch
from torch import multiprocessing
from torch.multiprocessing import Queue
from torch.optim import Adam
from agents.Base_Agent import Base_Agent
from utilities.Utility_Functions import create_actor_distribution, SharedAdam
from utilities.OU_Noise import OU_Noise
from environments.SO_DFJSP import SO_DFJSP_Environment
import torch.nn.functional as F
from torch import nn

class MyNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_sizes):
        super(MyNet, self).__init__()

        # 定义第一层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()])

        # 定义中间 hidden_layers 层
        for i in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())

        # 定义最后一层
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 构建策略网络类
class Actor_Net(nn.Module):
    def __init__(self, input_size_1, input_size_2, hidden_size, hidden_layer_1, hidden_layer_2, output_size_1, output_size_2):
        super().__init__()
        # 定义工序策略网络输入层
        self.layers_1 = nn.ModuleList([nn.Linear(input_size_1, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU()])
        # 定义工序策略网络隐藏层
        for i in range(hidden_layer_1 - 1):
            self.layers_1.append(nn.Linear(hidden_size, hidden_size))
            self.layers_1.append(nn.BatchNorm1d(hidden_size))
            self.layers_1.append(nn.ReLU())
        # 定义工序策略网络输出层
        self.layers_1.append(nn.Linear(hidden_size, output_size_1))
        # 定义机器策略网络输入层
        self.layers_2 = nn.ModuleList([nn.Linear(input_size_2, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU()])
        # 定义机器策略网络隐藏层
        for i in range(hidden_layer_2 - 1):
            self.layers_2.append(nn.Linear(hidden_size, hidden_size))
            self.layers_2.append(nn.BatchNorm1d(hidden_size))
            self.layers_2.append(nn.ReLU())
        # 定义机器策略网络输出层
        self.layers_2.append(nn.Linear(hidden_size, output_size_2))
    def forward(self, x):
        for layer in self.layers_1:
            x = layer(x)
        x = F.softmax(x)
        max_index_1 = x.argmax(dim=1).unsqueeze(1)
        x = torch.cat((x, max_index_1.float()), dim=1)
        for layer in self.layers_2:
            x = layer(x)
        x = F.softmax(x)
        max_index_2 = x.argmax(dim=1).unsqueeze(1)
        x = torch.cat((max_index_1.float(), max_index_2.float()), dim=1)
        return x
# 构建评论家网络
class Critic_Net(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super().__init__()
        # 定义评论家网络输入层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU()])
        # 定义评论家网络隐藏层
        for i in range(hidden_layer - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
        # 定义评论家网络输出层
        self.layers.append(nn.Linear(hidden_size, output_size))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class A3C(Base_Agent):
    """Actor critic A3C algorithm"""
    agent_name = "A3C"
    def __init__(self):
        Base_Agent.__init__()
        self.num_processes = multiprocessing.cpu_count()  # 电脑线程数量|四核八线程
        self.worker_processes = max(1, self.num_processes - 2)  # 启用线程数
        self.environment = self.generated_new_environment()  # 初始环境
        # 超参数
        self.state_size = 24  # 状态空间维度
        self.action_size = [6, 4]  # 二维离散动作空间维度
        self.actor_critic = Actor_Net(input_size_1=24, input_size_2=25, hidden_size=200, hidden_layer_1=3,
                                      hidden_layer_2=3, output_size_1=6, output_size_2=4)  # 演员网络
        self.critic_net = Critic_Net(input_size=24, hidden_size=200, hidden_layer=3, output_size=1)  # 评论家网络
        self.actor_critic_optimizer = SharedAdam(self.actor_critic.parameters(), lr=self.hyperparameters["learning_rate"], eps=1e-4)

    def generated_new_environment(self):
        """生成新环境"""
        DDT = random.uniform(0.5, 1.5)
        M = random.randint(10, 20)
        S = random.randint(10, 15)
        return SO_DFJSP_Environment(DDT, M, S)

    def run_n_episodes(self):
        """运行游戏n次直到完成，然后总结结果并保存模型(如果要求的话)"""
        start = time.time()
        results_queue = Queue()
        gradient_updates_queue = Queue()
        episode_number = multiprocessing.Value('i', 0)
        self.optimizer_lock = multiprocessing.Lock()
        episodes_per_process = int(self.config.num_episodes_to_run / self.worker_processes) + 1
        processes = []
        self.actor_critic.share_memory()
        self.actor_critic_optimizer.share_memory()

        optimizer_worker = multiprocessing.Process(target=self.update_shared_model, args=(gradient_updates_queue,))
        optimizer_worker.start()  # 启动总梯度更新主线程

        for process_num in range(self.worker_processes):
            worker = Actor_Critic_Worker(process_num, copy.deepcopy(self.environment), self.actor_critic, episode_number, self.optimizer_lock,
                                    self.actor_critic_optimizer, self.config, episodes_per_process,
                                    self.hyperparameters["epsilon_decay_rate_denominator"],
                                    self.action_size, self.action_types,
                                    results_queue, copy.deepcopy(self.actor_critic), gradient_updates_queue)
            worker.start()  # 启动run()函数
            processes.append(worker)
        self.print_results(episode_number, results_queue)
        for worker in processes:
            worker.join()  # 让子线程结束后主线程再结束
        optimizer_worker.kill()     # 主线程退出

        time_taken = time.time() - start
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def print_results(self, episode_number, results_queue):
        """Worker that prints out results as they get put into a queue"""
        while True:
            with episode_number.get_lock():
                carry_on = episode_number.value < self.config.num_episodes_to_run
            if carry_on:
                if not results_queue.empty():
                    self.total_episode_score_so_far = results_queue.get()
                    self.save_and_print_result()
            else:
                break

    def update_shared_model(self, gradient_updates_queue):
        """收到工作线程的梯度{信息传入队列}，更新全局网络梯度"""
        while True:
            gradients = gradient_updates_queue.get()
            with self.optimizer_lock:
                self.actor_critic_optimizer.zero_grad()
                for grads, params in zip(gradients, self.actor_critic.parameters()):
                    params._grad = grads  # maybe need to do grads.clone()
                self.actor_critic_optimizer.step()  # 依据新的梯度值更新参数

class Actor_Critic_Worker(torch.multiprocessing.Process):
    """Actor critic worker that will play the game for the designated number of episodes """
    def __init__(self, worker_num, environment, shared_model, counter, optimizer_lock, shared_optimizer,
                 config, episodes_to_run, epsilon_decay_denominator, action_size, action_types, results_queue,
                 local_model, gradient_updates_queue):
        super(Actor_Critic_Worker, self).__init__()
        self.noise = OU_Noise  # 连续动作的噪声
        self.environment = environment  # 环境对象
        self.config = config
        self.worker_num = worker_num
        self.gradient_clipping_norm = self.config.hyperparameters["gradient_clipping_norm"]
        self.discount_rate = self.config.hyperparameters["discount_rate"]
        self.normalise_rewards = self.config.hyperparameters["normalise_rewards"]
        self.action_size = action_size
        self.set_seeds(self.worker_num)
        self.shared_model = shared_model  # 全局网络
        self.local_model = local_model  # 线程局部网络
        self.local_optimizer = Adam(self.local_model.parameters(), lr=0.0, eps=1e-4)
        self.counter = counter
        self.optimizer_lock = optimizer_lock
        self.shared_optimizer = shared_optimizer
        self.episodes_to_run = episodes_to_run
        self.epsilon_decay_denominator = epsilon_decay_denominator
        self.exploration_worker_difference = self.config.hyperparameters["exploration_worker_difference"]
        self.action_types = action_types
        self.results_queue = results_queue
        self.episode_number = 0
        self.gradient_updates_queue = gradient_updates_queue
        self.episode_states = []  # 状态列表
        self.episode_actions = []  # 动作列表
        self.episode_rewards = []  # 回报列表
        self.episode_log_action_probabilities = []  # 动作log概率列表
        self.critic_outputs = []  # 评论家输出的V值列表

    def set_seeds(self, worker_num):
        """Sets random seeds for this worker"""
        torch.manual_seed(self.config.seed + worker_num)
        self.environment.seed(self.config.seed + worker_num)

    def run(self):
        """Starts the worker"""
        torch.set_num_threads(1)
        for ep_ix in range(self.episodes_to_run):
            with self.optimizer_lock:  # 锁定网络更新线程网络参数
                Base_Agent.copy_model_over(self.shared_model, self.local_model)
            epsilon_exploration = self.calculate_new_exploration()  # 计算新的探索参数
            state = self.reset_game_for_worker()  # 初始化状态
            done = False
            self.episode_states = []  # 状态列表
            self.episode_actions = []  # 动作列表
            self.episode_rewards = []  # 回报列表
            self.episode_log_action_probabilities = []  # 动作log概率列表
            self.critic_outputs = []  # 评论家输出的V值列表
            # 采样一条轨迹
            while not done:
                action, action_log_prob, critic_outputs = self.pick_action_and_get_critic_values(self.local_model, state, epsilon_exploration)
                next_state, reward, done, _ = self.environment.step(action)
                self.episode_states.append(state)
                self.episode_actions.append(action)
                self.episode_rewards.append(reward)
                self.episode_log_action_probabilities.append(action_log_prob)
                self.critic_outputs.append(critic_outputs)
                state = next_state

            total_loss = self.calculate_total_loss()
            self.put_gradients_in_queue(total_loss)
            self.episode_number += 1
            with self.counter.get_lock():
                self.counter.value += 1
                self.results_queue.put(np.sum(self.episode_rewards))  # 存储该轨迹上总的回报

    def calculate_new_exploration(self):
        """计算新的勘探参数。它在当前的上下3X范围内随机选取一个点"""
        with self.counter.get_lock():
            epsilon = 1.0 / (1.0 + (self.counter.value / self.epsilon_decay_denominator))
        epsilon = max(0.0, random.uniform(epsilon / self.exploration_worker_difference,
                                          epsilon * self.exploration_worker_difference))
        return epsilon

    def reset_game_for_worker(self):
        """重置游戏环境，这样就可以开始新一周期"""
        state = self.environment.reset()
        if self.action_types == "CONTINUOUS":
            self.noise.reset()
        return state

    def pick_action_and_get_critic_values(self, policy, state, epsilon_exploration=None):
        """使用策略选择一个动作"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        model_output = policy.forward(state)
        # 我们只使用第一列来决定动作，最后一列是状态值
        actor_output = model_output[:, list(range(self.action_size))]
        critic_output = model_output[:, -1]  # 评论家的V(s_t)_critic值
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)  # 动作分布实例
        action = action_distribution.sample().cpu().numpy()  # 采样一个动作
        if self.action_types == "CONTINUOUS":
            action += self.noise.sample()
        if self.action_types == "DISCRETE":
            if random.random() <= epsilon_exploration:
                action = random.randint(0, self.action_size - 1)
            else:
                action = action[0]
        action_log_prob = self.calculate_log_action_probability(action, action_distribution)
        return action, action_log_prob, critic_output

    def calculate_log_action_probability(self, actions, action_distribution):
        """计算所选动作的log概率"""
        policy_distribution_log_prob = action_distribution.log_prob(torch.Tensor([actions]))
        return policy_distribution_log_prob

    def calculate_total_loss(self):
        """Calculates the actor loss + critic loss"""
        discounted_returns = self.calculate_discounted_returns()
        if self.normalise_rewards:
            discounted_returns = self.normalise_discounted_returns(discounted_returns)
        critic_loss, advantages = self.calculate_critic_loss_and_advantages(discounted_returns)
        actor_loss = self.calculate_actor_loss(advantages)
        total_loss = actor_loss + critic_loss
        return total_loss

    def calculate_discounted_returns(self):
        """
        计算一集的累计折现收益，然后我们将在学习迭代中使用
        蒙特卡洛估计计算V(s)_target 值
        """
        discounted_returns = [0]
        for ix in range(len(self.episode_states)):
            return_value = self.episode_rewards[-(ix + 1)] + self.discount_rate*discounted_returns[-1]
            discounted_returns.append(return_value)
        discounted_returns = discounted_returns[1:]
        discounted_returns = discounted_returns[::-1]
        return discounted_returns

    def normalise_discounted_returns(self, discounted_returns):
        """通过除以该时段收益的均值和标准差，使贴现收益归一化"""
        mean = np.mean(discounted_returns)
        std = np.std(discounted_returns)
        discounted_returns -= mean
        discounted_returns /= (std + 1e-5)
        return discounted_returns

    def calculate_critic_loss_and_advantages(self, all_discounted_returns):
        """计算评论家的损失和优势"""
        critic_values = torch.cat(self.critic_outputs)
        advantages = torch.Tensor(all_discounted_returns) - critic_values  # 计算优势函数值|V(s_t)_target - V(s_t)_critic
        advantages = advantages.detach()
        critic_loss = (torch.Tensor(all_discounted_returns) - critic_values)**2
        critic_loss = critic_loss.mean()
        return critic_loss, advantages

    def calculate_actor_loss(self, advantages):
        """计算参与者的损失"""
        action_log_probabilities_for_all_episodes = torch.cat(self.episode_log_action_probabilities)
        actor_loss = -1.0 * action_log_probabilities_for_all_episodes * advantages
        actor_loss = actor_loss.mean()
        return actor_loss

    def put_gradients_in_queue(self, total_loss):
        """将梯度放入队列中，以供优化过程用于更新共享模型"""
        self.local_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.gradient_clipping_norm)  # 梯度裁剪
        gradients = [param.grad.clone() for param in self.local_model.parameters()]  # 线程子网络梯度
        self.gradient_updates_queue.put(gradients)  # 线程子网络梯度加入队列