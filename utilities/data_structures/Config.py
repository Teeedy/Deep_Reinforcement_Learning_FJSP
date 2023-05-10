
class Config():
    """对象来保存代理/游戏的配置要求"""
    def __init__(self):
        self.seed = None  # 随机种子
        self.environment = None  # 环境名称
        self.requirements_to_solve_game = None  # 配置需求
        self.num_episodes_to_run = None  # 运行周期
        self.file_to_save_data_results = None  # 保存结果数据的位置
        self.file_to_save_results_graph = None  # 保存结果图的位置
        self.use_GPU = None  # 是否使用GPU
        self.overwrite_existing_results_file = None  # 是否覆盖存在的结果文件
        self.save_model = False  # 是否保存模型
        self.hyper_parameters = None  # 算法超参数



