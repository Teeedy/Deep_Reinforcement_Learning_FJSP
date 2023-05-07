"""
matlab 引擎求解器
"""
import matlab, time
import matlab.engine

# 启动matlab引擎求解模型
data_list = []  # 输入数据为二维列表，或二维元组，或numpy 保证每行的元素相同可形成一维或二维矩阵格式
engine = matlab.engine.start_matlab()  # 启动引擎
fluid_solve = engine.line_cplex(matlab.double(data_list), matlab.double(data_list), nargout=2)  # 二维决策变量
engine.exit()  # 关闭引擎
