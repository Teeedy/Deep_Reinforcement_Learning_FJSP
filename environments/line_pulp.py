# -*- coding: utf-8 -*-
from pulp import *

# Create a LP problem
prob = LpProblem("LP problem", LpMaximize)

# Define decision variables
x1 = LpVariable("x1", lowBound=0)
x2 = LpVariable("x2", lowBound=0)
t = LpVariable("t", lowBound=0)

# Define the objective function
prob += (2*x1 + 3*x2)/t, "Objective"

# Define the constraints
prob += 4*x1 + 5*x2 == t, "Denominator"

# Solve the problem
status = prob.solve()

# Print the optimal solution
print("Optimal Solution:")
print("x1 =", value(x1))
print("x2 =", value(x2))
print("t =", value(t))
print("Objective =", value(prob.objective))

from pulp import *

# 定义问题
prob = LpProblem("Maximize 1/(x+1)", LpMaximize)

# 定义决策变量
x = LpVariable("x", 10, None)

# 定义目标函数
prob += 1/(x+1)

# 添加约束条件
prob += x >= 10

# 求解问题
status = prob.solve()

# 打印结果
print(f"status: {LpStatus[status]}, x: {value(x)}, obj: {value(prob.objective)}")

