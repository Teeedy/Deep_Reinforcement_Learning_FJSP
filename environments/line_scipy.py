from scipy.optimize import minimize
from numpy import ones

def objective(x):
    return (2+x[0])/(1+x[1]) - 3*x[0]+4*x[2]

LB=[0.1]*3; UB=[0.9]*3  # 变量范围
bound = tuple(zip(LB, UB))
cons = ({'type': 'ineq', 'fun': lambda x: 1.5 - sum([x[0], x[1], x[2]])})

solution = minimize(objective, ones(3), bounds=bound, constraints=cons)
print(solution)

