function [H1, H2] = line_cplex(machine_list, class_list, rate_machine_class_index, class_number, class_number_time, number_task)
% 流体模型求解函数

m_number = size(machine_list,2);
k_number = size(class_list, 2);
%%% 定义决策变量 %%%
x = sdpvar(m_number, k_number, 'full');
%%% 目标函数 %%%
class_finish = [];
for k = 1:k_number
    class_finish = [class_finish, class_number(k)/sum(x(:,k).*rate_machine_class_index(:,k))];
end
z = max(class_finish);
%%% 添加约束 %%%
C = [];
% 决策变量范围约束
for m = 1:m_number
    for k = 1:k_number
        C = [C, 0<=x(m,k)<=1];
        if rate_machine_class_index(m,k)==0
            C = [C, x(m,k)==0];
        end
    end
end
% 机器利用率约束
for m = 1:m_number
    C = [C, sum(x(m,:))<=1];
end
% 生成非首工序类集合
kind_number = size(number_task, 2);  % 工件种类数
class_first_task = ones(1, kind_number);  % 首工序类数组
for j = 2:kind_number
    class_first_task(j) = sum(number_task(1,1:j-1)) + 1;
end
% 流体解可行性约束
for k = 1:k_number
    if class_number_time(k)~=0|ismember(k, class_first_task)
        continue
    else
        C = [C, sum(x(:,k).*rate_machine_class_index(:,k))<=sum(x(:,k-1).*rate_machine_class_index(:,k-1))];
    end
end
%%% 配置 %%%
ops = sdpsettings('verbose',0,'solver','cplex');
%%% 求解 %%%
reuslt = optimize(C,z);
H1 = value(x);
H2 = value(z);
% 判断是否求解成功
if reuslt.problem == 0 % problem =0 代表求解成功
    value(x)
    value(z)   
else
    disp('求解出错');
end

end

