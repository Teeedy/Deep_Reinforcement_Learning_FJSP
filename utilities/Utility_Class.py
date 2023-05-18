"""
通用类
"""
import matplotlib.pyplot as plt

"""异常类"""
class MyError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

"""画图类"""
class FigGan():
    def __init__(self, object):
        self.kind_dict = object.kind_dict

    def figure(self):
        for kind, kind_object in self.kind_dict.items():
            for job_object in kind_object.job_arrive_list:
                for task_object in job_object.task_list:
                    machine = task_object.machine
                    plt.barh(machine, task_object.time_end - task_object.time_begin, left=task_object.time_begin,
                             height=0.4)
                    plt.text(task_object.time_begin, machine + 0.4,
                             '%s|%s|%s)' % (task_object.kind, task_object.number, task_object.task),
                             fontdict={'fontsize': 6})
        plt.show()


# 测试各类
if __name__ == '__main__':
    figure_object = FigGan

