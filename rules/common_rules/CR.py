import matplotlib.pyplot as plt
import random
loss_history = []

for epoch in range(100):
    # 在每个epoch结束时记录损失值
    loss = epoch + random.randint(2, 10)  # 训练模型
    loss_history.append(loss)

    # 实时绘制损失值曲线图
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.pause(0.01)  # 暂停一小段时间，使图像更新可见
    plt.show()

