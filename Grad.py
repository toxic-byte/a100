import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100个0~10之间的随机数
y = 3 * X + 5 + np.random.randn(100, 1) * 2  # y = 3x + 5 + 噪声

# 2. 初始化参数
w = np.random.randn(1)  # 随机初始化权重
b = np.random.randn(1)  # 随机初始化偏置
learning_rate = 0.01
epochs = 1000

# 3. 梯度下降
loss_history = []
for epoch in range(epochs):
    # 前向传播计算预测值
    y_pred = w * X + b
    
    # 计算损失（均方误差）
    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)
    
    # 计算梯度（反向传播）
    dw = 2 * np.mean((y_pred - y) * X)  # dL/dw
    db = 2 * np.mean(y_pred - y)        # dL/db
    
    # 更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # 每100次打印一次损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, w: {w[0]:.4f}, b: {b[0]:.4f}")

# 4. 结果可视化
plt.figure(figsize=(12, 5))

# 数据点和拟合直线
plt.subplot(1, 2, 1)
plt.scatter(X, y, s=10, label="真实数据")
plt.plot(X, w * X + b, color='red', label="拟合直线")
plt.xlabel("X")
plt.ylabel("y")
plt.title("线性回归拟合")
plt.legend()

# 损失下降曲线
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("损失函数下降曲线")

plt.tight_layout()
plt.show()

# 打印最终参数
print(f"\n最终参数: w = {w[0]:.4f}, b = {b[0]:.4f}")