import numpy as np
import matplotlib.pyplot as plt
import math

# 定义Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义Sigmoid导数
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 创建数据
x = np.linspace(-10, 10, 1000)
y_sigmoid = sigmoid(x)
y_derivative = sigmoid_derivative(x)

# 创建图表
plt.figure(figsize=(15, 10))

# 1. Sigmoid函数曲线
plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, 'b-', linewidth=2, label='Sigmoid: σ(x) = 1/(1+e⁻ˣ)')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='y=0.5')
plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='x=0')
plt.title('Sigmoid Function Curve', fontsize=14)
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.grid(True, alpha=0.3)
plt.legend()

# 2. Sigmoid导数曲线
plt.subplot(2, 2, 2)
plt.plot(x, y_derivative, 'r-', linewidth=2, label="Sigmoid Derivative: σ'(x)")
plt.axhline(y=0.25, color='g', linestyle='--', alpha=0.7, label='Max derivative=0.25')
plt.axvline(x=0, color='g', linestyle='--', alpha=0.7, label='x=0')
plt.title('Sigmoid Derivative Curve', fontsize=14)
plt.xlabel('x')
plt.ylabel("σ'(x)")
plt.grid(True, alpha=0.3)
plt.legend()

# 3. 双坐标轴：函数值+导数
plt.subplot(2, 2, 3)
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(x, y_sigmoid, 'b-', linewidth=2, label='Sigmoid')
ax2.plot(x, y_derivative, 'r-', linewidth=2, label='Derivative')

ax1.set_xlabel('x')
ax1.set_ylabel('σ(x)', color='b')
ax2.set_ylabel("σ'(x)", color='r')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
plt.title('Sigmoid and its Derivative', fontsize=14)
plt.grid(True, alpha=0.3)

# 4. 关键点标注
plt.subplot(2, 2, 4)
plt.plot(x, y_sigmoid, 'b-', linewidth=2)

# 标注关键点
key_points = [-5, -2, 0, 2, 5]
for point in key_points:
    y_val = sigmoid(point)
    plt.plot(point, y_val, 'ro', markersize=8)
    plt.text(point + 0.5, y_val - 0.05, f'({point}, {y_val:.3f})', fontsize=10)

plt.title('Key Points on Sigmoid Curve', fontsize=14)
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig('sigmoid_function_and_derivative.png', dpi=300)