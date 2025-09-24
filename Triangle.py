import numpy as np
import matplotlib.pyplot as plt

def can_form_triangle(a, b):
    """判断给定折断点是否能形成三角形"""
    # 计算三段长度
    lengths = sorted([min(a, b), abs(a - b), 1 - max(a, b)])
    x, y, z = lengths
    
    # 检查三角不等式
    return (x + y > z) and (x + z > y) and (y + z > x)

# 蒙特卡洛模拟
def monte_carlo_simulation(n_samples=100000):
    count = 0
    for _ in range(n_samples):
        a, b = np.random.random(2)  # 随机两个折断点
        if can_form_triangle(a, b):
            count += 1
    return count / n_samples

# 理论计算函数
def theoretical_probability():
    """理论概率计算"""
    # 通过积分可得概率为 1/4
    return 0.25

# 运行模拟
prob_sim = monte_carlo_simulation()
prob_theory = theoretical_probability()

print(f"蒙特卡洛模拟概率: {prob_sim:.4f}")
print(f"理论概率: {prob_theory:.4f}")