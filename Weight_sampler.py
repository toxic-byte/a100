import random
import bisect

class WeightedSampler:
    """
    使用二分查找进行高效权重采样
    """
    def __init__(self, items, weights):
        self.items = items
        # 计算前缀和数组
        self.prefix_sums = []
        current_sum = 0
        for w in weights:
            current_sum += w
            self.prefix_sums.append(current_sum)
        self.total_weight = current_sum

    def sample(self):
        # 生成随机数
        rand_num = random.uniform(0, self.total_weight)
        # 使用二分查找找到第一个大于等于 rand_num 的位置
        idx = bisect.bisect_left(self.prefix_sums, rand_num)
        return self.items[idx]

# 示例
fruits = ['苹果', '香蕉', '橙子']
weights = [10, 1, 1]

sampler = WeightedSampler(fruits, weights)

# 进行多次抽样，验证结果
results = {}
for _ in range(10000):
    choice = sampler.sample()
    results[choice] = results.get(choice, 0) + 1

print("抽样10000次的结果：", results)
